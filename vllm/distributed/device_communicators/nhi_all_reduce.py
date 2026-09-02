# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Narrow direct-NHI all-reduce backend for two-node gfx1151 decode.

This experimental backend intentionally supports one operation only: an
out-of-place BF16 sum of a contiguous ``[1, 4096]`` tensor across two ranks.
Every other tensor remains on the existing vLLM collective path.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ABI_VERSION = 1
_NUM_ELEMENTS = 4096


class _Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_path", ctypes.c_char_p),
        ("gpu_index", ctypes.c_int32),
        ("rank", ctypes.c_uint32),
        ("ring_size", ctypes.c_uint32),
        ("timeout_ms", ctypes.c_uint32),
        ("epoch", ctypes.c_uint64),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("exchanges", ctypes.c_uint64),
        ("tx_events", ctypes.c_uint64),
        ("rx_events", ctypes.c_uint64),
        ("timeouts", ctypes.c_uint64),
        ("protocol_errors", ctypes.c_uint64),
        ("last_sequence", ctypes.c_uint64),
        ("driver_event_drops", ctypes.c_uint64),
        ("driver_failures", ctypes.c_uint64),
        ("driver_last_error", ctypes.c_uint32),
        ("failed", ctypes.c_uint32),
    ]


class NhiAllReduce:
    def __init__(
        self,
        *,
        library_path: str,
        device_path: str,
        gpu_index: int,
        rank: int,
        epoch: int,
        timeout_ms: int = 1000,
    ) -> None:
        if rank not in (0, 1):
            raise ValueError(f"direct NHI requires rank 0 or 1, got {rank}")
        if epoch <= 0:
            raise ValueError("direct NHI epoch must be nonzero")
        if not Path(library_path).is_file():
            raise FileNotFoundError(f"direct NHI library not found: {library_path}")
        if not Path(device_path).exists():
            raise FileNotFoundError(f"direct NHI endpoint not found: {device_path}")

        self._lib = ctypes.CDLL(library_path)
        self._bind_abi()
        self._handle = ctypes.c_void_p()
        self._next_sequence = 1
        self.disabled = False

        device_bytes = device_path.encode()
        options = _Options(
            struct_size=ctypes.sizeof(_Options),
            abi_version=_ABI_VERSION,
            device_path=device_bytes,
            gpu_index=gpu_index,
            rank=rank,
            ring_size=4095,
            timeout_ms=timeout_ms,
            epoch=epoch,
        )
        rc = self._lib.vllm_nhi_ar_open(
            ctypes.byref(options), ctypes.byref(self._handle)
        )
        if rc != 0 or not self._handle.value:
            message = self._error_text(rc)
            self._handle = ctypes.c_void_p()
            raise RuntimeError(f"direct NHI open failed ({rc}): {message}")

    def _bind_abi(self) -> None:
        self._lib.vllm_nhi_ar_open.argtypes = [
            ctypes.POINTER(_Options),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.vllm_nhi_ar_open.restype = ctypes.c_int
        self._lib.vllm_nhi_ar_exchange_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self._lib.vllm_nhi_ar_exchange_bf16.restype = ctypes.c_int
        self._lib.vllm_nhi_ar_get_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Stats),
        ]
        self._lib.vllm_nhi_ar_get_stats.restype = ctypes.c_int
        self._lib.vllm_nhi_ar_last_error.argtypes = [ctypes.c_void_p]
        self._lib.vllm_nhi_ar_last_error.restype = ctypes.c_char_p
        self._lib.vllm_nhi_ar_strerror.argtypes = [ctypes.c_int]
        self._lib.vllm_nhi_ar_strerror.restype = ctypes.c_char_p
        self._lib.vllm_nhi_ar_close.argtypes = [ctypes.c_void_p]
        self._lib.vllm_nhi_ar_close.restype = ctypes.c_int

    @staticmethod
    def should_use_nhi_ar(input_: torch.Tensor) -> bool:
        if (
            not input_.is_cuda
            or input_.dtype != torch.bfloat16
            or tuple(input_.shape) != (1, _NUM_ELEMENTS)
            or tuple(input_.stride()) != (_NUM_ELEMENTS, 1)
            or not input_.is_contiguous()
        ):
            return False
        return not torch.cuda.is_current_stream_capturing()

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if not self.should_use_nhi_ar(input_):
            raise ValueError("tensor does not satisfy the direct-NHI all-reduce gate")
        if not self._handle.value:
            raise RuntimeError("direct NHI communicator is closed")

        output = torch.empty_like(input_)
        stream = torch.cuda.current_stream(input_.device)
        sequence = self._next_sequence
        rc = self._lib.vllm_nhi_ar_exchange_bf16(
            self._handle,
            ctypes.c_void_p(input_.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(stream.cuda_stream),
            ctypes.c_uint64(sequence),
        )
        if rc != 0:
            raise RuntimeError(
                f"direct NHI exchange {sequence} failed ({rc}): "
                f"{self._error_text(rc)}"
            )
        self._next_sequence += 1
        return output

    def stats(self) -> dict[str, int]:
        if not self._handle.value:
            return {}
        stats = _Stats(
            struct_size=ctypes.sizeof(_Stats),
            abi_version=_ABI_VERSION,
        )
        rc = self._lib.vllm_nhi_ar_get_stats(self._handle, ctypes.byref(stats))
        if rc != 0:
            raise RuntimeError(f"direct NHI stats failed ({rc}): {self._error_text(rc)}")
        return {
            name: int(getattr(stats, name))
            for name, _ in _Stats._fields_
            if name not in {"struct_size", "abi_version"}
        }

    def close(self) -> None:
        if not self._handle.value:
            return
        stats: dict[str, int] = {}
        try:
            stats = self.stats()
        finally:
            rc = self._lib.vllm_nhi_ar_close(self._handle)
            self._handle = ctypes.c_void_p()
        if rc != 0:
            raise RuntimeError(f"direct NHI close failed ({rc}): {self._error_text(rc)}")
        logger.info("Direct NHI all-reduce final stats: %s", stats)

    def _error_text(self, rc: int) -> str:
        raw = self._lib.vllm_nhi_ar_last_error(self._handle)
        if not raw:
            raw = self._lib.vllm_nhi_ar_strerror(rc)
        return raw.decode(errors="replace") if raw else "unknown error"
