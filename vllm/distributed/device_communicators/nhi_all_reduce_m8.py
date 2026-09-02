# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Narrow direct-NHI M8 all-reduce backend for two-node gfx1151 decode.

This experimental backend intentionally supports one operation only: an
out-of-place BF16 sum of a contiguous ``[8, 4096]`` tensor across two ranks.
Every other tensor remains on the existing vLLM collective path.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_ABI_VERSION = 2
_NUM_ROWS = 8
_HIDDEN_SIZE = 4096


@eager_break_during_capture
def _breakable_exchange_into(
    communicator: "NhiAllReduceM8",
    input_: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    communicator._all_reduce_into(input_, output)
    return output


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


class NhiAllReduceM8:
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
        self._breakable_outputs: dict[int, torch.Tensor] = {}
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
        logger.info("Direct NHI M8 exchange format: %s", self._exchange_name)

    def _bind_abi(self) -> None:
        self._lib.vllm_nhi_ar_open.argtypes = [
            ctypes.POINTER(_Options),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.vllm_nhi_ar_open.restype = ctypes.c_int
        if hasattr(self._lib, "vllm_nhi_ar_exchange_bf16"):
            self._exchange = self._lib.vllm_nhi_ar_exchange_bf16
            self._exchange_name = "bf16"
        elif hasattr(self._lib, "vllm_nhi_ar_exchange_i8g32"):
            self._exchange = self._lib.vllm_nhi_ar_exchange_i8g32
            self._exchange_name = "i8g32"
        else:
            raise RuntimeError(
                "direct NHI M8 library exports neither the BF16 nor I8G32 "
                "exchange symbol"
            )
        self._exchange.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self._exchange.restype = ctypes.c_int
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
    def matches_nhi_ar(input_: torch.Tensor) -> bool:
        return bool(
            input_.is_cuda
            and input_.dtype == torch.bfloat16
            and tuple(input_.shape) == (_NUM_ROWS, _HIDDEN_SIZE)
            and tuple(input_.stride()) == (_HIDDEN_SIZE, 1)
            and input_.is_contiguous()
        )

    @classmethod
    def should_use_nhi_ar(cls, input_: torch.Tensor) -> bool:
        if not cls.matches_nhi_ar(input_):
            return False
        return not torch.cuda.is_current_stream_capturing()

    def _all_reduce_into(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if (
            not self.matches_nhi_ar(input_)
            or output.dtype != input_.dtype
            or output.shape != input_.shape
            or not output.is_contiguous()
        ):
            raise ValueError("tensor does not satisfy the direct-NHI all-reduce gate")
        if not self._handle.value:
            raise RuntimeError("direct NHI communicator is closed")

        stream = torch.cuda.current_stream(input_.device)
        sequence = self._next_sequence
        rc = self._exchange(
            self._handle,
            ctypes.c_void_p(input_.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(stream.cuda_stream),
            ctypes.c_uint64(sequence),
        )
        if rc != 0:
            raise RuntimeError(
                f"direct NHI {self._exchange_name} exchange {sequence} "
                f"failed ({rc}): "
                f"{self._error_text(rc)}"
            )
        self._next_sequence += 1

    def all_reduce_breakable(self, input_: torch.Tensor) -> torch.Tensor:
        if not self.matches_nhi_ar(input_):
            raise ValueError("tensor does not satisfy the direct-NHI all-reduce gate")
        if not BreakableCUDAGraphCapture.is_active():
            return self.all_reduce(input_)

        key = int(input_.data_ptr())
        output = self._breakable_outputs.get(key)
        if output is None:
            output = torch.empty_like(input_)
            self._breakable_outputs[key] = output
        return _breakable_exchange_into(self, input_, output)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if not self.should_use_nhi_ar(input_):
            raise ValueError("tensor does not satisfy the direct-NHI all-reduce gate")

        output = torch.empty_like(input_)
        self._all_reduce_into(input_, output)
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
        self._breakable_outputs.clear()
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
        logger.info("Direct NHI M8 all-reduce final stats: %s", stats)

    def _error_text(self, rc: int) -> str:
        raw = self._lib.vllm_nhi_ar_last_error(self._handle)
        if not raw:
            raw = self._lib.vllm_nhi_ar_strerror(rc)
        return raw.decode(errors="replace") if raw else "unknown error"
