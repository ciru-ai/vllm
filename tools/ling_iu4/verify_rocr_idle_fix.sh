#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
pid="${1:-}"

if [[ -z "$pid" ]]; then
  engine_pids=()
  for comm_path in /proc/[0-9]*/comm; do
    [[ -r "$comm_path" ]] || continue
    IFS= read -r comm <"$comm_path" || continue
    [[ "$comm" == VLLM::EngineCor* ]] || continue
    candidate_pid="${comm_path#/proc/}"
    engine_pids+=("${candidate_pid%/comm}")
  done
  if (( ${#engine_pids[@]} != 1 )); then
    printf 'Expected exactly one EngineCore process; found %d. Pass its PID explicitly.\n' \
      "${#engine_pids[@]}" >&2
    exit 2
  fi
  pid="${engine_pids[0]}"
fi

[[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/maps" ]] || {
  echo "EngineCore PID is not readable: $pid" >&2
  exit 2
}

runtime_dir="$(
  tr '\0' '\n' <"/proc/$pid/environ" |
    sed -n 's/^ROCR_IDLE_RUNTIME_DIR=//p' | head -n 1
)"
runtime_dir="${runtime_dir:-$repo_root/runtime-overrides/rocr-7.15-gcc-idle}"
expected_runtime="$runtime_dir/libhsa-runtime64.so.1.21.0"
[[ -r "$expected_runtime" ]] || {
  echo "Corrected ROCr runtime is not readable: $expected_runtime" >&2
  exit 2
}
expected_hash="$(sha256sum "$expected_runtime" | awk '{print $1}')"

mapfile -t mapped_runtimes < <(
  awk '/libhsa-runtime64/ {print $6}' "/proc/$pid/maps" | sort -u
)
(( ${#mapped_runtimes[@]} > 0 )) || {
  echo "FAIL: EngineCore has no mapped libhsa-runtime64" >&2
  exit 1
}

matched=0
for mapped_runtime in "${mapped_runtimes[@]}"; do
  mapped_hash=unreadable
  if [[ -r "$mapped_runtime" ]]; then
    mapped_hash="$(sha256sum "$mapped_runtime" | awk '{print $1}')"
  fi
  printf 'EngineCore ROCr: %s (%s)\n' "$mapped_runtime" "$mapped_hash"
  [[ "$mapped_hash" == "$expected_hash" ]] && matched=1
done

if (( matched == 0 )); then
  printf 'FAIL: EngineCore is not using corrected ROCr checksum %s\n' \
    "$expected_hash" >&2
  exit 1
fi
printf 'ROCr mapping: PASS (%s)\n' "$expected_hash"

"${PYTHON_BIN:-python}" - "$pid" <<'PY'
from pathlib import Path
import os
import sys
import time

pid = int(sys.argv[1])
sample_seconds = float(os.environ.get("IDLE_CPU_SAMPLE_SECONDS", "3"))
clock_ticks = os.sysconf("SC_CLK_TCK")


def sample_threads():
    values = {}
    for stat_path in Path(f"/proc/{pid}/task").glob("*/stat"):
        try:
            text = stat_path.read_text()
            close = text.rfind(")")
            comm = text[text.find("(") + 1:close]
            fields = text[close + 2:].split()
            values[int(stat_path.parent.name)] = (
                comm,
                int(fields[11]) + int(fields[12]),
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return values


before = sample_threads()
started = time.monotonic()
time.sleep(sample_seconds)
elapsed = time.monotonic() - started
after = sample_threads()
cpu = []
for tid, (comm, ticks_after) in after.items():
    if tid not in before:
        continue
    ticks_before = before[tid][1]
    percent = (ticks_after - ticks_before) / clock_ticks / elapsed * 100.0
    cpu.append((percent, tid, comm))
cpu.sort(reverse=True)
aggregate = sum(row[0] for row in cpu)
hottest = cpu[0] if cpu else (0.0, pid, "unknown")
print(f"Idle CPU sample: aggregate={aggregate:.2f}% "
      f"hottest={hottest[0]:.2f}% tid={hottest[1]} ({hottest[2]}) "
      f"window={elapsed:.2f}s")
if aggregate >= 10.0 or hottest[0] >= 5.0:
    raise SystemExit(
        "FAIL: idle CPU exceeds the release threshold; make sure no request "
        "or compilation is active"
    )
print("Idle CPU: PASS")
PY
