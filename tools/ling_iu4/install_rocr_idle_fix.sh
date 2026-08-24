#!/usr/bin/env bash
set -Eeuo pipefail

rocm_systems_repo=https://github.com/ROCm/rocm-systems.git
rocm_systems_commit=44be71b52284948e58c93f65f46910399773fdcd

die() {
  printf 'install_rocr_idle_fix.sh: %s\n' "$*" >&2
  exit 2
}

absolute_path() {
  local path="$1"
  # The quoted patterns intentionally match literal shell-variable paths.
  # shellcheck disable=SC2016
  case "$path" in
    '~') path="$HOME" ;;
    [~]/*) path="$HOME/${path#\~/}" ;;
    '$HOME') path="$HOME" ;;
    '$HOME/'*) path="$HOME/${path#\$HOME/}" ;;
    '${HOME}') path="$HOME" ;;
    '${HOME}/'*) path="$HOME/${path#\$\{HOME\}/}" ;;
    '$PWD') path="$PWD" ;;
    '$PWD/'*) path="$PWD/${path#\$PWD/}" ;;
    '${PWD}') path="$PWD" ;;
    '${PWD}/'*) path="$PWD/${path#\$\{PWD\}/}" ;;
  esac
  case "$path" in
    *'$'*) die "unexpanded shell variable in path: $path" ;;
  esac
  if command -v realpath >/dev/null 2>&1; then
    realpath -m -- "$path"
  elif [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PWD" "${path#./}"
  fi
}

repo_root_arg="$PWD"
force=
case "$#" in
  0) ;;
  1)
    if [ "$1" = --force ]; then
      force=--force
    else
      [ -n "$1" ] || die 'repository root cannot be empty'
      repo_root_arg="$1"
    fi
    ;;
  2)
    [ -n "$1" ] || die 'repository root cannot be empty'
    [ "$2" = --force ] || die "unknown option: $2"
    repo_root_arg="$1"
    force=--force
    ;;
  *) die 'usage: install_rocr_idle_fix.sh [REPO_ROOT] [--force]' ;;
esac

repo_root="$(absolute_path "$repo_root_arg")"
case "$repo_root" in
  *[[:space:]:]*)
    die "repository root cannot contain whitespace or ':' because ROCr is loaded through LD_PRELOAD: $repo_root"
    ;;
esac
venv="$repo_root/.venv"
source_root="$(absolute_path "${CIRU_ROCR_SOURCE_DIR:-$repo_root/.deps/rocr-idle-fix/source}")"
build_dir="$(absolute_path "${CIRU_ROCR_BUILD_DIR:-$repo_root/.deps/rocr-idle-fix/build}")"
override_dir="$repo_root/runtime-overrides/rocr-7.15-gcc-idle"
runtime_name=libhsa-runtime64.so.1.21.0
runtime_path="$override_dir/$runtime_name"
provenance="$override_dir/BUILD-PROVENANCE.txt"

if [ "$(uname -s)" != Linux ]; then
  die 'the ROCr idle-runtime build requires Linux'
fi
[ -x "$venv/bin/python" ] || die "runtime Python is missing: $venv/bin/python"

if [ "$force" != --force ] && [ -r "$runtime_path" ] && \
   [ -r "$override_dir/SHA256SUMS" ] && [ -r "$provenance" ] && \
   grep -q "^rocm_systems_commit=$rocm_systems_commit$" "$provenance" && \
   (cd "$override_dir" && sha256sum --check --status SHA256SUMS); then
  printf 'ROCr idle-runtime override is already installed and verified: %s\n' \
    "$runtime_path"
  exit 0
fi

host_id="$(
  # shellcheck disable=SC1091
  source /etc/os-release 2>/dev/null || true
  printf '%s\n' "${ID:-unknown}"
)"
if [ "$host_id" = nixos ] && \
   { ! command -v pkg-config >/dev/null 2>&1 || \
     ! command -v xxd >/dev/null 2>&1; }; then
  command -v nix-shell >/dev/null 2>&1 || \
    die 'NixOS build needs nix-shell to provide pkg-config and xxd'
  [ "${CIRU_NIX_ROCR_DEPS_ACTIVE:-0}" != 1 ] || \
    die 'NixOS dependency shell did not provide pkg-config and xxd'
  printf -v rerun_command '%q ' env CIRU_NIX_ROCR_DEPS_ACTIVE=1 \
    bash "${BASH_SOURCE[0]}" "$repo_root_arg"
  if [ "$force" = --force ]; then
    printf -v rerun_command '%s%q ' "$rerun_command" --force
  fi
  printf 'Entering an ephemeral Nix shell for ROCr host dependencies.\n'
  exec nix-shell -p pkg-config xxd libdrm elfutils numactl openssl \
    --run "$rerun_command"
fi

for command_name in git gcc g++ pkg-config sha256sum xxd; do
  command -v "$command_name" >/dev/null 2>&1 || \
    die "missing $command_name; install the documented ROCr build dependencies"
done

site="$("$venv/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
sdk="$site/_rocm_sdk_devel"
core="$site/_rocm_sdk_core"
[ -x "$sdk/llvm/bin/clang" ] || \
  die "private ROCm SDK is missing clang: $sdk/llvm/bin/clang"
[ -r "$sdk/lib/cmake/clang/ClangConfig.cmake" ] || \
  die 'private ROCm SDK is missing Clang CMake metadata'
[ -r "$sdk/lib/cmake/llvm/LLVMConfig.cmake" ] || \
  die 'private ROCm SDK is missing LLVM CMake metadata'
[ -x "$venv/bin/cmake" ] || die "cmake is missing from $venv"
[ -x "$venv/bin/ninja" ] || die "ninja is missing from $venv"

mkdir -p "$(dirname -- "$source_root")" "$build_dir" "$override_dir"
source_checkout_created=0
if [ ! -d "$source_root/.git" ]; then
  git clone --filter=blob:none --no-checkout "$rocm_systems_repo" "$source_root"
  source_checkout_created=1
fi

current_commit="$(git -C "$source_root" rev-parse HEAD 2>/dev/null || true)"
if [ "$current_commit" != "$rocm_systems_commit" ]; then
  source_status="$(git -C "$source_root" status --porcelain 2>/dev/null || true)"
  source_worktree_entry="$(
    find "$source_root" -mindepth 1 -maxdepth 1 ! -name .git -print -quit
  )"
  if [ -n "$source_status" ] && [ "$source_checkout_created" != 1 ] && \
     [ -n "$source_worktree_entry" ]; then
    die "refusing to replace a modified ROCr source checkout: $source_root"
  fi
  git -C "$source_root" fetch --depth=1 origin "$rocm_systems_commit"
  if [ "$source_checkout_created" = 1 ] || [ -z "$source_worktree_entry" ]; then
    git -C "$source_root" switch --discard-changes --detach "$rocm_systems_commit"
  else
    git -C "$source_root" switch --detach "$rocm_systems_commit"
  fi
fi
test "$(git -C "$source_root" rev-parse HEAD)" = "$rocm_systems_commit"

# NixOS has no /bin/bash. Keep this generated-code helper portable without
# changing the runtime source or behavior.
helper="$source_root/projects/rocr-runtime/runtime/hsa-runtime/image/blit_src/create_hsaco_ascii_file.sh"
"$venv/bin/python" - "$helper" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "#!/bin/bash -e\n"
new = "#!/usr/bin/env -S bash -e\n"
if text.startswith(old):
    path.write_text(new + text[len(old):])
elif not text.startswith(new):
    raise SystemExit(f"unexpected helper shebang: {path}")
PY

export PATH="$sdk/bin:$sdk/llvm/bin:$venv/bin:$PATH"
export LD_LIBRARY_PATH="$sdk/lib:$core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cc="${CC:-gcc}"
cxx="${CXX:-g++}"
max_jobs="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')}"
host_cmake_prefix="${CMAKE_PREFIX_PATH:-}"
host_cmake_prefix="${host_cmake_prefix//:/;}"
cmake_prefix="$sdk;$core${host_cmake_prefix:+;$host_cmake_prefix}"

if [ "$host_id" = nixos ]; then
  glibc_runtime_lib="$(dirname -- "$(gcc -print-file-name=crt1.o)")"
  [ -f "$glibc_runtime_lib/libc.so" ] || \
    die "cannot resolve the NixOS libc linker input: $glibc_runtime_lib/libc.so"
  nix_library_path="$(
    printf '%s\n' "${NIX_LDFLAGS:-}" | tr ' ' '\n' |
      sed -n 's/^-L\(\/nix\/store\/.*\)$/\1/p' | paste -sd: -
  )"
  nix_include_path="$(
    printf '%s\n' "${NIX_CFLAGS_COMPILE:-}" | tr ' ' '\n' |
      awk 'previous == "-isystem" && /^\/nix\/store\// { print }
           { previous=$0 }' | paste -sd: -
  )"
  [ -n "$nix_library_path" ] || \
    die 'NixOS dependency shell did not expose library search paths'
  [ -n "$nix_include_path" ] || \
    die 'NixOS dependency shell did not expose include search paths'
  export CMAKE_LIBRARY_PATH="$glibc_runtime_lib:$nix_library_path${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
  export CMAKE_INCLUDE_PATH="$nix_include_path${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
  export LIBRARY_PATH="$glibc_runtime_lib:$nix_library_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export CPATH="$nix_include_path${CPATH:+:$CPATH}"
  export LD_LIBRARY_PATH="$glibc_runtime_lib:$nix_library_path:$LD_LIBRARY_PATH"
fi

printf 'Building pinned ROCr idle-runtime override\n'
printf '  source: %s @ %s\n  compiler: %s / %s\n' \
  "$rocm_systems_repo" "$rocm_systems_commit" "$cc" "$cxx"
cmake_args=(
  -S "$source_root/projects/rocr-runtime"
  -B "$build_dir"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER="$cc"
  -DCMAKE_CXX_COMPILER="$cxx"
  -DCMAKE_PREFIX_PATH="$cmake_prefix"
  -DClang_DIR="$sdk/lib/cmake/clang"
  -DLLVM_DIR="$sdk/lib/cmake/llvm"
)
if [ -n "${CMAKE_LIBRARY_PATH:-}" ]; then
  cmake_args+=("-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH//:/;}")
fi
if [ -n "${CMAKE_INCLUDE_PATH:-}" ]; then
  cmake_args+=("-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH//:/;}")
fi
"$venv/bin/cmake" "${cmake_args[@]}"
"$venv/bin/cmake" --build "$build_dir" --target hsa-runtime64 --parallel "$max_jobs"

built_runtime="$build_dir/rocr/lib/$runtime_name"
[ -r "$built_runtime" ] || die "build did not produce $built_runtime"
install -m 0755 "$built_runtime" "$runtime_path.new"
mv -f "$runtime_path.new" "$runtime_path"
(
  cd "$override_dir"
  sha256sum "$runtime_name" >SHA256SUMS
)
{
  printf 'rocm_systems_repo=%s\n' "$rocm_systems_repo"
  printf 'rocm_systems_commit=%s\n' "$rocm_systems_commit"
  printf 'runtime_name=%s\n' "$runtime_name"
  printf 'runtime_sha256=%s\n' "$(sha256sum "$runtime_path" | awk '{print $1}')"
  printf 'cc=%s\n' "$("$cc" --version | head -n 1)"
  printf 'cxx=%s\n' "$("$cxx" --version | head -n 1)"
} >"$provenance"
(cd "$override_dir" && sha256sum --check SHA256SUMS)

printf 'ROCr idle-runtime override installed: %s\n' "$runtime_path"
printf 'The CIRU Ling IU4 launcher verifies and preloads it automatically.\n'
