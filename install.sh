#!/usr/bin/env bash
set -euo pipefail

original_dir="$(pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
fi

if command -v git >/dev/null 2>&1; then
  git submodule update --init --recursive third-party/cutlass third-party/fmt
fi

resolve_cutlass_include() {
  local -a candidates=(
    "${CUTLASS_INCLUDE_DIR:-}"
    "${script_dir}/third-party/cutlass/include"
    "${script_dir}/build/_deps/cutlass-src/include"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -n "${c}" && -d "${c}/cutlass" && -d "${c}/cute" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

mkdir -p gdn_cuda/include
cutlass_include="$(resolve_cutlass_include || true)"
if [[ -z "${cutlass_include}" ]]; then
  echo "Error: could not locate CUTLASS include dir with cutlass/ and cute/ subdirs." >&2
  echo "Set CUTLASS_INCLUDE_DIR=/path/to/cutlass/include and rerun." >&2
  exit 1
fi

ln -sfn "${cutlass_include}/cutlass" gdn_cuda/include/cutlass
ln -sfn "${cutlass_include}/cute" gdn_cuda/include/cute
echo "Linked CUTLASS headers from: ${cutlass_include}"

# Wipe old artifacts to avoid stale extension/binary mismatches.
rm -rf build build_* dist
rm -rf ./*.egg-info
rm -f ./*.so

if command -v uv >/dev/null 2>&1; then
  uv run python setup.py bdist_wheel
  uv pip install dist/*.whl --reinstall
else
  python setup.py bdist_wheel
  python -m pip install dist/*.whl --force-reinstall
fi

cd "${original_dir}"
