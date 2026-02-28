#!/usr/bin/env bash
set -euo pipefail

# One-command configure + build wrapper.
# Usage:
#   ./run_cmake.sh            # configure+build in ./build
#   ./run_cmake.sh build_cmake

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
fi

BUILD_DIR="${1:-build}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo "Build complete."
echo "Artifacts are in: ${BUILD_DIR}/"
