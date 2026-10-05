#!/bin/bash
# Builds the net-free MonoTrack line-detection binary. Needs OpenCV (any
# 4.x or 5.x) and a C++11 compiler; on HPC/module-based systems, load an
# opencv module first (e.g. `module load opencv/4.10.0`).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/src/build"
cd "$HERE/src/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target detect -j4
echo "Built: $HERE/src/build/detect"
