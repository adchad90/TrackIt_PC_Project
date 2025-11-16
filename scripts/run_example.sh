#!/usr/bin/env bash
# example run script
# ./scripts/run_example.sh input.mp4 output.mp4

set -e
if [ $# -lt 2 ]; then
  echo "Usage: $0 input.mp4 output.mp4"
  exit 1
fi
INPUT="$1"
OUTPUT="$2"

mkdir -p build
pushd build
cmake ..
make -j
./tracker "../${INPUT}" "../${OUTPUT}" --template_crop 100 200 64 48
popd
