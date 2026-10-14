#!/bin/sh
# Build the OpenCASCADE static libraries refusion links against (~10 min on 16 cores).
#   OCCT_PREFIX  install location   (default: <repo>/third_party/occt)
#   OCCT_WORK    source/build dir   (default: <repo>/third_party)
# Used directly for native builds and by the Dockerfile.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${OCCT_VERSION:-7_8_1}"
PREFIX="${OCCT_PREFIX:-$ROOT/third_party/occt}"
WORK="${OCCT_WORK:-$ROOT/third_party}"
mkdir -p "$WORK/src" "$WORK/build-occt"
cd "$WORK/src"
[ -d "OCCT-$VER" ] || curl -sSL "https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V$VER.tar.gz" | tar xz
cd "$WORK/build-occt"
cmake "../src/OCCT-$VER" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_LIBRARY_TYPE=Static -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_MODULE_Draw=OFF -DBUILD_MODULE_Visualization=OFF -DBUILD_MODULE_ApplicationFramework=OFF \
  -DBUILD_MODULE_DETools=OFF -DBUILD_DOC_Overview=OFF \
  -DUSE_FREETYPE=OFF -DUSE_TK=OFF -DUSE_TCL=OFF -DUSE_OPENGL=OFF -DUSE_XLIB=OFF -DUSE_RAPIDJSON=OFF \
  -DUSE_DRACO=OFF -DUSE_TBB=OFF -DUSE_VTK=OFF -DUSE_FREEIMAGE=OFF -DUSE_FFMPEG=OFF
make -j"$(nproc)"
cmake --install .
