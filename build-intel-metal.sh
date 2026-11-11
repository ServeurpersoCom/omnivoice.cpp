#!/usr/bin/env bash
# Explicit experimental Intel Radeon build; leaves default builds unchanged.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != x86_64 ]]; then
    echo 'This experimental recipe requires an Intel Mac.' >&2
    exit 1
fi
git submodule update --init ggml
if git -C ggml apply --check ../patches/ggml-intel-metal.patch 2>/dev/null; then
    git -C ggml apply ../patches/ggml-intel-metal.patch
elif ! git -C ggml apply --reverse --check ../patches/ggml-intel-metal.patch 2>/dev/null; then
    echo 'GGML patch does not apply cleanly; refusing to alter this checkout.' >&2
    exit 1
fi
cmake -S . -B build-intel-metal -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF \
    -DOMNIVOICE_TEST_METAL_TRANSFERS=ON
cmake --build build-intel-metal --target omnivoice-tts test-metal-unaligned -j "${OV_BUILD_JOBS:-4}"
./build-intel-metal/test-metal-unaligned
