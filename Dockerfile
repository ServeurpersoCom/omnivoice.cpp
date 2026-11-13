# =============================================================================
# omnivoice.cpp — tts-server on a Pascal GTX 1080 Ti (compute capability 6.1)
#
# Two-stage build:
#   1) "build"   : nvcc / CUDA 12.9 toolchain compiles the `tts-server` target.
#   2) "runtime" : slim CUDA *runtime* image that carries only the built binary
#                  + its dlopen'ed ggml backend .so files + the NVIDIA runtime
#                  libraries that ggml's CUDA backend needs at load time.
#
# Why CUDA 12.9: the newest CUDA that still ships a Pascal (sm_61) target.
# CUDA 13 dropped the 6.x / 7.0 architectures, so 12.9 is the ceiling for a
# 1080 Ti. We pin CMAKE_CUDA_ARCHITECTURES=61 (no fat binary for newer GPUs).
#
# The ggml backend .so files (e.g. libggml-cuda.so) are dlopen'ed at runtime by
# ggml_backend_load_all(); CMAKE_BUILD_RPATH=$ORIGIN makes them resolve from the
# binary's own directory, so the runtime image must keep them beside the
# executable. That is why the COPY below pulls the .so* artifacts with the binary.
#
# ggml is pulled from OUR fork (herr-schreiber/ggml) — see .gitmodules. The
# Docker build does a normal `git clone --recurse-submodules` of this repo, so
# it automatically checks out ggml at the fork (no separate `RUN git submodule`).
# =============================================================================

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build

ARG CMAKE_CUDA_ARCHITECTURES=61
ENV DEBIAN_FRONTEND=noninteractive

# Build toolchain. The devel image already ships nvcc + the CUDA runtime headers
# needed by the ggml CUDA backend, but the host still needs a C/C++ compiler,
# git (to materialise the ggml submodule) and cmake.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git ca-certificates cmake pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Source tree: the main repo plus the vendored cpp-httplib / yyjson (plain
# tracked files) and the ggml submodule.
COPY . /src

# Build only the OpenAI-compatible TTS server (and the shared omnivoice lib).
#   -DGGML_CUDA=ON                  : compile the CUDA backend (sm_61 below)
#   -DCMAKE_CUDA_ARCHITECTURES=61   : Pascal 1080 Ti only (no Turing+ code)
#   -DGGML_CUDA_NCCL=OFF            : Pascal needs no multi-GPU collectives
#   -DOMNIVOICE_SHARED=ON           : also emit libomnivoice.so (ABI consumers)
#   -DCMAKE_BUILD_RPATH=$ORIGIN     : ggml .so backends resolve from the app dir
#   --target tts-server             : stop after the one target we need to ship
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -DGGML_CUDA_NCCL=OFF \
        -DOMNIVOICE_SHARED=ON \
        '-DCMAKE_BUILD_RPATH=$ORIGIN' \
    && cmake --build build --target tts-server -j"$(nproc)"

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

# The CUDA *runtime* base image ships the CUDA *library* layer; the actual GPU
# driver (libcuda / libnvidia-*) is injected by the NVIDIA container runtime from
# the host, so there is nothing to install here to talk to the GPU.

ENV DEBIAN_FRONTEND=noninteractive
# Only extra runtime dependency is curl, used by the HEALTHCHECK below.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# tts-server + every .so it dlopen's / links against. The build root holds the
# binary and all backend .so files in one directory (CMAKE_*_OUTPUT_DIRECTORY),
# so a single wildcard brings the full set along.
COPY --from=build /src/build/tts-server   /app/tts-server
COPY --from=build /src/build/*.so*       /app/

EXPOSE 8080

# Liveness probe: GET /health -> {"status":"ok"} (see src/tts-server.h).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8080/health" || exit 1

# --model / --codec are the GGUFs mounted at /app/models (see compose.yaml).
# The entrypoint is the server itself; start it with, e.g.:
#   tts-server --model /app/models/omnivoice-base.gguf \
#              --codec /app/models/omnivoice-tokenizer.gguf --host 0.0.0.0
ENTRYPOINT ["/app/tts-server"]
