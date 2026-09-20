#!/bin/bash

rm -rf build
mkdir build
cd build

cmake .. -DGGML_VULKAN=ON
cmake --build . --config Release -j "$(getconf _NPROCESSORS_ONLN)"
