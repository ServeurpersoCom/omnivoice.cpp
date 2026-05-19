#!/bin/bash

rm -rf build
mkdir build
cd build

cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build . --config Release -j "$(nproc)"
