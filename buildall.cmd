@echo off

call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

rd /s /q build 2>nul
mkdir build 2>nul
cd build

cmake .. -G "Ninja Multi-Config" -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
