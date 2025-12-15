@echo off
REM Build PyGPUkit with CUDA 12.4 using Ninja generator
REM This script sets up VS environment for cl.exe and uses CUDA 12.4

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set CudaToolkitDir=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo Building PyGPUkit with CUDA 12.4 (Ninja generator)...
echo CUDA_PATH=%CUDA_PATH%
echo.

pip install -e . --no-build-isolation -v
