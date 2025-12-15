@echo off
REM Build PyGPUkit with CUDA 13.1 using Ninja generator
REM This script sets up VS environment for cl.exe and uses CUDA 13.1

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDA_PATH_V13_1=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo Building PyGPUkit with CUDA 13.1 (Ninja generator)...
echo CUDA_PATH=%CUDA_PATH%
echo.

pip install -e . --no-build-isolation -v
