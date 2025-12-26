@echo off
REM Build and run GEMV benchmark (vs cuBLASLt)
REM Run from Windows Command Prompt

setlocal EnableDelayedExpansion

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to setup Visual Studio environment
    exit /b 1
)

REM Setup CUDA environment
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" (
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
    set SM_ARCH=120
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" (
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
    set SM_ARCH=86
) else (
    echo ERROR: CUDA not found
    exit /b 1
)

set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo ============================================
echo  GEMV Benchmark Build
echo ============================================
echo CUDA: %CUDA_PATH%
echo SM: %SM_ARCH%
echo.

REM Change to script directory
cd /d %~dp0

REM Build benchmark (linking cuBLASLt)
echo Building benchmark_gemv.cu...
nvcc -std=c++17 -O3 -arch=sm_%SM_ARCH% benchmark_gemv.cu -lcublasLt -o benchmark_gemv.exe
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)

echo.
echo Running benchmark...
echo.
"%~dp0benchmark_gemv.exe"

endlocal
