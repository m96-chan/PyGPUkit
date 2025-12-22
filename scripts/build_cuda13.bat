@echo off
REM Build PyGPUkit with CUDA 13.1
REM Run this from Windows Command Prompt (not Git Bash)
REM
REM Usage:
REM   build_cuda13.bat          - Build for all SM (80, 86, 89, 90, 100, 120)
REM   build_cuda13.bat 86       - Build for SM 86 only (RTX 3090 Ti)
REM   build_cuda13.bat 89       - Build for SM 89 only (RTX 4090)
REM   build_cuda13.bat 90       - Build for SM 90 only (H100)
REM   build_cuda13.bat 100      - Build for SM 100 only (Blackwell datacenter)
REM   build_cuda13.bat 120      - Build for SM 120 only (RTX 5090)

setlocal EnableDelayedExpansion

REM Parse SM argument
set SM_ARG=%1
if "%SM_ARG%"=="" (
    set SM_ARCH=80;86;89;90;100;120
    set SM_DESC=all (80, 86, 89, 90, 100, 120)
) else (
    set SM_ARCH=%SM_ARG%
    set SM_DESC=%SM_ARG%
)

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: Failed to setup Visual Studio environment
    exit /b 1
)

REM Setup CUDA 13.1 environment
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDA_PATH_V13_1=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%
set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
set CMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe

REM Verify environment
echo.
echo ============================================
echo  PyGPUkit Build with CUDA 13.1
echo ============================================
echo.
echo CUDA_PATH: %CUDA_PATH%
echo CUDACXX: %CUDACXX%
echo SM Target: %SM_DESC%
echo.

where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found in PATH
    exit /b 1
)

echo NVCC version:
nvcc --version
echo.

where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found - VS environment not set up correctly
    exit /b 1
)

echo CL version:
cl 2>&1 | findstr "Version"
echo.

REM Clean previous build cache (optional, uncomment if needed)
REM if exist build rmdir /s /q build

REM Build with CMAKE_ARGS to override SM architecture
echo Starting build...
echo.
set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=%SM_ARCH%
pip install -e . --no-build-isolation

if errorlevel 1 (
    echo.
    echo ============================================
    echo  BUILD FAILED
    echo ============================================
    exit /b 1
)

echo.
echo ============================================
echo  BUILD SUCCESSFUL
echo ============================================

endlocal
