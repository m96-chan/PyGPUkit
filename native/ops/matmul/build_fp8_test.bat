@echo off
REM Build FP8 GEMM test with CUTLASS alignment patch
REM This tests if the alignment fix enables FP8 to work on SM120

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUTLASS_PATH=%SCRIPT_DIR%..\..\..\third_party\cutlass\include
set CUTLASS_TOOLS_PATH=%SCRIPT_DIR%..\..\..\third_party\cutlass\tools\util\include
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo Current directory: %CD%
echo CUTLASS path: %CUTLASS_PATH%
echo CUTLASS tools path: %CUTLASS_TOOLS_PATH%
echo.
echo Building test_fp8_patched.cu for SM120a (architecture-specific features)...
echo.

REM Use sm_120a to enable __CUDA_ARCH_FEAT_SM120_ALL macro
REM This is required for CUTLASS kernel selection (Issue #2902 workaround)
REM Add -DPYGPUKIT_DEBUG_LDSM to enable printf debugging in LDSM operations
nvcc -arch=sm_120a -std=c++17 -O3 ^
     -I"%CUTLASS_PATH%" ^
     -I"%CUTLASS_TOOLS_PATH%" ^
     -DCUTLASS_ARCH_MMA_SM120_SUPPORTED ^
     -DPYGPUKIT_DEBUG_LDSM ^
     --expt-relaxed-constexpr ^
     -Xcompiler "/Zc:preprocessor" ^
     -o test_fp8_patched.exe test_fp8_patched.cu

if errorlevel 1 (
    echo.
    echo Build failed!
    exit /b 1
)

echo.
echo Build succeeded!
echo.
echo Running test...
echo.
test_fp8_patched.exe
