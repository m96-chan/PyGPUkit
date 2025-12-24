@echo off
REM Build FP8 BF16 GEMM test for SM120

setlocal

REM CUDA 13.1+ required for SM120
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=%CUDA_PATH%\bin;%PATH%

REM CUTLASS paths
set CUTLASS_DIR=..\..\..\third_party\cutlass
set CUTLASS_INCLUDE=%CUTLASS_DIR%\include
set CUTLASS_EXAMPLES=%CUTLASS_DIR%\examples\common

echo Building FP8 BF16 GEMM test for SM120...
echo CUDA: %CUDA_PATH%

nvcc -o test_fp8_bf16_sm120.exe test_fp8_bf16_sm120.cu ^
    -arch=sm_120a ^
    -I "%CUTLASS_INCLUDE%" ^
    -I "%CUTLASS_EXAMPLES%" ^
    -DCUTLASS_ARCH_MMA_SM120_SUPPORTED ^
    --expt-relaxed-constexpr ^
    /Zc:preprocessor ^
    -std=c++17 ^
    -O2

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo Run: test_fp8_bf16_sm120.exe
) else (
    echo Build failed with error %ERRORLEVEL%
)

endlocal
