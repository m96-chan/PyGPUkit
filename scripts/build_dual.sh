#!/bin/bash
# Build PyGPUkit with both CUDA 12.9 and CUDA 13.1 native modules
# Usage: ./scripts/build_dual.sh [SM_VERSION]
#
# This creates both _pygpukit_native_cu129.pyd and _pygpukit_native_cu131.pyd
# for automatic driver-based selection at runtime.
#
# Examples:
#   ./scripts/build_dual.sh          # Build for SM 86 (RTX 3090 Ti)
#   ./scripts/build_dual.sh 120      # Build for SM 120 (RTX 5090)

SM_VERSION=${1:-86}

echo "=== PyGPUkit Dual CUDA Build ==="
echo "SM Version: $SM_VERSION"
echo ""

# Check if CUDA versions are available
CUDA_129_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
CUDA_131_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"

HAS_129=false
HAS_131=false

if [ -d "$CUDA_129_PATH" ]; then
    echo "Found CUDA 12.9 at $CUDA_129_PATH"
    HAS_129=true
else
    echo "WARNING: CUDA 12.9 not found"
fi

if [ -d "$CUDA_131_PATH" ]; then
    echo "Found CUDA 13.1 at $CUDA_131_PATH"
    HAS_131=true
else
    echo "WARNING: CUDA 13.1 not found"
fi

if [ "$HAS_129" = false ] && [ "$HAS_131" = false ]; then
    echo "ERROR: No CUDA toolkit found. Install CUDA 12.9 or 13.1."
    exit 1
fi

echo ""

# Function to build with specific CUDA version
build_cuda() {
    local CUDA_VERSION=$1
    local MODULE_SUFFIX=$2

    echo "=== Building with CUDA $CUDA_VERSION (suffix: $MODULE_SUFFIX) ==="

    TEMP_BAT=$(mktemp --suffix=.bat)
    cat > "$TEMP_BAT" << EOFBAT
@echo off
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1
set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v${CUDA_VERSION}
set PATH=%CUDA_PATH%\\bin;%PATH%
set CUDACXX=%CUDA_PATH%\\bin\\nvcc.exe
set CMAKE_CUDA_COMPILER=%CUDA_PATH%\\bin\\nvcc.exe
set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=${SM_VERSION}
set PYGPUKIT_MODULE_SUFFIX=${MODULE_SUFFIX}
set PYGPUKIT_DISABLE_CUTLASS=1
pip install -e . --no-build-isolation
EOFBAT

    WIN_BAT=$(cygpath -w "$TEMP_BAT")
    cmd //c "$WIN_BAT"
    RESULT=$?
    rm -f "$TEMP_BAT"

    if [ $RESULT -ne 0 ]; then
        echo "=== Build failed for CUDA $CUDA_VERSION ==="
        return 1
    fi

    echo "=== Build successful for CUDA $CUDA_VERSION ==="
    return 0
}

# Clean previous builds
echo "Cleaning previous build..."
rm -rf build/ 2>/dev/null

# Build CUDA 12.9 version
if [ "$HAS_129" = true ]; then
    build_cuda "12.9" "_cu129"
    if [ $? -ne 0 ]; then
        echo "CUDA 12.9 build failed!"
        exit 1
    fi
fi

# Clean build directory between versions
rm -rf build/ 2>/dev/null

# Build CUDA 13.1 version
if [ "$HAS_131" = true ]; then
    build_cuda "13.1" "_cu131"
    if [ $? -ne 0 ]; then
        echo "CUDA 13.1 build failed!"
        exit 1
    fi
fi

echo ""
echo "=== DUAL BUILD COMPLETE ==="
echo "Built modules:"
ls -la src/pygpukit/_pygpukit_native*.pyd 2>/dev/null || echo "(check install location)"
