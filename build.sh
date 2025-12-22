#!/bin/bash
# Build script for Git Bash
# Usage: ./build.sh [SM_VERSION] [CUDA_VERSION]
#
# Examples:
#   ./build.sh 86           # SM 86, CUDA 13.1 (default)
#   ./build.sh 120          # SM 120, CUDA 13.1
#   ./build.sh 120 12.9     # SM 120, CUDA 12.9
#   ./build.sh 86 12.4      # SM 86, CUDA 12.4
#
# Supported SM versions: 80, 86, 89, 90, 100, 120
# Supported CUDA versions: 12.4, 12.9, 13.1

SM_VERSION=${1:-86}
CUDA_VERSION=${2:-13.1}

echo "=== PyGPUkit Build (Git Bash) ==="
echo "SM Version: $SM_VERSION"
echo "CUDA Version: $CUDA_VERSION"

# Validate CUDA path exists
CUDA_PATH_CHECK="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}"
if [ ! -d "$CUDA_PATH_CHECK" ]; then
    echo "ERROR: CUDA $CUDA_VERSION not found at $CUDA_PATH_CHECK"
    echo "Available CUDA versions:"
    ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/"* 2>/dev/null | xargs -n1 basename
    exit 1
fi

# Create a temporary batch file and execute it
TEMP_BAT=$(mktemp --suffix=.bat)
cat > "$TEMP_BAT" << EOFBAT
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${CUDA_VERSION}
set PATH=%CUDA_PATH%\bin;%PATH%
set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
set CMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe
set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=${SM_VERSION}
pip install -e . --no-build-isolation
EOFBAT

# Convert to Windows path and execute
WIN_BAT=$(cygpath -w "$TEMP_BAT")
cmd //c "$WIN_BAT"
RESULT=$?

rm -f "$TEMP_BAT"

if [ $RESULT -eq 0 ]; then
    echo "=== BUILD SUCCESS ==="
    echo "Built with CUDA $CUDA_VERSION for SM $SM_VERSION"
else
    echo "=== BUILD FAILED ==="
    exit 1
fi
