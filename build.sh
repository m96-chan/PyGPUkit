#!/bin/bash
# Build script for Git Bash
# Usage: ./build.sh [SM_VERSION] [CUDA_VERSION] [MODULE_SUFFIX]
#
# Examples:
#   ./build.sh 120a            # SM 120a, CUDA 13.1 (default)
#   ./build.sh 86              # SM 86, CUDA 13.1
#   ./build.sh 120a 13.1       # SM 120a, CUDA 13.1
#   ./build.sh 86 12.4         # SM 86, CUDA 12.4
#   ./build.sh 120a 13.1 _cu131 # SM 120a, CUDA 13.1, module suffix _cu131
#
# Supported SM versions: 80, 86, 89, 90, 100, 120a
# Note: RTX 5090 requires 120a (full accelerated features: tensor cores, block-scaled MMA)
# Supported CUDA versions: 12.4, 12.9, 13.1
# Module suffix: _cu129, _cu131, or empty for default name
#
# Build logs are saved to .claude/logs/build/

SM_VERSION=${1:-120a}
CUDA_VERSION=${2:-13.1}
MODULE_SUFFIX=${3:-}

# Setup logging
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/.claude/logs/build"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/build_sm${SM_VERSION}_cuda${CUDA_VERSION}_${TIMESTAMP}.log"

# Logging function - output to both console and log file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "=== PyGPUkit Build (Git Bash) ==="
log "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
log "SM Version: $SM_VERSION"
log "CUDA Version: $CUDA_VERSION"
log "Log File: $LOG_FILE"
if [ -n "$MODULE_SUFFIX" ]; then
    log "Module Suffix: $MODULE_SUFFIX"
fi

# Validate CUDA path exists
CUDA_PATH_CHECK="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}"
if [ ! -d "$CUDA_PATH_CHECK" ]; then
    log "ERROR: CUDA $CUDA_VERSION not found at $CUDA_PATH_CHECK"
    log "Available CUDA versions:"
    ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/"* 2>/dev/null | xargs -n1 basename | tee -a "$LOG_FILE"
    exit 1
fi
log ""

# Create a temporary batch file and execute it
TEMP_BAT=$(mktemp --suffix=.bat)
WIN_LOG=$(cygpath -w "$LOG_FILE")
cat > "$TEMP_BAT" << EOFBAT
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${CUDA_VERSION}
set PATH=%CUDA_PATH%\bin;%PATH%
set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
set CMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe
set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=${SM_VERSION}
set PYGPUKIT_MODULE_SUFFIX=${MODULE_SUFFIX}
pip install -e . --no-build-isolation 2>&1
EOFBAT

# Convert to Windows path and execute (capture output to log and console)
WIN_BAT=$(cygpath -w "$TEMP_BAT")
log "=== Build Output ==="
cmd //c "$WIN_BAT" 2>&1 | tee -a "$LOG_FILE"
RESULT=${PIPESTATUS[0]}

rm -f "$TEMP_BAT"
log ""

if [ $RESULT -eq 0 ]; then
    log "=== BUILD SUCCESS ==="
    log "Built with CUDA $CUDA_VERSION for SM $SM_VERSION"
    if [ -n "$MODULE_SUFFIX" ]; then
        log "Module: _pygpukit_native${MODULE_SUFFIX}"
    fi
    log "Log saved: $LOG_FILE"
else
    log "=== BUILD FAILED ==="
    log "Check log for details: $LOG_FILE"
    # Keep last 5 failed logs, clean older ones
    ls -t "$LOG_DIR"/build_*.log 2>/dev/null | tail -n +20 | xargs -r rm -f
    exit 1
fi

# Clean up old logs (keep last 10)
ls -t "$LOG_DIR"/build_*.log 2>/dev/null | tail -n +11 | xargs -r rm -f
