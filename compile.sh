#!/bin/bash
set -e

# Auto-detect CUDA from MATLAB if nvcc not on PATH
if ! command -v nvcc &> /dev/null; then
    MATLAB_CUDA=$(ls -d /usr/local/dcs/versions/MATLAB/*/sys/cuda/glnxa64/cuda 2>/dev/null | tail -1)
    if [ -n "$MATLAB_CUDA" ]; then
        export CUDA_HOME="$MATLAB_CUDA"
        export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        echo "Using CUDA from: $CUDA_HOME"
    else
        echo "ERROR: nvcc not found. Install CUDA toolkit or set CUDA_HOME."
        exit 1
    fi
fi

# Get pybind11 and python include flags
PY_INCLUDES=$(python3 -m pybind11 --includes)
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/mnistocr${EXT_SUFFIX}"

echo "Compiling step 1: .cu -> .o"
nvcc -c --compiler-options -fPIC -std=c++17 -O2 \
    -I "$DIR/src" \
    -I "$CUDA_HOME/include" \
    $PY_INCLUDES \
    "$DIR/src/logistic.cu" -o /tmp/logistic.o

nvcc -c --compiler-options -fPIC -std=c++17 -O2 \
    -I "$DIR/src" \
    -I "$CUDA_HOME/include" \
    $PY_INCLUDES \
    "$DIR/src/bindings.cu" -o /tmp/bindings.o

echo "Compiling step 2: link .o -> .so"
g++ -shared \
    /tmp/logistic.o /tmp/bindings.o \
    -L "$CUDA_HOME/lib64" -lcudart_static \
    -o "$OUT"

rm -f /tmp/logistic.o /tmp/bindings.o

# Symlink for simple import
ln -sf "$(basename "$OUT")" "$DIR/mnistocr.so" 2>/dev/null || true

echo "Built: $OUT"
echo "Run: python3 python/cli.py"
