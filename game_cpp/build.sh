#!/bin/bash

# Azul C++ Build Script
echo "Building Azul C++ implementation..."

# Create build directory
mkdir -p build
cd build

echo "Configuring with CMake..."

# Determine pybind11 path automatically if in virtual environment
PYBIND11_CMAKE_ARGS=""
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
    PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
    if [[ -n "$PYBIND11_DIR" ]]; then
        echo "Found pybind11 at: $PYBIND11_DIR"
        PYBIND11_CMAKE_ARGS="-Dpybind11_DIR=$PYBIND11_DIR"
    else
        echo "Warning: pybind11 not found in virtual environment"
    fi
fi

# Check for OpenSpiel (optional)
if [[ -n "$OPENSPIEL_ROOT" ]]; then
    echo "Building with OpenSpiel support: $OPENSPIEL_ROOT"
    cmake $PYBIND11_CMAKE_ARGS -DOPENSPIEL_ROOT="$OPENSPIEL_ROOT" ..
else
    echo "Building without OpenSpiel support"
    cmake $PYBIND11_CMAKE_ARGS ..
fi

if [[ $? -ne 0 ]]; then
    echo "CMake configuration failed!"
    exit 1
fi

echo "Building..."
# Use all available cores for parallel build
if command -v nproc >/dev/null 2>&1; then
    # Linux
    CORES=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    # macOS
    CORES=$(sysctl -n hw.ncpu)
else
    # Fallback
    CORES=4
fi

make -j$CORES

if [[ $? -ne 0 ]]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"
echo "Executables are in the build/ directory"
echo "Python bindings: azul_cpp_bindings.cpython-*.so"

# Run tests
echo "Running tests..."
if ctest --output-on-failure; then
    echo "✓ CTest: All tests passed!"
else
    echo "⚠ CTest: Tests failed." >&2
    exit 1
fi

echo ""
echo "To use the C++ implementation in Python:"
echo "1. Make sure the build directory is in your Python path"
echo "2. Import: from game_cpp.azul_cpp_wrapper import *"
echo ""
echo "To run performance benchmark:"
echo "python -c \"from game_cpp.azul_cpp_wrapper import benchmark_performance; print(benchmark_performance())\"" 