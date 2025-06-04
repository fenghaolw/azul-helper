#!/bin/bash

# Build script for Azul C++ implementation

set -e  # Exit on any error

echo "Building Azul C++ implementation..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
if [ -n "$OPENSPIEL_ROOT" ]; then
    echo "Using OpenSpiel at: $OPENSPIEL_ROOT"
    cmake -DOPENSPIEL_ROOT="$OPENSPIEL_ROOT" ..
else
    echo "Building without OpenSpiel support"
    cmake ..
fi

# Build
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests
echo "Running tests..."
if [ -f "./test_basic" ]; then
    ./test_basic
    echo "✓ All tests passed!"
else
    echo "⚠ Test executable not found"
fi

echo "Build completed successfully!"
echo ""
echo "To use the C++ implementation in Python:"
echo "1. Make sure the build directory is in your Python path"
echo "2. Import: from game_cpp.azul_cpp_wrapper import *"
echo ""
echo "To run performance benchmark:"
echo "python -c \"from game_cpp.azul_cpp_wrapper import benchmark_performance; print(benchmark_performance())\"" 