#!/bin/bash

set -e  # Exit on any error

echo "🎮 Azul C++ Game Engine Test Suite"
echo "====================================="
echo

# Build directory
BUILD_DIR="build"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure and build
echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug

echo "Building all targets (including tests)..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo
echo "Running all test suites with CTest..."
echo "======================================"
ctest --output-on-failure

if [ $? -eq 0 ]; then
    echo
    echo "🎉 CTest: ALL TESTS PASSED! 🎉"
    exit 0
else
    echo
    echo "❌ CTest: SOME TESTS FAILED"
    exit 1
fi
