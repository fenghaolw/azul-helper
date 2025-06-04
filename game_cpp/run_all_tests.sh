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

echo "Building all tests..."
make -j$(nproc)

echo
echo "Running all test suites..."
echo "========================="

# Test executables
TESTS=(
    "test_basic_functionality"
    "test_legal_moves" 
    "test_game_flow"
    "test_scoring"
    "test_edge_cases"
)

# Track results
PASSED=0
FAILED=0
START_TIME=$(date +%s)

for test in "${TESTS[@]}"; do
    echo
    echo "============================================================"
    echo "Running: $test"
    echo "============================================================"
    
    if ./"$test"; then
        echo "✅ $test PASSED"
        ((PASSED++))
    else
        echo "❌ $test FAILED"
        ((FAILED++))
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo
echo "============================================================"
echo "TEST SUMMARY"
echo "============================================================"
echo "Total test suites: ${#TESTS[@]}"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total time: ${DURATION}s"

if [ $FAILED -eq 0 ]; then
    echo
    echo "🎉 ALL TESTS PASSED! 🎉"
    echo "The C++ Azul game engine is working correctly!"
    exit 0
else
    echo
    echo "❌ SOME TESTS FAILED"
    echo "Please check the errors above and fix the issues."
    exit 1
fi 