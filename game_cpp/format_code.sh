#!/bin/bash

# Format all C++ files using clang-format with Google style
# This script will format .cpp, .cc, .h, and .hpp files

echo "Formatting C++ files with Google C++ Style Guide..."

# Find and format all C++ files
find . -name "*.cpp" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" | while read file; do
    echo "Formatting: $file"
    clang-format -i -style=file "$file"
done

echo "Done! All C++ files have been formatted." 