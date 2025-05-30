#!/bin/bash
# Setup script to create a python alias for systems that only have python3
# This is optional - the project now uses python3 explicitly where needed

echo "Setting up python alias for this session..."

# Check if python3 exists
if command -v python3 &> /dev/null; then
    echo "✓ python3 found"
    
    # Check if python already exists
    if command -v python &> /dev/null; then
        echo "✓ python already exists"
        python --version
    else
        echo "Creating python alias for this session..."
        alias python=python3
        echo "✓ python alias created (python -> python3)"
        echo "To make this permanent, add 'alias python=python3' to your ~/.bashrc or ~/.zshrc"
    fi
else
    echo "❌ python3 not found. Please install Python 3."
    exit 1
fi

echo ""
echo "You can now use either 'python' or 'python3'" 