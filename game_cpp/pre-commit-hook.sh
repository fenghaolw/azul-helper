#!/bin/bash

# Git pre-commit hook to format C++ files with clang-format
# To install: cp pre-commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

echo "Running clang-format on staged C++ files..."

# Get the list of staged C++ files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|cc|h|hpp)$')

if [ -z "$STAGED_FILES" ]; then
    echo "No C++ files staged for commit."
    exit 0
fi

# Format each staged file
for FILE in $STAGED_FILES; do
    if [ -f "$FILE" ]; then
        echo "Formatting: $FILE"
        clang-format -i -style=file "$FILE"
        git add "$FILE"
    fi
done

echo "✅ All staged C++ files have been formatted and re-staged."
exit 0 