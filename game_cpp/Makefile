# Makefile for Azul C++ project

.PHONY: all build clean format test

# Default build target
all: build

# Build the project
build:
	@mkdir -p build
	@cd build && cmake .. && make

# Clean build artifacts
clean:
	@rm -rf build

# Format all C++ files using clang-format with Google style
format:
	@echo "Formatting C++ files with Google C++ Style Guide..."
	@find . -name "*.cpp" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" | while read file; do \
		echo "Formatting: $$file"; \
		clang-format -i -style=file "$$file"; \
	done
	@echo "Done! All C++ files have been formatted."

# Check formatting without modifying files
format-check:
	@echo "Checking C++ formatting..."
	@find . -name "*.cpp" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" | while read file; do \
		if ! clang-format -style=file "$$file" | diff -q "$$file" - > /dev/null; then \
			echo "❌ $$file is not formatted correctly"; \
			exit 1; \
		else \
			echo "✅ $$file"; \
		fi; \
	done
	@echo "All files are properly formatted!"

# Run the evaluation demo
demo: build
	@cd build && ./azul_evaluation_demo

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the project (default)"
	@echo "  build       - Build the project"
	@echo "  clean       - Clean build artifacts"
	@echo "  format      - Format all C++ files using Google style"
	@echo "  format-check - Check if files are properly formatted"
	@echo "  demo        - Run the evaluation demo"
	@echo "  help        - Show this help message" 