#!/bin/bash
# Neural MCTS Training Script for Azul
# 
# This script builds and runs the neural MCTS training using OpenSpiel's
# AlphaZero implementation with ResNet-style neural networks.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
STEPS=1000
ACTORS=2
EVALUATORS=1
SIMULATIONS=400
MODEL="resnet"
WIDTH=128
DEPTH=6
LEARNING_RATE=0.001
BATCH_SIZE=32
DEVICE="auto"
CHECKPOINT_DIR="models/neural_mcts_azul"
USE_LIBTORCH=false
NO_EXPLICIT_LEARNING=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
show_help() {
    echo "Neural MCTS Training Script for Azul"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -s, --steps=N           Training steps (default: $STEPS)"
    echo "  -a, --actors=N          Actor threads (default: $ACTORS)"
    echo "  -e, --evaluators=N      Evaluator threads (default: $EVALUATORS)"
    echo "  -sim, --simulations=N   MCTS simulations per move (default: $SIMULATIONS)"
    echo "  -m, --model=TYPE        NN model: mlp|conv2d|resnet (default: $MODEL)"
    echo "  -w, --width=N           NN width (default: $WIDTH)"
    echo "  -d, --depth=N           NN depth (default: $DEPTH)"
    echo "  -lr, --learning-rate=F  Learning rate (default: $LEARNING_RATE)"
    echo "  -b, --batch=N           Batch size (default: $BATCH_SIZE)"
    echo "  --device=TYPE           Device: cpu|cuda|mps|auto (default: $DEVICE)"
    echo "  --dir=PATH              Checkpoint directory (default: $CHECKPOINT_DIR)"
    echo "  --libtorch              Use LibTorch AlphaZero (pure C++) instead of Python"
    echo "  --no-explicit-learning  Disable explicit learning (for single device setups)"
    echo "  --build-only            Only build, don't run training"
    echo "  --run-only              Only run training (skip build)"
    echo "  --clean                 Clean build before building"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Quick training with defaults"
    echo "  $0 --steps=5000 --actors=4           # Longer training with more actors"
    echo "  $0 --model=resnet --width=256 --depth=10  # Larger ResNet model"
    echo "  $0 --device=mps                      # Use Apple Silicon GPU"
    echo "  $0 --device=cuda                     # Use NVIDIA GPU"
    echo "  $0 --libtorch                        # Use pure C++ LibTorch implementation"
}

# Parse command line arguments
BUILD_ONLY=false
RUN_ONLY=false
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--steps=*)
            if [[ "$1" == *"="* ]]; then
                STEPS="${1#*=}"
            else
                STEPS="$2"
                shift
            fi
            ;;
        -a|--actors=*)
            if [[ "$1" == *"="* ]]; then
                ACTORS="${1#*=}"
            else
                ACTORS="$2"
                shift
            fi
            ;;
        -e|--evaluators=*)
            if [[ "$1" == *"="* ]]; then
                EVALUATORS="${1#*=}"
            else
                EVALUATORS="$2"
                shift
            fi
            ;;
        -sim|--simulations=*)
            if [[ "$1" == *"="* ]]; then
                SIMULATIONS="${1#*=}"
            else
                SIMULATIONS="$2"
                shift
            fi
            ;;
        -m|--model=*)
            if [[ "$1" == *"="* ]]; then
                MODEL="${1#*=}"
            else
                MODEL="$2"
                shift
            fi
            ;;
        -w|--width=*)
            if [[ "$1" == *"="* ]]; then
                WIDTH="${1#*=}"
            else
                WIDTH="$2"
                shift
            fi
            ;;
        -d|--depth=*)
            if [[ "$1" == *"="* ]]; then
                DEPTH="${1#*=}"
            else
                DEPTH="$2"
                shift
            fi
            ;;
        -lr|--learning-rate=*)
            if [[ "$1" == *"="* ]]; then
                LEARNING_RATE="${1#*=}"
            else
                LEARNING_RATE="$2"
                shift
            fi
            ;;
        -b|--batch=*)
            if [[ "$1" == *"="* ]]; then
                BATCH_SIZE="${1#*=}"
            else
                BATCH_SIZE="$2"
                shift
            fi
            ;;
        --device=*)
            DEVICE="${1#*=}"
            ;;
        --dir=*)
            CHECKPOINT_DIR="${1#*=}"
            ;;
        --libtorch)
            USE_LIBTORCH=true
            ;;
        --no-explicit-learning)
            NO_EXPLICIT_LEARNING=true
            ;;
        --build-only)
            BUILD_ONLY=true
            ;;
        --run-only)
            RUN_ONLY=true
            ;;
        --clean)
            CLEAN_BUILD=true
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# Print configuration
print_info "Neural MCTS Training Configuration:"
echo "  Implementation: $([ "$USE_LIBTORCH" = true ] && echo "LibTorch (C++)" || echo "OpenSpiel Python")"
echo "  Training steps: $STEPS"
echo "  Actors: $ACTORS, Evaluators: $EVALUATORS"
echo "  MCTS simulations: $SIMULATIONS"
echo "  Model: $MODEL (${WIDTH}x${DEPTH})"
echo "  Learning rate: $LEARNING_RATE, Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo ""

# Build phase
if [ "$RUN_ONLY" = false ]; then
    print_info "Building neural MCTS training executable..."
    
    # Clean build if requested
    if [ "$CLEAN_BUILD" = true ]; then
        print_info "Cleaning build directory..."
        rm -rf build/
        mkdir -p build
    fi
    
    # Ensure build directory exists
    mkdir -p build
    cd build
    
    # Configure and build
    if ! cmake .. > cmake.log 2>&1; then
        print_error "CMake configuration failed. Check cmake.log for details."
        exit 1
    fi
    
    # Use sysctl on macOS, nproc on Linux
    if command -v nproc >/dev/null 2>&1; then
        JOBS=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        JOBS=$(sysctl -n hw.ncpu)
    else
        JOBS=4
    fi
    
    # Build the appropriate target
    if [ "$USE_LIBTORCH" = true ]; then
        TARGET="libtorch_alphazero_trainer"
    else
        TARGET="neural_mcts_trainer"
    fi
    
    if ! make $TARGET -j$JOBS > make.log 2>&1; then
        print_error "Build failed for $TARGET. Check make.log for details."
        exit 1
    fi
    
    cd ..
    print_success "Build completed successfully!"
fi

# Run phase
if [ "$BUILD_ONLY" = false ]; then
    print_info "Starting neural MCTS training..."
    
    # Determine executable name
    if [ "$USE_LIBTORCH" = true ]; then
        EXECUTABLE="build/libtorch_alphazero_trainer"
        IMPLEMENTATION="LibTorch (C++)"
    else
        EXECUTABLE="build/neural_mcts_trainer"
        IMPLEMENTATION="OpenSpiel Python"
    fi
    
    # Check if executable exists
    if [ ! -f "$EXECUTABLE" ]; then
        print_error "Executable not found: $EXECUTABLE. Please build first."
        exit 1
    fi
    
    # Create checkpoint directory
    mkdir -p "$CHECKPOINT_DIR"
    
    # Prepare arguments
    ARGS="--steps=$STEPS"
    ARGS="$ARGS --actors=$ACTORS"
    ARGS="$ARGS --evaluators=$EVALUATORS"
    ARGS="$ARGS --simulations=$SIMULATIONS"
    ARGS="$ARGS --model=$MODEL"
    ARGS="$ARGS --width=$WIDTH"
    ARGS="$ARGS --depth=$DEPTH"
    ARGS="$ARGS --lr=$LEARNING_RATE"
    ARGS="$ARGS --batch=$BATCH_SIZE"
    ARGS="$ARGS --device=$DEVICE"
    ARGS="$ARGS --dir=$CHECKPOINT_DIR"
    
    # Add no-explicit-learning flag if requested
    if [ "$NO_EXPLICIT_LEARNING" = true ]; then
        ARGS="$ARGS --no-explicit-learning"
    fi
    
    print_info "Using: $IMPLEMENTATION"
    print_info "Running: ./$EXECUTABLE $ARGS"
    echo ""
    
    # Run the training
    if ! ./$EXECUTABLE $ARGS; then
        print_error "Training failed!"
        exit 1
    fi
    
    print_success "Training completed successfully!"
    print_info "Check $CHECKPOINT_DIR for trained models"
fi 