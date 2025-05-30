#!/usr/bin/env python3
"""
Test GPU availability and run basic GPU profiling.
"""

import os
import sys
import time

import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.neural_network import AzulNeuralNetwork


def test_gpu_availability():
    """Test GPU availability and performance."""
    print("=" * 60)
    print("GPU AVAILABILITY TEST")
    print("=" * 60)

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        # Test GPU vs CPU performance
        test_gpu_vs_cpu_performance()
    else:
        print("No GPU available. Running on CPU only.")
        print(
            "Consider using Google Colab or a machine with GPU for better performance."
        )

    # Check MPS (Apple Silicon) availability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
        test_mps_performance()


def test_gpu_vs_cpu_performance():
    """Test neural network performance on GPU vs CPU."""
    print("\n" + "=" * 60)
    print("GPU vs CPU PERFORMANCE TEST")
    print("=" * 60)

    # Create small neural network for testing
    print("Creating neural network...")

    # Test on CPU
    print("\nTesting CPU performance...")
    nn_cpu = AzulNeuralNetwork(config_name="small", device="cpu")

    # Create dummy game state for testing
    from game.game_state import GameState

    dummy_state = GameState(num_players=2, seed=42)

    # Time CPU inference
    start_time = time.time()
    num_tests = 10
    for _ in range(num_tests):
        _ = nn_cpu.evaluate(dummy_state)
    cpu_time = time.time() - start_time

    print(
        f"CPU: {num_tests} evaluations in {cpu_time:.3f}s ({cpu_time/num_tests*1000:.1f}ms per eval)"
    )

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\nTesting GPU performance...")
        nn_gpu = AzulNeuralNetwork(config_name="small", device="cuda")

        # Warm up GPU
        for _ in range(3):
            _ = nn_gpu.evaluate(dummy_state)

        # Time GPU inference
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_tests):
            _ = nn_gpu.evaluate(dummy_state)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        print(
            f"GPU: {num_tests} evaluations in {gpu_time:.3f}s ({gpu_time/num_tests*1000:.1f}ms per eval)"
        )

        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.1f}x faster than CPU")

        if speedup > 1.5:
            print("✅ GPU provides significant speedup - use GPU for profiling")
        else:
            print("⚠️  GPU speedup is minimal - CPU might be sufficient")


def test_mps_performance():
    """Test MPS (Apple Silicon) performance."""
    print("\n" + "=" * 60)
    print("MPS (APPLE SILICON) PERFORMANCE TEST")
    print("=" * 60)

    # Create neural network for MPS
    print("Testing MPS performance...")
    nn_mps = AzulNeuralNetwork(config_name="small", device="mps")

    # Create dummy game state
    from game.game_state import GameState

    dummy_state = GameState(num_players=2, seed=42)

    # Time MPS inference
    start_time = time.time()
    num_tests = 10
    for _ in range(num_tests):
        _ = nn_mps.evaluate(dummy_state)
    mps_time = time.time() - start_time

    print(
        f"MPS: {num_tests} evaluations in {mps_time:.3f}s ({mps_time/num_tests*1000:.1f}ms per eval)"
    )


def main():
    """Main entry point."""
    test_gpu_availability()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if torch.cuda.is_available():
        print("✅ Use GPU for neural network inference")
        print("   Run profiling with: --device cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Use MPS (Apple Silicon) for neural network inference")
        print("   Run profiling with: --device mps")
    else:
        print("⚠️  No GPU acceleration available")
        print("   Consider using a machine with GPU for better performance")

    print("\nNext steps:")
    print("1. Install profiling dependencies: pip install -r requirements-dev.txt")
    print("2. Fix game state copying (biggest bottleneck)")
    print("3. Re-run profiling with GPU if available")
    print("4. Implement batch neural network evaluation")


if __name__ == "__main__":
    main()
