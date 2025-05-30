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
    else:
        print("No CUDA GPU available.")

    # Check MPS (Apple Silicon) availability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    else:
        print("MPS (Apple Silicon) not available.")

    # Run unified performance comparison
    test_gpu_vs_cpu_performance()


def test_gpu_vs_cpu_performance():
    """Test neural network performance: CPU vs available accelerated device (GPU/MPS)."""
    print("\n" + "=" * 60)
    print("DEVICE PERFORMANCE COMPARISON")
    print("=" * 60)

    # Create dummy game state for testing
    from game.game_state import GameState

    dummy_state = GameState(num_players=2, seed=42)
    num_tests = 20

    # Test CPU performance first
    print("Testing CPU performance...")
    nn_cpu = AzulNeuralNetwork(config_name="small", device="cpu")

    start_time = time.time()
    for _ in range(num_tests):
        _ = nn_cpu.evaluate(dummy_state)
    cpu_time = time.time() - start_time

    print(
        f"CPU: {num_tests} evaluations in {cpu_time:.3f}s ({cpu_time/num_tests*1000:.1f}ms per eval)"
    )

    # Test accelerated device if available
    accelerated_device = None
    accelerated_time = None

    if torch.cuda.is_available():
        accelerated_device = "cuda"
        print(f"\nTesting GPU (CUDA) performance...")
        nn_gpu = AzulNeuralNetwork(config_name="small", device="cuda")

        # Warm up GPU
        for _ in range(3):
            _ = nn_gpu.evaluate(dummy_state)

        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_tests):
            _ = nn_gpu.evaluate(dummy_state)
        torch.cuda.synchronize()
        accelerated_time = time.time() - start_time

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerated_device = "mps"
        print(f"\nTesting MPS (Apple Silicon) performance...")
        nn_mps = AzulNeuralNetwork(config_name="small", device="mps")

        # Warm up MPS
        for _ in range(3):
            _ = nn_mps.evaluate(dummy_state)

        start_time = time.time()
        for _ in range(num_tests):
            _ = nn_mps.evaluate(dummy_state)
        accelerated_time = time.time() - start_time

    # Show comparison results
    if accelerated_device and accelerated_time:
        device_name = (
            "GPU (CUDA)" if accelerated_device == "cuda" else "MPS (Apple Silicon)"
        )
        print(
            f"{device_name}: {num_tests} evaluations in {accelerated_time:.3f}s ({accelerated_time/num_tests*1000:.1f}ms per eval)"
        )

        speedup = cpu_time / accelerated_time
        print(f"\n🚀 PERFORMANCE COMPARISON:")
        print(
            f"   CPU time:     {cpu_time:.3f}s ({cpu_time/num_tests*1000:.1f}ms per eval)"
        )
        print(
            f"   {device_name} time: {accelerated_time:.3f}s ({accelerated_time/num_tests*1000:.1f}ms per eval)"
        )
        print(f"   Speedup:      {speedup:.1f}x faster")

        if speedup > 2.0:
            print(
                "   ✅ Excellent acceleration - strongly recommend using accelerated device"
            )
        elif speedup > 1.5:
            print("   ✅ Good acceleration - recommend using accelerated device")
        elif speedup > 1.2:
            print(
                "   ⚠️  Modest acceleration - accelerated device provides some benefit"
            )
        else:
            print("   ⚠️  Minimal acceleration - CPU performance is comparable")
    else:
        print(f"\n⚠️  No accelerated device available - running on CPU only")
        print(
            "   Consider using a machine with GPU or Apple Silicon for better performance"
        )


def test_mps_performance():
    """This function is now integrated into test_gpu_vs_cpu_performance()."""
    pass


def main():
    """Main entry point."""
    test_gpu_availability()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if torch.cuda.is_available():
        print("✅ CUDA GPU detected")
        print("   Run profiling with: --device cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) detected")
        print("   Run profiling with: --device mps")
    else:
        print("⚠️  No GPU acceleration available")
        print(
            "   Will use CPU - consider using a machine with GPU for better performance"
        )

    print("\nNext steps:")
    print("1. Install profiling dependencies: pip install -r requirements-dev.txt")
    print("2. Run the performance comparison above to see actual speedup")
    print("3. Fix game state copying if it's a bottleneck")
    print("4. Use the faster device for neural network training and evaluation")
    print(
        "5. Consider implementing batch neural network evaluation for further speedup"
    )


if __name__ == "__main__":
    main()
