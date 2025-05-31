"""
Comprehensive performance profiler for Azul self-play.

This module provides detailed timing and profiling capabilities for:
- Neural Network forward pass timing
- Game operations (get_legal_actions, apply_action, clone)
- MCTS simulation loop profiling
- GPU utilization monitoring
- Python hotspot identification
"""

import cProfile
import functools
import io
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class ProfilerStats:
    """Container for profiling statistics."""

    def __init__(self):
        self.timing_data: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        self.memory_usage: Dict[str, Union[float, np.floating[Any]]] = {}
        self.gpu_stats: Dict[str, Any] = {}

    def add_timing(self, name: str, duration: float):
        """Add a timing measurement."""
        if name not in self.timing_data:
            self.timing_data[name] = []
            self.call_counts[name] = 0
        self.timing_data[name].append(duration)
        self.call_counts[name] += 1

    def add_memory_usage(self, name: str, memory_mb: float):
        """Add memory usage measurement."""
        self.memory_usage[name] = memory_mb

    def add_gpu_stats(self, name: str, gpu_data: Dict[str, Any]):
        """Add GPU statistics."""
        self.gpu_stats[name] = gpu_data

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive timing summary."""
        summary = {}

        for name, times in self.timing_data.items():
            summary[name] = {
                "total_time": sum(times),
                "avg_time": np.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "std_time": np.std(times),
                "call_count": self.call_counts[name],
                "times_per_second": float(
                    self.call_counts[name] / sum(times) if sum(times) > 0 else 0
                ),
            }

        summary["memory_usage"] = self.memory_usage
        summary["gpu_stats"] = self.gpu_stats
        return summary

    def print_summary(self):
        """Print a formatted summary of profiling results."""
        print("\n" + "=" * 80)
        print("AZUL SELF-PLAY PERFORMANCE PROFILING RESULTS")
        print("=" * 80)

        summary = self.get_summary()

        # Sort by total time to show biggest bottlenecks first
        timing_items = [
            (name, data)
            for name, data in summary.items()
            if isinstance(data, dict) and "total_time" in data
        ]
        timing_items.sort(key=lambda x: x[1]["total_time"], reverse=True)

        print("\nTIMING BREAKDOWN (sorted by total time):")
        print("-" * 80)
        print(
            f"{'Operation':<30} {'Total(s)':<10} {'Avg(ms)':<10} {'Count':<8} {'Calls/s':<10}"
        )
        print("-" * 80)

        for name, data in timing_items:
            print(
                f"{name:<30} {data['total_time']:<10.3f} "
                f"{data['avg_time']*1000:<10.2f} {data['call_count']:<8} "
                f"{data['times_per_second']:<10.1f}"
            )

        # Memory usage
        if summary.get("memory_usage"):
            print("\nMEMORY USAGE:")
            print("-" * 40)
            for name, memory_mb in summary["memory_usage"].items():
                print(f"{name:<30} {memory_mb:.2f} MB")

        # GPU stats
        if summary.get("gpu_stats"):
            print("\nGPU STATISTICS:")
            print("-" * 40)
            for name, gpu_data in summary["gpu_stats"].items():
                print(f"{name}:")
                for key, value in gpu_data.items():
                    print(f"  {key}: {value}")


class AzulProfiler:
    """Main profiler class for Azul self-play analysis."""

    def __init__(
        self, enable_memory_profiling: bool = True, enable_gpu_profiling: bool = True
    ):
        self.stats = ProfilerStats()
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_gpu_profiling = enable_gpu_profiling
        self._cprofile_data: Optional[cProfile.Profile] = None

        if enable_memory_profiling:
            tracemalloc.start()

    @contextmanager
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        start_time = time.perf_counter()

        # Memory tracking
        memory_before = None
        if self.enable_memory_profiling:
            current, peak = tracemalloc.get_traced_memory()
            memory_before = current / 1024 / 1024  # Convert to MB

        # GPU memory tracking
        gpu_memory_before = None
        if self.enable_gpu_profiling and torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.stats.add_timing(name, duration)

            # Memory tracking
            if self.enable_memory_profiling and memory_before is not None:
                current, peak = tracemalloc.get_traced_memory()
                memory_after = current / 1024 / 1024
                memory_delta = memory_after - memory_before
                self.stats.add_memory_usage(f"{name}_memory_delta", memory_delta)

            # GPU memory tracking
            if (
                self.enable_gpu_profiling
                and torch.cuda.is_available()
                and gpu_memory_before is not None
            ):
                gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_delta = gpu_memory_after - gpu_memory_before
                self.stats.add_gpu_stats(
                    name,
                    {
                        "gpu_memory_before_mb": gpu_memory_before,
                        "gpu_memory_after_mb": gpu_memory_after,
                        "gpu_memory_delta_mb": gpu_memory_delta,
                    },
                )

    def profile_function(self, name: str):
        """Decorator for timing function calls."""

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.time_operation(name):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator

    def start_cprofile(self):
        """Start cProfile for detailed Python profiling."""
        self._cprofile_data = cProfile.Profile()
        self._cprofile_data.enable()

    def stop_cprofile(self) -> str:
        """Stop cProfile and return formatted results."""
        if self._cprofile_data is None:
            return "cProfile was not started"

        self._cprofile_data.disable()

        # Create string buffer for output
        output_buffer = io.StringIO()
        ps = pstats.Stats(self._cprofile_data, stream=output_buffer)
        ps.sort_stats("cumulative")
        ps.print_stats(30)  # Top 30 functions

        result = output_buffer.getvalue()
        output_buffer.close()

        self._cprofile_data = None
        return result

    def profile_gpu_utilization(self) -> Dict[str, Any]:
        """Profile GPU utilization if available."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        gpu_stats = {
            "gpu_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }

        # Try to get additional GPU info if available
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            gpu_stats.update(
                {
                    "gpu_utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(
                        handle
                    ).gpu,
                    "memory_utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(
                        handle
                    ).memory,
                    "temperature_c": pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power_usage_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                }
            )
        except ImportError:
            gpu_stats["pynvml_available"] = False
        except Exception as e:
            gpu_stats["pynvml_error"] = str(e)

        return gpu_stats

    def get_summary(self) -> Dict[str, Any]:
        """Get complete profiling summary."""
        return self.stats.get_summary()

    def print_summary(self):
        """Print formatted profiling results."""
        self.stats.print_summary()

        # Add GPU utilization summary
        gpu_util = self.profile_gpu_utilization()
        if gpu_util["gpu_available"]:
            print("\nCURRENT GPU STATUS:")
            print("-" * 40)
            print(f"Device: {gpu_util['device_name']}")
            print(f"Memory Allocated: {gpu_util['memory_allocated_mb']:.2f} MB")
            print(f"Memory Reserved: {gpu_util['memory_reserved_mb']:.2f} MB")
            print(f"Max Memory Used: {gpu_util['max_memory_allocated_mb']:.2f} MB")

            if "gpu_utilization_percent" in gpu_util:
                print(f"GPU Utilization: {gpu_util['gpu_utilization_percent']}%")
                print(f"Memory Utilization: {gpu_util['memory_utilization_percent']}%")
                print(f"Temperature: {gpu_util['temperature_c']}°C")
                print(f"Power Usage: {gpu_util['power_usage_w']:.1f} W")
        else:
            print("\nGPU: Not available")


# Convenience functions for creating timed versions of key game operations
def create_profiled_game_state(profiler: AzulProfiler):
    """Create a GameState class with profiled methods."""
    from game.game_state import GameState as OriginalGameState

    class ProfiledGameState(OriginalGameState):
        def get_legal_actions(self, player_id=None):
            with profiler.time_operation("game.get_legal_actions"):
                return super().get_legal_actions(player_id)

        def apply_action(self, action):
            with profiler.time_operation("game.apply_action"):
                return super().apply_action(action)

        def copy(self):
            with profiler.time_operation("game.copy"):
                return super().copy()

        def is_action_legal(self, action, player_id=None):
            with profiler.time_operation("game.is_action_legal"):
                return super().is_action_legal(action, player_id)

    return ProfiledGameState


def create_profiled_neural_network(profiler: AzulProfiler):
    """Create a neural network wrapper with profiled forward pass."""

    class ProfiledNeuralNetwork:
        def __init__(self, original_nn):
            self.original_nn = original_nn
            self._profiler = profiler

        def evaluate(self, state):
            with self._profiler.time_operation("nn.forward_pass"):
                # Also profile the state representation conversion
                with self._profiler.time_operation("nn.state_conversion"):
                    try:
                        numerical_state = state.get_numerical_state()
                        _ = numerical_state.get_flat_state_vector(normalize=True)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to get numerical state representation: {e}"
                        )

                # Profile the actual neural network inference
                with self._profiler.time_operation("nn.inference"):
                    return self.original_nn.evaluate(state)

        def __getattr__(self, name):
            # Delegate all other attributes to the original network
            return getattr(self.original_nn, name)

    return ProfiledNeuralNetwork


def create_profiled_mcts(profiler: AzulProfiler):
    """Create an MCTS class with profiled simulation loop."""
    from agents.mcts import MCTS as OriginalMCTS

    class ProfiledMCTS(OriginalMCTS):
        def search(self, root_state):
            with profiler.time_operation("mcts.full_search"):
                return super().search(root_state)

        def _simulate_with_depth(self, root):
            with profiler.time_operation("mcts.single_simulation"):
                return super()._simulate_with_depth(root)

        def _select_action(self, node):
            with profiler.time_operation("mcts.action_selection"):
                return super()._select_action(node)

        def _expand_and_evaluate(self, node, legal_actions=None):
            with profiler.time_operation("mcts.expand_and_evaluate"):
                return super()._expand_and_evaluate(node, legal_actions)

        def _backpropagate(self, path, leaf, value):
            with profiler.time_operation("mcts.backpropagate"):
                return super()._backpropagate(path, leaf, value)

    return ProfiledMCTS
