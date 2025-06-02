"""
OpenSpiel Agent Performance Profiler.

This module provides detailed profiling capabilities specifically for OpenSpiel-based agents:
- OpenSpiel MCTS agent timing
- OpenSpiel AlphaZero agent timing
- State conversion overhead analysis
- OpenSpiel internal operation profiling
- Memory and GPU usage tracking
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


class OpenSpielProfilerStats:
    """Container for OpenSpiel agent profiling statistics."""

    def __init__(self):
        self.timing_data: Dict[str, List[float]] = {}
        self.call_counts: Dict[str, int] = {}
        self.memory_usage: Dict[str, Union[float, np.floating[Any]]] = {}
        self.gpu_stats: Dict[str, Any] = {}
        self.openspiel_stats: Dict[str, Any] = {}

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

    def add_openspiel_stats(self, name: str, stats: Dict[str, Any]):
        """Add OpenSpiel-specific statistics."""
        self.openspiel_stats[name] = stats

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
        summary["openspiel_stats"] = self.openspiel_stats
        return summary

    def print_summary(self):
        """Print a formatted summary of profiling results."""
        print("\n" + "=" * 80)
        print("OPENSPIEL AGENT PERFORMANCE PROFILING RESULTS")
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
            f"{'Operation':<35} {'Total(s)':<10} {'Avg(ms)':<10} {'Count':<8} {'Calls/s':<10}"
        )
        print("-" * 80)

        for name, data in timing_items:
            print(
                f"{name:<35} {data['total_time']:<10.3f} "
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

        # OpenSpiel-specific stats
        if summary.get("openspiel_stats"):
            print("\nOPENSPIEL STATISTICS:")
            print("-" * 40)
            for name, os_data in summary["openspiel_stats"].items():
                print(f"{name}:")
                for key, value in os_data.items():
                    print(f"  {key}: {value}")


class OpenSpielProfiler:
    """Main profiler class for OpenSpiel agents."""

    def __init__(
        self, enable_memory_profiling: bool = True, enable_gpu_profiling: bool = True
    ):
        self.stats = OpenSpielProfilerStats()
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


# OpenSpiel-specific profiling functions
def create_profiled_openspiel_mcts_agent(profiler: OpenSpielProfiler):
    """Create an OpenSpiel MCTS agent with profiled methods."""
    from agents.openspiel_agents import OpenSpielMCTSAgent

    class ProfiledOpenSpielMCTSAgent(OpenSpielMCTSAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._profiler = profiler

        def select_action(self, state, deterministic=False):
            with self._profiler.time_operation("openspiel_mcts.select_action"):
                # Profile state conversion
                with self._profiler.time_operation("openspiel_mcts.state_conversion"):
                    openspiel_state = self._convert_to_openspiel_state(state)

                # Profile OpenSpiel search
                with self._profiler.time_operation("openspiel_mcts.search"):
                    if deterministic:
                        # Profile step_with_policy for deterministic selection
                        with self._profiler.time_operation(
                            "openspiel_mcts.step_with_policy"
                        ):
                            result = self._searcher.step_with_policy(openspiel_state)
                            if isinstance(result, tuple) and len(result) == 2:
                                policy_list, action_int = result
                            else:
                                action_int = result
                    else:
                        # Profile step for stochastic selection
                        with self._profiler.time_operation("openspiel_mcts.step"):
                            action_int = self._searcher.step(openspiel_state)

                # Profile action conversion
                with self._profiler.time_operation("openspiel_mcts.action_conversion"):
                    # Handle different action_int formats
                    if isinstance(action_int, (list, np.ndarray)):
                        action_int = action_int[0] if len(action_int) > 0 else 0
                    elif isinstance(action_int, tuple):
                        action_int = action_int[0]

                    if isinstance(action_int, tuple):
                        while isinstance(action_int, tuple):
                            action_int = action_int[0]

                    action_int = int(action_int)
                    return self._convert_to_azul_action(action_int, openspiel_state)

        def get_action_probabilities(self, state):
            with self._profiler.time_operation(
                "openspiel_mcts.get_action_probabilities"
            ):
                # Profile state conversion
                with self._profiler.time_operation(
                    "openspiel_mcts.prob_state_conversion"
                ):
                    openspiel_state = self._convert_to_openspiel_state(state)

                # Profile policy computation
                with self._profiler.time_operation("openspiel_mcts.policy_computation"):
                    policy_list, _ = self._searcher.step_with_policy(openspiel_state)
                    policy_dict = {action: prob for action, prob in policy_list}

                # Profile probability conversion
                with self._profiler.time_operation(
                    "openspiel_mcts.probability_conversion"
                ):
                    legal_actions = state.get_legal_actions()
                    probs = np.zeros(len(legal_actions))

                    for i, action in enumerate(legal_actions):
                        action_int = self._action_to_int(action, openspiel_state)
                        if action_int in policy_dict:
                            probs[i] = policy_dict[action_int]

                    # Normalize
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        probs = np.ones(len(legal_actions)) / len(legal_actions)

                    return probs

        def get_stats(self):
            """Get enhanced stats including profiling data."""
            base_stats = super().get_stats()
            profiling_stats = self._profiler.get_summary()

            # Add OpenSpiel-specific stats
            enhanced_stats = {
                **base_stats,
                "profiling_summary": profiling_stats,
                "total_profiled_operations": sum(
                    data.get("call_count", 0)
                    for data in profiling_stats.values()
                    if isinstance(data, dict) and "call_count" in data
                ),
            }

            return enhanced_stats

    return ProfiledOpenSpielMCTSAgent


def create_profiled_openspiel_alphazero_agent(profiler: OpenSpielProfiler):
    """Create an OpenSpiel AlphaZero agent with profiled methods."""
    from agents.openspiel_agents import OpenSpielAlphaZeroAgent

    class ProfiledOpenSpielAlphaZeroAgent(OpenSpielAlphaZeroAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._profiler = profiler

        def select_action(self, state, deterministic=False):
            with self._profiler.time_operation("openspiel_alphazero.select_action"):
                # Profile state conversion
                with self._profiler.time_operation(
                    "openspiel_alphazero.state_conversion"
                ):
                    openspiel_state = self._convert_to_openspiel_state(state)

                # Handle temperature setting
                old_temp = self.temperature
                if deterministic:
                    self.temperature = 0.0

                try:
                    # Profile AlphaZero step (includes neural network inference + MCTS)
                    with self._profiler.time_operation(
                        "openspiel_alphazero.alphazero_step"
                    ):
                        action_int = self._bot.step(openspiel_state)

                    # Profile action conversion
                    with self._profiler.time_operation(
                        "openspiel_alphazero.action_conversion"
                    ):
                        return self._convert_to_azul_action(action_int, openspiel_state)
                finally:
                    self.temperature = old_temp

        def get_action_probabilities(self, state):
            with self._profiler.time_operation(
                "openspiel_alphazero.get_action_probabilities"
            ):
                # Profile state conversion
                with self._profiler.time_operation(
                    "openspiel_alphazero.prob_state_conversion"
                ):
                    openspiel_state = self._convert_to_openspiel_state(state)

                # Profile policy computation (neural network + MCTS)
                with self._profiler.time_operation(
                    "openspiel_alphazero.policy_computation"
                ):
                    policy_list, _ = self._bot.step_with_policy(openspiel_state)
                    policy_dict = {action: prob for action, prob in policy_list}

                # Profile probability conversion
                with self._profiler.time_operation(
                    "openspiel_alphazero.probability_conversion"
                ):
                    legal_actions = state.get_legal_actions()
                    probs = np.zeros(len(legal_actions))

                    for i, action in enumerate(legal_actions):
                        action_int = self._action_to_int(action, openspiel_state)
                        if action_int in policy_dict:
                            probs[i] = policy_dict[action_int]

                    # Normalize
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        probs = np.ones(len(legal_actions)) / len(legal_actions)

                    return probs

        def get_stats(self):
            """Get enhanced stats including profiling data."""
            base_stats = super().get_stats()
            profiling_stats = self._profiler.get_summary()

            # Add OpenSpiel-specific stats
            enhanced_stats = {
                **base_stats,
                "profiling_summary": profiling_stats,
                "total_profiled_operations": sum(
                    data.get("call_count", 0)
                    for data in profiling_stats.values()
                    if isinstance(data, dict) and "call_count" in data
                ),
            }

            return enhanced_stats

    return ProfiledOpenSpielAlphaZeroAgent


def create_profiled_random_agent(profiler: OpenSpielProfiler):
    """Create a Random agent with profiled methods."""
    from agents.openspiel_agents import RandomAgent

    class ProfiledRandomAgent(RandomAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._profiler = profiler

        def select_action(self, state, deterministic=False):
            with self._profiler.time_operation("random_agent.select_action"):
                # Profile getting legal actions
                with self._profiler.time_operation("random_agent.get_legal_actions"):
                    legal_actions = state.get_legal_actions()

                # Profile random selection
                with self._profiler.time_operation("random_agent.random_choice"):
                    if not legal_actions:
                        raise ValueError("No legal actions available")
                    return np.random.choice(legal_actions)

        def get_action_probabilities(self, state):
            with self._profiler.time_operation("random_agent.get_action_probabilities"):
                # Profile getting legal actions
                with self._profiler.time_operation(
                    "random_agent.prob_get_legal_actions"
                ):
                    legal_actions = state.get_legal_actions()

                # Profile uniform distribution creation
                with self._profiler.time_operation("random_agent.uniform_distribution"):
                    return np.ones(len(legal_actions)) / len(legal_actions)

        def get_stats(self):
            """Get enhanced stats including profiling data."""
            base_stats = super().get_stats()
            profiling_stats = self._profiler.get_summary()

            # Add profiling stats
            enhanced_stats = {
                **base_stats,
                "profiling_summary": profiling_stats,
                "total_profiled_operations": sum(
                    data.get("call_count", 0)
                    for data in profiling_stats.values()
                    if isinstance(data, dict) and "call_count" in data
                ),
            }

            return enhanced_stats

    return ProfiledRandomAgent


# Convenience function to create any profiled OpenSpiel agent
def create_profiled_agent(
    agent_type: str, profiler: OpenSpielProfiler, *args, **kwargs
):
    """
    Create a profiled OpenSpiel agent of the specified type.

    Args:
        agent_type: Type of agent ("mcts", "alphazero", "random")
        profiler: OpenSpielProfiler instance
        *args, **kwargs: Arguments for agent initialization

    Returns:
        Profiled agent instance
    """
    if agent_type.lower() == "mcts":
        agent_class = create_profiled_openspiel_mcts_agent(profiler)
        return agent_class(*args, **kwargs)
    elif agent_type.lower() == "alphazero":
        agent_class = create_profiled_openspiel_alphazero_agent(profiler)
        return agent_class(*args, **kwargs)
    elif agent_type.lower() == "random":
        agent_class = create_profiled_random_agent(profiler)
        return agent_class(*args, **kwargs)
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Choose from 'mcts', 'alphazero', 'random'"
        )


# Example usage function
def profile_openspiel_agents_comparison():
    """Example function showing how to profile multiple OpenSpiel agents."""
    from game.game_state import GameState

    print("Starting OpenSpiel agents profiling comparison...")

    # Create profiler
    profiler = OpenSpielProfiler(
        enable_memory_profiling=True, enable_gpu_profiling=True
    )

    # Create profiled agents
    mcts_agent = create_profiled_agent("mcts", profiler, num_simulations=100)
    random_agent = create_profiled_agent("random", profiler, seed=42)

    # Create test game state
    game_state = GameState(num_players=2, seed=42)

    # Test agents
    print("Testing MCTS agent...")
    for i in range(5):
        action = mcts_agent.select_action(game_state)
        probs = mcts_agent.get_action_probabilities(game_state)
        print(f"  Move {i+1}: {action}, probs shape: {probs.shape}")

    print("Testing Random agent...")
    for i in range(10):
        action = random_agent.select_action(game_state)
        probs = random_agent.get_action_probabilities(game_state)

    # Print comprehensive results
    profiler.print_summary()

    return profiler
