"""
ETA (Estimated Time of Arrival) tracking utility for training processes.

This module provides sophisticated ETA tracking capabilities with:
- Moving average computation for stable estimates
- Multiple estimation methods (linear, exponential smoothing)
- Phase-based tracking for different training stages
- Progress display and formatting utilities
"""

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple


class ETATracker:
    """
    Tracks training progress and estimates time remaining.

    Features:
    - Moving average estimation with configurable window
    - Exponential smoothing for adaptive estimates
    - Phase-based tracking (self-play, training, evaluation)
    - Automatic detection of changing performance
    """

    def __init__(
        self,
        total_iterations: int,
        moving_average_window: int = 10,
        exponential_smoothing_alpha: float = 0.3,
        enable_phase_tracking: bool = True,
    ):
        """
        Initialize ETA tracker.

        Args:
            total_iterations: Total number of iterations to complete
            moving_average_window: Size of window for moving average calculation
            exponential_smoothing_alpha: Alpha parameter for exponential smoothing (0-1)
            enable_phase_tracking: Whether to track individual phases
        """
        self.total_iterations = total_iterations
        self.moving_average_window = moving_average_window
        self.exponential_smoothing_alpha = exponential_smoothing_alpha
        self.enable_phase_tracking = enable_phase_tracking

        # Overall tracking
        self.start_time = time.time()
        self.current_iteration = 0
        self.iteration_times: Deque[float] = deque(maxlen=moving_average_window)
        self.exponential_avg_time: Optional[float] = None
        self.iteration_start_time: Optional[float] = None

        # Phase tracking
        self.phase_times: Dict[str, Deque[float]] = {
            "self_play": deque(maxlen=moving_average_window),
            "training": deque(maxlen=moving_average_window),
            "evaluation": deque(maxlen=moving_average_window),
        }
        self.current_phase_start: Optional[float] = None
        self.current_phase: Optional[str] = None

        # Statistics
        self.stats_history: List[Dict[str, Any]] = []

    def start_iteration(self, iteration: int) -> None:
        """Start tracking a new iteration."""
        self.current_iteration = iteration
        self.iteration_start_time = time.time()

    def start_phase(self, phase_name: str) -> None:
        """Start tracking a training phase (self_play, training, evaluation)."""
        if self.enable_phase_tracking:
            self.current_phase = phase_name
            self.current_phase_start = time.time()

    def end_phase(self) -> None:
        """End the current phase and record its duration."""
        if self.enable_phase_tracking and self.current_phase_start:
            phase_duration = time.time() - self.current_phase_start
            if self.current_phase in self.phase_times:
                self.phase_times[self.current_phase].append(phase_duration)
            self.current_phase_start = None
            self.current_phase = None

    def end_iteration(self) -> None:
        """End the current iteration and update estimates."""
        if self.iteration_start_time is not None:
            iteration_duration = time.time() - self.iteration_start_time
            self.iteration_times.append(iteration_duration)

            # Update exponential moving average
            if self.exponential_avg_time is None:
                self.exponential_avg_time = iteration_duration
            else:
                self.exponential_avg_time = (
                    self.exponential_smoothing_alpha * iteration_duration
                    + (1 - self.exponential_smoothing_alpha) * self.exponential_avg_time
                )

    def get_eta_estimates(self) -> Dict[str, Optional[float]]:
        """
        Get ETA estimates using different methods.

        Returns:
            Dictionary with different ETA estimates in seconds
        """
        if not self.iteration_times:
            return {
                "moving_average": None,
                "exponential_smoothing": None,
                "linear_regression": None,
                "best_estimate": None,
            }

        remaining_iterations = self.total_iterations - self.current_iteration

        # Moving average estimate
        avg_time = sum(self.iteration_times) / len(self.iteration_times)
        moving_avg_eta = avg_time * remaining_iterations

        # Exponential smoothing estimate
        exp_smoothing_eta = None
        if self.exponential_avg_time is not None:
            exp_smoothing_eta = self.exponential_avg_time * remaining_iterations

        # Simple linear regression on recent times
        linear_eta = None
        if len(self.iteration_times) >= 3:
            times = list(self.iteration_times)
            n = len(times)
            indices = list(range(n))

            # Calculate linear trend
            mean_x = sum(indices) / n
            mean_y = sum(times) / n

            numerator = sum(
                (indices[i] - mean_x) * (times[i] - mean_y) for i in range(n)
            )
            denominator = sum((indices[i] - mean_x) ** 2 for i in range(n))

            if denominator != 0:
                slope = numerator / denominator
                # Project future time based on trend
                projected_time = mean_y + slope * (n + remaining_iterations - mean_x)
                linear_eta = projected_time * remaining_iterations

        # Choose best estimate (prioritize exponential smoothing if available)
        best_estimate = (
            exp_smoothing_eta if exp_smoothing_eta is not None else moving_avg_eta
        )

        return {
            "moving_average": moving_avg_eta,
            "exponential_smoothing": exp_smoothing_eta,
            "linear_regression": linear_eta,
            "best_estimate": best_estimate,
        }

    def get_phase_estimates(self) -> Dict[str, Optional[float]]:
        """Get ETA estimates for individual phases."""
        if not self.enable_phase_tracking:
            return {}

        phase_estimates: Dict[str, Optional[float]] = {}
        for phase_name, times in self.phase_times.items():
            if times:
                avg_time = sum(times) / len(times)
                phase_estimates[f"{phase_name}_avg_time"] = avg_time
            else:
                phase_estimates[f"{phase_name}_avg_time"] = None

        return phase_estimates

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.current_iteration / self.total_iterations) * 100

        eta_estimates = self.get_eta_estimates()
        phase_estimates = self.get_phase_estimates()

        # Calculate completion time
        completion_time: Optional[datetime] = None
        if eta_estimates["best_estimate"] is not None:
            completion_time = datetime.now() + timedelta(
                seconds=eta_estimates["best_estimate"]
            )

        return {
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "progress_percent": progress_percent,
            "elapsed_time": elapsed_time,
            "avg_iteration_time": (
                sum(self.iteration_times) / len(self.iteration_times)
                if self.iteration_times
                else None
            ),
            "eta_seconds": eta_estimates["best_estimate"],
            "completion_time": completion_time,
            "eta_estimates": eta_estimates,
            "phase_estimates": phase_estimates,
        }

    def format_progress_bar(self, width: int = 50) -> str:
        """Create a visual progress bar."""
        progress = self.current_iteration / self.total_iterations
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = progress * 100
        return f"[{bar}] {percentage:.1f}%"

    def format_time_display(self, seconds: Optional[float]) -> str:
        """Format time duration for display."""
        if seconds is None:
            return "N/A"

        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def print_progress_update(self, detailed: bool = False) -> None:
        """Print formatted progress update."""
        summary = self.get_progress_summary()

        # Basic progress line
        progress_bar = self.format_progress_bar()
        elapsed_str = self.format_time_display(summary["elapsed_time"])
        eta_str = self.format_time_display(summary["eta_seconds"])

        print(f"\n{progress_bar}")
        print(f"Progress: {self.current_iteration}/{self.total_iterations} iterations")
        print(f"Elapsed: {elapsed_str} | ETA: {eta_str}")

        if summary["completion_time"]:
            completion_str = summary["completion_time"].strftime("%Y-%m-%d %H:%M:%S")
            print(f"Estimated completion: {completion_str}")

        if detailed and summary["avg_iteration_time"]:
            avg_iter_str = self.format_time_display(summary["avg_iteration_time"])
            print(f"Average iteration time: {avg_iter_str}")

            # Show phase timing if available
            phase_estimates = summary["phase_estimates"]
            if phase_estimates:
                print("\nPhase timing averages:")
                for phase in ["self_play", "training", "evaluation"]:
                    key = f"{phase}_avg_time"
                    if key in phase_estimates and phase_estimates[key]:
                        time_str = self.format_time_display(phase_estimates[key])
                        print(f"  {phase.replace('_', ' ').title()}: {time_str}")
