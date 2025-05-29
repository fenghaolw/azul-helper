"""
Real-time monitoring dashboard for Azul neural network training.

This script provides a live dashboard to monitor training progress including:
- Loss curves
- Self-play statistics
- Evaluation scores
- Training efficiency metrics
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class TrainingDashboard:
    """Live dashboard for monitoring training progress."""

    def __init__(self, log_file: str, update_interval: int = 5):
        """
        Initialize the dashboard.

        Args:
            log_file: Path to the training log JSON file
            update_interval: Update interval in seconds
        """
        if not HAS_MATPLOTLIB:
            raise RuntimeError("matplotlib is required for the dashboard")

        self.log_file = Path(log_file)
        self.update_interval = update_interval

        # Data storage
        self.data: List[Dict] = []
        self.last_update = 0

        # Create figure and subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f"Azul Training Monitor - {log_file}", fontsize=16)

        # Configure subplots
        self.ax_loss = self.axes[0, 0]
        self.ax_selfplay = self.axes[0, 1]
        self.ax_eval = self.axes[1, 0]
        self.ax_efficiency = self.axes[1, 1]

        # Set up plot styles
        plt.style.use("seaborn-v0_8-darkgrid")

    def load_data(self) -> bool:
        """
        Load data from log file.

        Returns:
            True if new data was loaded
        """
        if not self.log_file.exists():
            return False

        try:
            with open(self.log_file, "r") as f:
                new_data = json.load(f)

            if len(new_data) > len(self.data):
                self.data = new_data
                return True

        except (json.JSONDecodeError, IOError):
            pass

        return False

    def update_plots(self, frame):
        """Update all plots with latest data."""
        # Load new data
        if not self.load_data() and len(self.data) == 0:
            return

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Extract data
        iterations = [d["iteration"] for d in self.data]

        # Plot 1: Loss curves
        self._plot_losses(iterations)

        # Plot 2: Self-play statistics
        self._plot_selfplay(iterations)

        # Plot 3: Evaluation scores
        self._plot_evaluation(iterations)

        # Plot 4: Training efficiency
        self._plot_efficiency(iterations)

        # Update layout
        self.fig.tight_layout()

    def _plot_losses(self, iterations: List[int]):
        """Plot training loss curves."""
        # Extract loss data
        total_losses = []
        policy_losses = []
        value_losses = []

        for d in self.data:
            if "total_loss" in d:
                total_losses.append((d["iteration"], d["total_loss"]))
                policy_losses.append((d["iteration"], d["policy_loss"]))
                value_losses.append((d["iteration"], d["value_loss"]))

        if total_losses:
            iters, totals = zip(*total_losses)
            _, policies = zip(*policy_losses)
            _, values = zip(*value_losses)

            self.ax_loss.plot(iters, totals, "b-", label="Total Loss", linewidth=2)
            self.ax_loss.plot(iters, policies, "g--", label="Policy Loss", alpha=0.7)
            self.ax_loss.plot(iters, values, "r--", label="Value Loss", alpha=0.7)

            self.ax_loss.set_xlabel("Iteration")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.set_title("Training Losses")
            self.ax_loss.legend()
            self.ax_loss.grid(True, alpha=0.3)

            # Add latest values as text
            if len(totals) > 0:
                latest_text = f"Latest: {totals[-1]:.6f}"
                self.ax_loss.text(
                    0.02,
                    0.98,
                    latest_text,
                    transform=self.ax_loss.transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

    def _plot_selfplay(self, iterations: List[int]):
        """Plot self-play statistics."""
        # Extract self-play data
        buffer_sizes = []
        games_played = []
        avg_moves = []

        for d in self.data:
            if "buffer_size" in d:
                buffer_sizes.append((d["iteration"], d["buffer_size"]))
            if "games_played" in d:
                games_played.append((d["iteration"], d["games_played"]))
            if "avg_moves_per_game" in d:
                avg_moves.append((d["iteration"], d["avg_moves_per_game"]))

        # Create twin axes for different scales
        ax2 = self.ax_selfplay.twinx()

        # Plot buffer size
        if buffer_sizes:
            iters, sizes = zip(*buffer_sizes)
            line1 = self.ax_selfplay.plot(
                iters, sizes, "b-", label="Buffer Size", linewidth=2
            )
            self.ax_selfplay.set_ylabel("Buffer Size", color="b")
            self.ax_selfplay.tick_params(axis="y", labelcolor="b")

        # Plot average moves
        if avg_moves:
            iters, moves = zip(*avg_moves)
            line2 = ax2.plot(iters, moves, "r-", label="Avg Moves/Game", linewidth=2)
            ax2.set_ylabel("Avg Moves per Game", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

        self.ax_selfplay.set_xlabel("Iteration")
        self.ax_selfplay.set_title("Self-Play Statistics")
        self.ax_selfplay.grid(True, alpha=0.3)

        # Add combined legend
        if buffer_sizes and avg_moves:
            lines = line1 + line2
            labels = [line.get_label() for line in lines]
            self.ax_selfplay.legend(lines, labels, loc="upper left")

    def _plot_evaluation(self, iterations: List[int]):
        """Plot evaluation scores."""
        # Extract evaluation data
        eval_scores = []
        best_scores = []

        for d in self.data:
            if "eval_score" in d:
                eval_scores.append((d["iteration"], d["eval_score"]))
                if d.get("is_best", False):
                    best_scores.append((d["iteration"], d["eval_score"]))

        if eval_scores:
            iters, scores = zip(*eval_scores)
            self.ax_eval.plot(iters, scores, "b-", label="Eval Score", linewidth=2)

            # Mark best scores
            if best_scores:
                best_iters, best_vals = zip(*best_scores)
                self.ax_eval.scatter(
                    best_iters,
                    best_vals,
                    c="red",
                    s=100,
                    marker="*",
                    label="Best Model",
                    zorder=5,
                )

            self.ax_eval.set_xlabel("Iteration")
            self.ax_eval.set_ylabel("Evaluation Score")
            self.ax_eval.set_title("Model Evaluation")
            self.ax_eval.legend()
            self.ax_eval.grid(True, alpha=0.3)

            # Add latest score
            if len(scores) > 0:
                latest_text = f"Latest: {scores[-1]:.4f}"
                self.ax_eval.text(
                    0.02,
                    0.98,
                    latest_text,
                    transform=self.ax_eval.transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
                )

    def _plot_efficiency(self, iterations: List[int]):
        """Plot training efficiency metrics."""
        # Extract timing data
        iter_times = []
        total_times = []

        for d in self.data:
            if "iteration_time" in d:
                iter_times.append((d["iteration"], d["iteration_time"]))
            if "elapsed_time" in d:
                total_times.append(
                    (d["iteration"], d["elapsed_time"] / 3600)
                )  # Convert to hours

        # Create twin axes
        ax2 = self.ax_efficiency.twinx()

        # Plot iteration times
        if iter_times:
            iters, times = zip(*iter_times)
            self.ax_efficiency.bar(iters, times, alpha=0.6, label="Iteration Time")
            self.ax_efficiency.set_ylabel("Iteration Time (s)", color="b")
            self.ax_efficiency.tick_params(axis="y", labelcolor="b")

        # Plot total elapsed time
        if total_times:
            iters, totals = zip(*total_times)
            ax2.plot(iters, totals, "r-", linewidth=2, label="Total Time")
            ax2.set_ylabel("Total Elapsed Time (hours)", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

        self.ax_efficiency.set_xlabel("Iteration")
        self.ax_efficiency.set_title("Training Efficiency")
        self.ax_efficiency.grid(True, alpha=0.3)

        # Add statistics
        if iter_times and total_times:
            avg_iter_time = sum(t for _, t in iter_times) / len(iter_times)
            total_hours = total_times[-1][1] if total_times else 0

            stats_text = (
                f"Avg iteration: {avg_iter_time:.1f}s\nTotal: {total_hours:.1f}h"
            )
            self.ax_efficiency.text(
                0.02,
                0.98,
                stats_text,
                transform=self.ax_efficiency.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
            )

    def run(self):
        """Start the dashboard."""
        print(f"Starting training dashboard...")
        print(f"Monitoring: {self.log_file}")
        print(f"Update interval: {self.update_interval}s")
        print(f"Press Ctrl+C to stop")

        # Create animation
        _ = FuncAnimation(
            self.fig,
            self.update_plots,
            interval=self.update_interval * 1000,
            cache_frame_data=False,
        )

        # Show plot
        plt.show()


def print_training_summary(log_file: str):
    """Print a text summary of training progress."""
    log_path = Path(log_file)

    if not log_path.exists():
        print(f"Log file not found: {log_file}")
        return

    try:
        with open(log_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading log file: {e}")
        return

    if not data:
        print("No training data available yet")
        return

    latest = data[-1]

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\nLog file: {log_file}")
    print(f"Last updated: {latest.get('timestamp', 'Unknown')}")
    print(f"\nProgress: Iteration {latest['iteration']}")
    print(f"Total elapsed time: {latest.get('elapsed_time', 0)/3600:.1f} hours")

    print("\n--- Self-Play ---")
    print(f"Games played: {latest.get('games_played', 'N/A')}")
    print(f"Buffer size: {latest.get('buffer_size', 'N/A')}")
    print(f"Avg moves/game: {latest.get('avg_moves_per_game', 'N/A')}")

    if "total_loss" in latest:
        print("\n--- Training ---")
        print(f"Total loss: {latest['total_loss']:.6f}")
        print(f"Policy loss: {latest['policy_loss']:.6f}")
        print(f"Value loss: {latest['value_loss']:.6f}")

    if "eval_score" in latest:
        print("\n--- Evaluation ---")
        print(f"Eval score: {latest['eval_score']:.4f}")
        print(f"Best model: {'Yes' if latest.get('is_best', False) else 'No'}")

    # Find best score
    best_score = max(
        (d.get("eval_score", 0) for d in data if "eval_score" in d), default=0
    )
    best_iter = next(
        (d["iteration"] for d in data if d.get("eval_score", 0) == best_score), None
    )

    if best_iter:
        print(f"\nBest score: {best_score:.4f} (iteration {best_iter})")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor Azul neural network training")
    parser.add_argument("log_file", help="Path to training log JSON file")
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Update interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary only (no dashboard)"
    )

    args = parser.parse_args()

    if args.summary or not HAS_MATPLOTLIB:
        print_training_summary(args.log_file)
    else:
        dashboard = TrainingDashboard(args.log_file, args.interval)
        dashboard.run()


if __name__ == "__main__":
    main()
