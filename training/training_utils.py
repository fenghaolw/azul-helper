"""
Training utilities for neural network training with improved loss functions and batch sampling.

This module provides enhanced training functionality specifically for the user's requirements:
- Batch sampling from replay buffer with (state, MCTS_policy_target, game_outcome) tuples
- Cross-entropy loss for policy head (instead of KL divergence)
- MSE loss for value head
- Complete training step implementation
- Model checkpoint management
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from training.neural_network import AzulNetwork
from training.replay_buffer import Experience, ReplayBuffer


class EnhancedLossFunctions:
    """Enhanced loss functions for policy and value heads."""

    def __init__(
        self,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize loss functions.

        Args:
            policy_loss_weight: Weight for policy loss in combined loss
            value_loss_weight: Weight for value loss in combined loss
            temperature: Temperature for softmax in cross-entropy loss
        """
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.temperature = temperature

    def policy_cross_entropy_loss(
        self, predicted_logits: torch.Tensor, target_policy: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy loss for policy head.

        Args:
            predicted_logits: Raw logits from policy head (batch_size, action_space_size)
            target_policy: Target policy probabilities (batch_size, action_space_size)

        Returns:
            Cross-entropy loss
        """
        # Apply temperature scaling
        scaled_logits = predicted_logits / self.temperature

        # Apply log_softmax for numerical stability
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        # Compute cross-entropy: -sum(target * log(predicted))
        ce_loss = -torch.sum(target_policy * log_probs, dim=-1)

        # Return mean loss across batch
        return torch.mean(ce_loss)

    def value_mse_loss(
        self, predicted_value: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE loss for value head.

        Args:
            predicted_value: Predicted state values (batch_size, 1)
            target_value: Target state values (batch_size, 1)

        Returns:
            MSE loss
        """
        return F.mse_loss(predicted_value.squeeze(), target_value.squeeze())

    def combined_loss(
        self,
        predicted_logits: torch.Tensor,
        predicted_value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined policy and value loss.

        Args:
            predicted_logits: Raw logits from policy head
            predicted_value: Predicted state values
            target_policy: Target policy probabilities
            target_value: Target state values

        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        policy_loss = self.policy_cross_entropy_loss(predicted_logits, target_policy)
        value_loss = self.value_mse_loss(predicted_value, target_value)

        total_loss = (
            self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
        )

        return total_loss, policy_loss, value_loss


class BatchSampler:
    """Enhanced batch sampler for replay buffer data."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize batch sampler.

        Args:
            replay_buffer: Replay buffer to sample from
            batch_size: Size of batches to sample
            device: Device to move tensors to
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

    def sample_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a batch of (state, MCTS_policy_target, game_outcome) from replay buffer.

        Returns:
            Tuple of (state_batch, policy_batch, outcome_batch) or None if buffer too small
        """
        batch_data = self.replay_buffer.sample(self.batch_size, self.device)
        return batch_data

    def sample_multiple_batches(
        self, num_batches: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample multiple batches for training.

        Args:
            num_batches: Number of batches to sample

        Returns:
            List of batch tuples
        """
        batches = []
        for _ in range(num_batches):
            batch = self.sample_batch()
            if batch is not None:
                batches.append(batch)
        return batches


class TrainingStep:
    """Complete training step implementation."""

    def __init__(
        self,
        model: AzulNetwork,
        optimizer: torch.optim.Optimizer,
        loss_functions: EnhancedLossFunctions,
        batch_sampler: BatchSampler,
        gradient_clip_norm: float = 1.0,
    ):
        """
        Initialize training step.

        Args:
            model: Neural network model
            optimizer: Optimizer for model parameters
            loss_functions: Loss function calculator
            batch_sampler: Batch sampler for replay buffer
            gradient_clip_norm: Maximum norm for gradient clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.batch_sampler = batch_sampler
        self.gradient_clip_norm = gradient_clip_norm

    def single_training_step(self) -> Optional[Dict[str, float]]:
        """
        Perform a single training step.

        Returns:
            Dictionary with loss statistics or None if no batch available
        """
        # Sample batch
        batch_data = self.batch_sampler.sample_batch()
        if batch_data is None:
            return None

        state_batch, policy_batch, outcome_batch = batch_data

        # Set model to training mode
        self.model.train()

        # Forward pass
        # Note: We need to modify the forward to return logits for cross-entropy loss
        with torch.enable_grad():
            if hasattr(self.model, "forward_with_logits"):
                policy_logits, predicted_value = self.model.forward_with_logits(  # type: ignore[misc]
                    state_batch, return_features=False
                )
            else:
                # Fallback: get logits from policy head before softmax
                # We'll modify the network to support this
                x = self.model.input_layer(state_batch)
                for i, layer in enumerate(self.model.body_layers):
                    if self.model.use_residual:
                        residual = self.model.residual_projections[i](x)
                        x = layer(x) + residual
                    else:
                        x = layer(x)

                # Get logits before softmax
                policy_logits = self.model.policy_head(x)
                predicted_value = self.model.value_head(x)

        # Calculate losses
        total_loss, policy_loss, value_loss = self.loss_functions.combined_loss(
            policy_logits, predicted_value, policy_batch, outcome_batch
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.gradient_clip_norm
        )

        # Optimizer step
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def train_for_steps(self, num_steps: int) -> Dict[str, float]:
        """
        Train for a specified number of steps.

        Args:
            num_steps: Number of training steps to perform

        Returns:
            Dictionary with aggregated training statistics
        """
        total_losses = []
        policy_losses = []
        value_losses = []
        successful_steps = 0

        start_time = time.time()

        for step in range(num_steps):
            step_stats = self.single_training_step()

            if step_stats is not None:
                total_losses.append(step_stats["total_loss"])
                policy_losses.append(step_stats["policy_loss"])
                value_losses.append(step_stats["value_loss"])
                successful_steps += 1

        training_time = time.time() - start_time

        if successful_steps == 0:
            return {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "successful_steps": 0,
                "training_time": training_time,
            }

        return {
            "total_loss": float(np.mean(total_losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "successful_steps": successful_steps,
            "training_time": training_time,
        }


class ModelCheckpointManager:
    """Manage model checkpoints with automatic saving and loading."""

    def __init__(
        self,
        save_dir: str,
        save_frequency: int = 10,
        keep_best: bool = True,
        keep_latest: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N iterations
            keep_best: Whether to keep track of best model
            keep_latest: Number of latest checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.keep_best = keep_best
        self.keep_latest = keep_latest

        self.best_score = -float("inf")
        self.checkpoint_history: List[Dict[str, Any]] = []

    def should_save_checkpoint(self, iteration: int) -> bool:
        """Check if checkpoint should be saved based on iteration."""
        return iteration % self.save_frequency == 0

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        stats: Dict[str, Any],
        score: Optional[float] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            iteration: Current training iteration
            stats: Training statistics
            score: Model score for comparison
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": stats,
            "score": score,
            "timestamp": time.time(),
        }

        # Save regular checkpoint
        checkpoint_filename = f"checkpoint_iter_{iteration:06d}.pt"
        checkpoint_path = self.save_dir / checkpoint_filename

        torch.save(checkpoint_data, checkpoint_path)

        # Track checkpoint
        self.checkpoint_history.append(
            {
                "iteration": iteration,
                "path": str(checkpoint_path),
                "score": score,
                "timestamp": checkpoint_data["timestamp"],
            }
        )

        # Save best model if applicable
        if self.keep_best and score is not None and score > self.best_score:
            self.best_score = score
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N."""
        if len(self.checkpoint_history) <= self.keep_latest:
            return

        # Sort by iteration (newest first)
        sorted_checkpoints = sorted(
            self.checkpoint_history, key=lambda x: x["iteration"], reverse=True
        )

        # Remove old checkpoint files
        for checkpoint in sorted_checkpoints[self.keep_latest :]:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        # Update history
        self.checkpoint_history = sorted_checkpoints[: self.keep_latest]

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            device: Device to load tensors to

        Returns:
            Checkpoint metadata
        """
        checkpoint_data = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        model.load_state_dict(checkpoint_data["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        return {
            "iteration": checkpoint_data.get("iteration", 0),
            "stats": checkpoint_data.get("stats", {}),
            "score": checkpoint_data.get("score", None),
            "timestamp": checkpoint_data.get("timestamp", None),
        }

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        if not self.checkpoint_history:
            return None

        latest = max(self.checkpoint_history, key=lambda x: x["iteration"])
        return latest["path"]

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        best_path = self.save_dir / "best_model.pt"
        return str(best_path) if best_path.exists() else None


def create_training_components(
    model: AzulNetwork,
    replay_buffer: ReplayBuffer,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
) -> Tuple[TrainingStep, ModelCheckpointManager]:
    """
    Create all training components for convenience.

    Args:
        model: Neural network model
        replay_buffer: Replay buffer with training data
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        batch_size: Batch size for training
        device: Device for computations
        policy_loss_weight: Weight for policy loss
        value_loss_weight: Weight for value loss

    Returns:
        Tuple of (training_step, checkpoint_manager)
    """
    device = device or torch.device("cpu")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Create loss functions
    loss_functions = EnhancedLossFunctions(
        policy_loss_weight=policy_loss_weight, value_loss_weight=value_loss_weight
    )

    # Create batch sampler
    batch_sampler = BatchSampler(
        replay_buffer=replay_buffer, batch_size=batch_size, device=device
    )

    # Create training step
    training_step = TrainingStep(
        model=model,
        optimizer=optimizer,
        loss_functions=loss_functions,
        batch_sampler=batch_sampler,
    )

    # Create checkpoint manager
    checkpoint_manager = ModelCheckpointManager(save_dir="models/training_checkpoints")

    return training_step, checkpoint_manager
