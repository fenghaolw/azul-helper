"""
Replay Buffer for storing and sampling training experiences from self-play games.

This module implements a replay buffer that stores game experiences and provides
efficient sampling for neural network training. It supports outcome-based reward
assignment and balanced sampling strategies.
"""

import random
from collections import deque
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from game.game_state import GameState
from game.state_representation import AzulStateRepresentation


class Experience:
    """
    A single training experience containing state, policy target, and outcome.

    Attributes:
        state: The game state
        mcts_policy: The MCTS-improved policy probabilities
        player_id: ID of the player who made this move
        outcome: Game outcome from this player's perspective (1 for win, 0 for draw, -1 for loss)
    """

    def __init__(
        self,
        state: GameState,
        mcts_policy: np.ndarray,
        player_id: int,
        outcome: Optional[float] = None,
    ):
        self.state = state
        self.mcts_policy = mcts_policy
        self.player_id = player_id
        self.outcome = outcome

    def set_outcome(self, outcome: float) -> None:
        """Set the game outcome for this experience."""
        self.outcome = outcome

    def to_tensor(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert experience to tensors for training.

        Returns:
            Tuple of (state_tensor, policy_tensor, outcome_tensor)
        """
        # Get numerical state representation
        state_repr = self.state.get_numerical_state()
        state_vector = state_repr.get_flat_state_vector()

        # Convert to tensors
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)
        policy_tensor = torch.tensor(self.mcts_policy, dtype=torch.float32)
        outcome_tensor = torch.tensor(
            [self.outcome if self.outcome is not None else 0.0], dtype=torch.float32
        )

        if device is not None:
            state_tensor = state_tensor.to(device)
            policy_tensor = policy_tensor.to(device)
            outcome_tensor = outcome_tensor.to(device)

        return state_tensor, policy_tensor, outcome_tensor


class ReplayBuffer:
    """
    Replay buffer for storing and sampling training experiences.

    This buffer stores experiences from self-play games and provides efficient
    sampling for neural network training. It supports automatic capacity management
    and outcome-based reward assignment.
    """

    def __init__(
        self,
        capacity: int = 100000,
        min_size_for_sampling: int = 1000,
        balance_outcomes: bool = True,
    ):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            min_size_for_sampling: Minimum buffer size before sampling is allowed
            balance_outcomes: Whether to balance sampling by game outcomes
        """
        self.capacity = capacity
        self.min_size_for_sampling = min_size_for_sampling
        self.balance_outcomes = balance_outcomes

        self.buffer: deque = deque(maxlen=capacity)
        self.games_added = 0

    def add_game(self, experiences: List[Experience], outcomes: List[float]) -> None:
        """
        Add a complete game's experiences to the buffer.

        Args:
            experiences: List of experiences from the game
            outcomes: List of outcomes for each player
        """
        # Assign outcomes to experiences based on player IDs
        for exp in experiences:
            exp.set_outcome(outcomes[exp.player_id])

        # Add all experiences to buffer
        self.buffer.extend(experiences)
        self.games_added += 1

    def add_experience(self, experience: Experience) -> None:
        """Add a single experience to the buffer."""
        self.buffer.append(experience)

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a batch of experiences for training.

        Args:
            batch_size: Number of experiences to sample
            device: Device to move tensors to

        Returns:
            Tuple of (state_batch, policy_batch, outcome_batch) or None if buffer too small
        """
        if len(self.buffer) < self.min_size_for_sampling:
            return None

        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Sample experiences
        if self.balance_outcomes:
            experiences = self._balanced_sample(batch_size)
        else:
            experiences = random.sample(list(self.buffer), batch_size)

        # Convert to tensors
        states = []
        policies = []
        outcomes = []

        for exp in experiences:
            state_tensor, policy_tensor, outcome_tensor = exp.to_tensor(device)
            states.append(state_tensor)
            policies.append(policy_tensor)
            outcomes.append(outcome_tensor)

        state_batch = torch.stack(states)
        policy_batch = torch.stack(policies)
        outcome_batch = torch.stack(outcomes)

        return state_batch, policy_batch, outcome_batch

    def _balanced_sample(self, batch_size: int) -> List[Experience]:
        """
        Sample experiences with balanced outcomes (equal wins/losses).

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        # Separate experiences by outcome
        wins = [exp for exp in self.buffer if exp.outcome and exp.outcome > 0.5]
        losses = [exp for exp in self.buffer if exp.outcome and exp.outcome < -0.5]
        draws = [exp for exp in self.buffer if exp.outcome and abs(exp.outcome) <= 0.5]

        # Calculate how many of each to sample
        num_wins = min(len(wins), batch_size // 3)
        num_losses = min(len(losses), batch_size // 3)
        num_draws = min(len(draws), batch_size - num_wins - num_losses)

        # If we don't have enough of any category, fill from others
        remaining = batch_size - num_wins - num_losses - num_draws
        if remaining > 0:
            all_remaining = wins + losses + draws
            if len(all_remaining) >= remaining:
                additional = random.sample(all_remaining, remaining)
                num_wins += len([exp for exp in additional if exp.outcome > 0.5])
                num_losses += len([exp for exp in additional if exp.outcome < -0.5])
                num_draws += len([exp for exp in additional if abs(exp.outcome) <= 0.5])

        # Sample from each category
        sampled = []
        if num_wins > 0 and wins:
            sampled.extend(random.sample(wins, min(num_wins, len(wins))))
        if num_losses > 0 and losses:
            sampled.extend(random.sample(losses, min(num_losses, len(losses))))
        if num_draws > 0 and draws:
            sampled.extend(random.sample(draws, min(num_draws, len(draws))))

        # If still not enough, sample randomly from all
        if len(sampled) < batch_size:
            remaining_needed = batch_size - len(sampled)
            all_others = [exp for exp in self.buffer if exp not in sampled]
            if all_others:
                sampled.extend(
                    random.sample(all_others, min(remaining_needed, len(all_others)))
                )

        return sampled

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
        self.games_added = 0

    def size(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def is_ready_for_sampling(self) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= self.min_size_for_sampling

    def get_statistics(self) -> dict:
        """
        Get statistics about the replay buffer.

        Returns:
            Dictionary containing buffer statistics
        """
        if not self.buffer:
            return {
                "size": 0,
                "games_added": self.games_added,
                "capacity": self.capacity,
                "ready_for_sampling": False,
            }

        outcomes = [exp.outcome for exp in self.buffer if exp.outcome is not None]

        stats = {
            "size": len(self.buffer),
            "games_added": self.games_added,
            "capacity": self.capacity,
            "ready_for_sampling": self.is_ready_for_sampling(),
        }

        if outcomes:
            stats.update(
                {
                    "mean_outcome": np.mean(outcomes),
                    "std_outcome": np.std(outcomes),
                    "num_wins": len([o for o in outcomes if o > 0.5]),
                    "num_losses": len([o for o in outcomes if o < -0.5]),
                    "num_draws": len([o for o in outcomes if abs(o) <= 0.5]),
                }
            )

        return stats
