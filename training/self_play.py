"""
Self-play engine for generating training data through Azul games.

This module implements the self-play loop that generates training data by playing
complete games using MCTS-guided neural network agents. It collects state-action
pairs during gameplay and assigns outcomes based on final game results.
"""

import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from agents.mcts import MCTSAgent
from game.game_state import Action, GameState
from training.neural_network import AzulNeuralNetwork
from training.replay_buffer import Experience, ReplayBuffer


class SelfPlayEngine:
    """
    Engine for generating training data through self-play games.

    This engine orchestrates complete Azul games between MCTS agents,
    collecting experiences and storing them in a replay buffer with
    appropriate outcome assignments.
    """

    def __init__(
        self,
        neural_network: AzulNeuralNetwork,
        replay_buffer: ReplayBuffer,
        mcts_simulations: int = 800,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
        max_game_length: int = 200,
        verbose: bool = False,
    ):
        """
        Initialize the self-play engine.

        Args:
            neural_network: Neural network for MCTS guidance
            replay_buffer: Buffer to store training experiences
            mcts_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection (higher = more exploration)
            temperature_threshold: Move number after which to use temperature=0
            max_game_length: Maximum number of moves before terminating game
            verbose: Whether to print detailed game information
        """
        self.neural_network = neural_network
        self.replay_buffer = replay_buffer
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.max_game_length = max_game_length
        self.verbose = verbose

        # Create MCTS agent
        self.agent = MCTSAgent(
            neural_network=neural_network,
            num_simulations=mcts_simulations,
            temperature=temperature,
        )

        # Statistics
        self.games_played = 0
        self.total_moves = 0
        self.total_game_time = 0.0

    def play_game(
        self, num_players: int = 2, seed: Optional[int] = None
    ) -> List[Experience]:
        """
        Play a complete self-play game and return collected experiences.

        Args:
            num_players: Number of players in the game
            seed: Random seed for reproducible games

        Returns:
            List of experiences collected during the game
        """
        if self.verbose:
            print(
                f"Starting self-play game {self.games_played + 1} with {num_players} players"
            )

        start_time = time.time()

        # Initialize game
        game_state = GameState(num_players=num_players, seed=seed)
        experiences = []
        move_count = 0

        # Play until game over or max length reached
        while not game_state.game_over and move_count < self.max_game_length:
            # Check if there are legal actions available
            legal_actions = game_state.get_legal_actions()
            if not legal_actions:
                # No legal actions available - this might mean the game is actually over
                # or there's a state representation issue
                if self.verbose:
                    print(f"No legal actions available at move {move_count}")
                break

            # Determine temperature for this move
            current_temp = (
                self.temperature if move_count < self.temperature_threshold else 0.0
            )

            # Get MCTS policy for current state
            mcts_policy = self._get_mcts_policy(game_state, current_temp)

            # Store experience (outcome will be set later)
            experience = Experience(
                state=game_state.copy(),
                mcts_policy=mcts_policy,
                player_id=game_state.current_player,
                outcome=None,
            )
            experiences.append(experience)

            # Select action based on MCTS policy
            action = self._select_action_from_policy(
                game_state, mcts_policy, current_temp
            )

            if self.verbose and move_count % 10 == 0:
                print(
                    f"Move {move_count}: Player {game_state.current_player} -> {action}"
                )

            # Apply action
            success = game_state.apply_action(action)
            if not success:
                if self.verbose:
                    print(f"Invalid action applied: {action}")
                break

            move_count += 1

        # Calculate game outcomes
        outcomes = self._calculate_outcomes(game_state)

        # Add experiences to replay buffer
        self.replay_buffer.add_game(experiences, outcomes)

        # Update statistics
        game_time = time.time() - start_time
        self.games_played += 1
        self.total_moves += move_count
        self.total_game_time += game_time

        if self.verbose:
            scores = game_state.get_scores()
            print(f"Game {self.games_played} finished:")
            print(f"  Moves: {move_count}")
            print(f"  Time: {game_time:.2f}s")
            print(f"  Scores: {scores}")
            print(f"  Outcomes: {outcomes}")
            print(f"  Experiences collected: {len(experiences)}")

        return experiences

    def _get_mcts_policy(self, game_state: GameState, temperature: float) -> np.ndarray:
        """
        Get MCTS-improved policy for the current game state.

        Args:
            game_state: Current game state
            temperature: Temperature for action selection

        Returns:
            Policy probabilities for all possible actions (full action space)
        """
        # Update agent temperature
        original_temp = self.agent.mcts.temperature
        self.agent.mcts.temperature = temperature

        # Get action probabilities for legal actions only
        try:
            legal_action_probs: np.ndarray = self.agent.get_action_probabilities(game_state)  # type: ignore[arg-type]
        finally:
            # Restore original temperature
            self.agent.mcts.temperature = original_temp

        # Convert to full action space policy vector
        legal_actions = game_state.get_legal_actions()
        full_policy = np.zeros(500)  # Action space size from neural network

        # Import here to avoid circular imports
        from game.pettingzoo_env import AzulAECEnv

        # Create temporary environment for action encoding
        temp_env = AzulAECEnv(num_players=game_state.num_players)

        # Map legal action probabilities to full action space
        for i, action in enumerate(legal_actions):
            try:
                action_idx = temp_env._encode_action(action)
                if 0 <= action_idx < 500 and i < len(legal_action_probs):
                    full_policy[action_idx] = legal_action_probs[i]
            except Exception:
                # Skip actions that can't be encoded
                continue

        return full_policy

    def _select_action_from_policy(
        self, game_state: GameState, policy: np.ndarray, temperature: float
    ) -> Action:
        """
        Select an action from the MCTS policy.

        Args:
            game_state: Current game state
            policy: Policy probabilities for full action space
            temperature: Temperature for selection

        Returns:
            Selected action
        """
        legal_actions = game_state.get_legal_actions()

        # Handle edge case where no legal actions are available
        if not legal_actions:
            raise ValueError("No legal actions available in the current game state")

        # Import here to avoid circular imports
        from game.pettingzoo_env import AzulAECEnv

        # Create temporary environment for action encoding
        temp_env = AzulAECEnv(num_players=game_state.num_players)

        # Extract probabilities for legal actions
        legal_action_probs_list = []
        for action in legal_actions:
            try:
                action_idx = temp_env._encode_action(action)
                if 0 <= action_idx < len(policy):
                    legal_action_probs_list.append(policy[action_idx])
                else:
                    legal_action_probs_list.append(0.0)
            except Exception:
                legal_action_probs_list.append(0.0)

        legal_action_probs: np.ndarray = np.array(legal_action_probs_list)

        # Normalize probabilities
        if legal_action_probs.sum() > 0:
            legal_action_probs = legal_action_probs / legal_action_probs.sum()
        else:
            # Fallback to uniform distribution
            legal_action_probs = np.ones(len(legal_actions)) / len(legal_actions)

        # Select action based on temperature
        if temperature == 0.0:
            # Deterministic selection (argmax)
            action_idx = int(np.argmax(legal_action_probs))
        else:
            # Stochastic selection with temperature
            if temperature != 1.0:
                scaled_probs = np.power(legal_action_probs, 1.0 / temperature)
                scaled_probs = scaled_probs / np.sum(scaled_probs)
            else:
                scaled_probs = legal_action_probs

            # Sample from distribution
            action_idx = int(np.random.choice(len(legal_actions), p=scaled_probs))

        return legal_actions[action_idx]

    def _calculate_outcomes(self, final_game_state: GameState) -> List[float]:
        """
        Calculate game outcomes for each player.

        Args:
            final_game_state: Final state of the completed game

        Returns:
            List of outcomes for each player (1.0 for win, 0.0 for draw, -1.0 for loss)
        """
        scores = final_game_state.get_scores()
        max_score = max(scores)

        # Count how many players achieved the maximum score
        winners = [i for i, score in enumerate(scores) if score == max_score]

        outcomes = []
        for i, score in enumerate(scores):
            if score == max_score:
                if len(winners) == 1:
                    # Clear winner
                    outcomes.append(1.0)
                else:
                    # Draw
                    outcomes.append(0.0)
            else:
                # Loss
                outcomes.append(-1.0)

        return outcomes

    def play_games(
        self,
        num_games: int,
        num_players: int = 2,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> List[List[Experience]]:
        """
        Play multiple self-play games.

        Args:
            num_games: Number of games to play
            num_players: Number of players per game
            progress_callback: Optional callback for progress updates

        Returns:
            List of experience lists (one per game)
        """
        all_experiences = []

        for game_idx in range(num_games):
            # Use different seeds for variety
            seed = self.games_played + game_idx + 42

            # Play game
            experiences = self.play_game(num_players=num_players, seed=seed)
            all_experiences.append(experiences)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(game_idx + 1, num_games, self.get_statistics())

            # Print progress occasionally
            if self.verbose and (game_idx + 1) % 10 == 0:
                stats = self.get_statistics()
                print(f"Completed {game_idx + 1}/{num_games} games")
                print(f"  Avg moves/game: {stats['avg_moves_per_game']:.1f}")
                print(f"  Avg time/game: {stats['avg_time_per_game']:.2f}s")
                print(f"  Buffer size: {self.replay_buffer.size()}")

        return all_experiences

    def get_statistics(self) -> dict:
        """
        Get statistics about self-play performance.

        Returns:
            Dictionary containing performance statistics
        """
        stats = {
            "games_played": self.games_played,
            "total_moves": self.total_moves,
            "total_time": self.total_game_time,
            "avg_moves_per_game": self.total_moves / max(1, self.games_played),
            "avg_time_per_game": self.total_game_time / max(1, self.games_played),
            "moves_per_second": self.total_moves / max(1e-6, self.total_game_time),
        }

        # Add replay buffer statistics
        buffer_stats = self.replay_buffer.get_statistics()
        stats.update({f"buffer_{key}": value for key, value in buffer_stats.items()})

        return stats

    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.games_played = 0
        self.total_moves = 0
        self.total_game_time = 0.0
