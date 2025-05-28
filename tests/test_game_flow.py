#!/usr/bin/env python3
"""
Comprehensive tests for game flow including round transitions and game termination.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_state import create_game
from game.tile import Tile, TileColor


class TestRoundTransitions:
    """Test end-of-round transitions and scoring."""

    def test_round_end_detection(self):
        """Test detection of round end conditions."""
        game = create_game(num_players=2, seed=42)

        # Initially round should not be over
        assert (
            not game.factory_area.is_round_over()
        ), "Round should not be over initially"

        # Empty all factories and center
        for factory in game.factory_area.factories:
            factory.tiles = []
        game.factory_area.center.tiles = []
        game.factory_area.center.has_first_player_marker = False

        assert (
            game.factory_area.is_round_over()
        ), "Round should be over when all sources empty"

    def test_round_end_scoring(self):
        """Test scoring at end of round."""
        game = create_game(num_players=2, seed=42)

        # Set up completed pattern lines for both players
        game.players[0].place_tiles_on_pattern_line(0, [Tile(TileColor.BLUE)])
        game.players[1].place_tiles_on_pattern_line(1, [Tile(TileColor.RED)] * 2)

        initial_scores = [player.score for player in game.players]

        # Force round end
        game._end_round()

        # Scores should have changed
        final_scores = [player.score for player in game.players]
        assert final_scores[0] > initial_scores[0], "Player 0 should have scored"
        assert final_scores[1] > initial_scores[1], "Player 1 should have scored"

    def test_round_increment(self):
        """Test round number increments correctly."""
        game = create_game(num_players=2, seed=42)
        initial_round = game.round_number

        # Force round end without game end
        game._end_round()

        assert game.round_number == initial_round + 1, "Round number should increment"
        assert not game.game_over, "Game should not be over"

    def test_first_player_marker_transfer(self):
        """Test first player marker transfer between rounds."""
        game = create_game(num_players=2, seed=42)

        # Give first player marker to player 1
        game.players[1].place_tiles_on_floor_line([Tile.create_first_player_marker()])

        # Force round end
        game._end_round()

        # Player 1 should be current player in new round
        assert (
            game.current_player == 1
        ), "Player with first player marker should go first"
        assert not game.players[
            1
        ].has_first_player_marker(), "First player marker should be removed"

    def test_discard_pile_management(self):
        """Test discard pile management during round transitions."""
        game = create_game(num_players=2, seed=42)
        initial_discard_size = len(game.discard_pile)

        # Set up pattern lines that will generate discard tiles
        game.players[0].place_tiles_on_pattern_line(1, [Tile(TileColor.BLUE)] * 2)
        game.players[1].place_tiles_on_pattern_line(2, [Tile(TileColor.RED)] * 3)

        # Force round end
        game._end_round()

        # Discard pile should have grown
        assert (
            len(game.discard_pile) > initial_discard_size
        ), "Discard pile should have excess tiles"

    def test_new_round_setup(self):
        """Test new round setup after round end."""
        game = create_game(num_players=2, seed=42)

        # Force round end
        game._end_round()

        # Check new round setup
        assert (
            game.factory_area.center.has_first_player_marker
        ), "Center should have first player marker"

        # All factories should have tiles (if bag has enough)
        for factory in game.factory_area.factories:
            if len(game.bag) >= 4:
                assert len(factory.tiles) == 4, "Factory should have 4 tiles"


class TestGameTermination:
    """Test game termination conditions and final scoring."""

    def test_game_end_condition_completed_row(self):
        """Test game ends when player completes a row."""
        game = create_game(num_players=2, seed=42)

        # Complete a row for player 0
        for col in range(5):
            game.players[0].wall.filled[0][col] = True

        # Force round end
        game._end_round()

        assert game.game_over, "Game should end when player completes a row"
        assert game.winner is not None, "Winner should be determined"

    def test_game_continues_without_completed_row(self):
        """Test game continues when no rows are completed."""
        game = create_game(num_players=2, seed=42)

        # Partially fill rows but don't complete any
        game.players[0].wall.filled[0][0] = True
        game.players[0].wall.filled[0][1] = True
        game.players[1].wall.filled[1][0] = True

        # Force round end
        game._end_round()

        assert not game.game_over, "Game should continue without completed rows"

    def test_final_scoring_bonuses(self):
        """Test final scoring bonuses are applied."""
        game = create_game(num_players=2, seed=42)

        # Set up completed patterns for player 0
        # Complete first row
        for col in range(5):
            game.players[0].wall.filled[0][col] = True
        # Complete first column
        for row in range(5):
            game.players[0].wall.filled[row][0] = True
        # Complete blue color
        blue_positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        for row, col in blue_positions:
            game.players[0].wall.filled[row][col] = True

        initial_score = game.players[0].score

        # Force game end
        game._end_game()

        # Score should include bonuses (2 + 7 + 10 = 19)
        expected_bonus = 2 + 7 + 10  # row + column + color
        assert (
            game.players[0].score >= initial_score + expected_bonus
        ), "Should apply final bonuses"

    def test_winner_determination_by_score(self):
        """Test winner determination by highest score."""
        game = create_game(num_players=2, seed=42)

        # Set different scores
        game.players[0].score = 50
        game.players[1].score = 30

        # Complete a row to trigger game end
        for col in range(5):
            game.players[0].wall.filled[0][col] = True

        game._end_game()

        assert game.winner == 0, "Player with highest score should win"

    def test_tiebreaker_by_completed_rows(self):
        """Test tiebreaker using completed rows."""
        game = create_game(num_players=2, seed=42)

        # Set same scores
        game.players[0].score = 50
        game.players[1].score = 50

        # Player 0 has more completed rows
        for col in range(5):
            game.players[0].wall.filled[0][col] = True  # Row 0
            game.players[0].wall.filled[1][col] = True  # Row 1
            game.players[1].wall.filled[0][col] = True  # Only row 0

        game._end_game()

        assert game.winner == 0, "Player with more completed rows should win tiebreaker"

    def test_multiple_players_game_end(self):
        """Test game end with multiple players."""
        game = create_game(num_players=4, seed=42)

        # Set different scores
        game.players[0].score = 30
        game.players[1].score = 50  # Highest
        game.players[2].score = 40
        game.players[3].score = 35

        # Complete a row to trigger game end
        for col in range(5):
            game.players[1].wall.filled[0][col] = True

        game._end_game()

        assert game.winner == 1, "Player 1 should win with highest score"


class TestTurnManagement:
    """Test player turn management and transitions."""

    def test_initial_player_turn(self):
        """Test initial player turn setup."""
        game = create_game(num_players=3, seed=42)

        # Should start with player 0
        assert game.current_player == 0, "Should start with player 0"

    def test_turn_progression(self):
        """Test turn progression between players."""
        game = create_game(num_players=3, seed=42)

        # Apply actions and check turn progression
        for expected_player in [0, 1, 2, 0, 1]:  # Cycle through players
            assert (
                game.current_player == expected_player
            ), f"Should be player {expected_player}'s turn"

            # Apply a legal action
            actions = game.get_legal_actions()
            if actions and not game.factory_area.is_round_over():
                game.apply_action(actions[0])

    def test_turn_does_not_change_on_round_end(self):
        """Test turn doesn't change when round ends."""
        game = create_game(num_players=2, seed=42)

        # Empty all factories except one
        for i in range(len(game.factory_area.factories) - 1):
            game.factory_area.factories[i].tiles = []

        # Last factory has one tile
        game.factory_area.factories[-1].tiles = [Tile(TileColor.BLUE)]
        game.factory_area.center.tiles = []

        # Take last action (should end round)
        actions = game.get_legal_actions()
        game.apply_action(actions[0])

        # Current player should be determined by first player marker,
        # not turn progression
        # (This test verifies the round end logic takes precedence)
        assert (
            game.round_number > 1 or game.game_over
        ), "Round should have ended or game over"

    def test_no_turn_change_when_game_over(self):
        """Test no turn changes when game is over."""
        game = create_game(num_players=2, seed=42)
        game.game_over = True

        # Try to apply action (should fail)
        actions = game.get_legal_actions()
        assert len(actions) == 0, "Should have no legal actions when game over"

    def test_player_turn_validation(self):
        """Test player turn validation in legal actions."""
        game = create_game(num_players=3, seed=42)

        # Get legal actions for current player
        current_actions = game.get_legal_actions()

        # Get legal actions for other players
        other_player_actions = game.get_legal_actions((game.current_player + 1) % 3)

        # Actions should be based on player-specific board state
        # (This is more of a sanity check that the method accepts player_id parameter)
        assert isinstance(current_actions, list), "Should return list of actions"
        assert isinstance(other_player_actions, list), "Should return list of actions"


class TestComplexGameFlows:
    """Test complex game flow scenarios."""

    def test_complete_game_simulation(self):
        """Test a complete game from start to finish."""
        game = create_game(num_players=2, seed=42)

        max_actions = 1000  # Prevent infinite loops
        action_count = 0

        while not game.game_over and action_count < max_actions:
            actions = game.get_legal_actions()
            if not actions:
                break

            # Prefer pattern line actions over floor line actions
            # for better game progression
            pattern_actions = [a for a in actions if a.destination >= 0]
            if pattern_actions:
                chosen_action = pattern_actions[0]
            else:
                chosen_action = actions[0]

            # Apply chosen action
            success = game.apply_action(chosen_action)
            assert success, "Legal action should apply successfully"

            action_count += 1

        # Game should eventually end
        assert (
            action_count < max_actions
        ), "Game should end within reasonable number of actions"

        # Check final state
        if game.game_over:
            assert game.winner is not None, "Winner should be determined"
            assert all(
                player.score >= 0 for player in game.players
            ), "All scores should be non-negative"

    def test_bag_refill_from_discard(self):
        """Test bag refill from discard pile when needed."""
        game = create_game(num_players=2, seed=42)

        # Artificially reduce bag size
        game.bag = game.bag[:10]  # Only 10 tiles left
        game.discard_pile = [Tile(TileColor.BLUE)] * 20  # 20 tiles in discard

        _ = len(game.bag)  # initial_bag_size
        initial_discard_size = len(game.discard_pile)

        # Force new round (should trigger bag refill)
        game._start_new_round()

        # Discard pile should be consumed (moved to bag, then used for factories)
        assert (
            len(game.discard_pile) < initial_discard_size
        ), "Discard pile should be reduced"
        # The bag may not be larger because tiles are immediately consumed by factories
        # But the total tiles available should have increased
        total_factory_tiles = sum(
            len(factory.tiles) for factory in game.factory_area.factories
        )
        expected_tiles_used = game.factory_area.num_factories * 4
        assert (
            total_factory_tiles == expected_tiles_used
        ), "Factories should be properly filled"

    def test_multiple_round_progression(self):
        """Test progression through multiple rounds."""
        game = create_game(num_players=2, seed=42)

        initial_round = game.round_number
        rounds_to_simulate = 3

        for round_idx in range(rounds_to_simulate):
            # Play until round ends with safety counter
            max_actions_per_round = 100  # Prevent infinite loops
            action_count = 0

            while (
                not game.factory_area.is_round_over()
                and not game.game_over
                and action_count < max_actions_per_round
            ):
                actions = game.get_legal_actions()
                if actions:
                    # Prefer pattern line actions for better game progression
                    pattern_actions = [a for a in actions if a.destination >= 0]
                    if pattern_actions:
                        chosen_action = pattern_actions[0]
                    else:
                        chosen_action = actions[0]
                    game.apply_action(chosen_action)
                    action_count += 1
                else:
                    break

            # Ensure we don't get stuck in infinite loops
            assert (
                action_count < max_actions_per_round
            ), f"Round {round_idx + 1} took too many actions"

            if not game.game_over:
                game._end_round()

        expected_round = initial_round + rounds_to_simulate
        assert (
            game.round_number == expected_round or game.game_over
        ), "Should progress through multiple rounds or end game"

    def test_edge_case_empty_bag_and_discard(self):
        """Test edge case where both bag and discard pile are empty."""
        game = create_game(num_players=2, seed=42)

        # Empty both bag and discard pile
        game.bag = []
        game.discard_pile = []

        # Try to start new round
        game._start_new_round()

        # Should handle gracefully (factories will be empty)
        for factory in game.factory_area.factories:
            assert (
                len(factory.tiles) == 0
            ), "Factories should be empty when no tiles available"

    def test_first_player_marker_edge_cases(self):
        """Test first player marker edge cases."""
        game = create_game(num_players=3, seed=42)

        # Case 1: No one has first player marker (shouldn't happen in normal game)
        for player in game.players:
            player.floor_line = []

        original_player = game.current_player
        game._start_new_round()

        # Should maintain current player if no first player marker found
        assert game.current_player == original_player, "Should maintain current player"

        # Case 2: Multiple players have first player marker (shouldn't happen)
        game.players[0].place_tiles_on_floor_line([Tile.create_first_player_marker()])
        game.players[1].place_tiles_on_floor_line([Tile.create_first_player_marker()])

        game._start_new_round()

        # Should pick first player found
        assert game.current_player == 0, "Should pick first player with marker"


if __name__ == "__main__":
    pytest.main([__file__])
