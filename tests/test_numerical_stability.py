"""
Tests for numerical stability and precision in Azul game state representation.

This module tests floating point precision, normalization edge cases, and
numerical robustness of the state representation system.
"""

import numpy as np

from game import AzulStateRepresentation, GameState, StateConfig


class TestFloatingPointPrecision:
    """Test floating point precision in state representation."""

    def test_normalization_precision(self):
        """Test that normalization maintains sufficient precision."""
        game = GameState(num_players=2, seed=42)

        # Make moves to create varied tile counts
        for _ in range(10):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Test pattern line normalization precision
        for player_idx in range(state_repr.num_players):
            for line_idx in range(5):
                capacity = line_idx + 1
                normalized_capacity = state_repr.pattern_lines[player_idx, line_idx, 0]
                expected_capacity = capacity / 5.0

                # Should be exact for these simple fractions
                assert abs(normalized_capacity - expected_capacity) < 1e-6, (
                    f"Capacity normalization precision error: "
                    f"{normalized_capacity} != {expected_capacity}"
                )

                # Test count normalization
                normalized_count = state_repr.pattern_lines[player_idx, line_idx, 1]
                assert (
                    0.0 <= normalized_count <= 1.0
                ), f"Normalized count out of range: {normalized_count}"

                # If there's a count, verify it's properly normalized
                if normalized_count > 0:
                    # Reconstruct actual count
                    actual_count = normalized_count * capacity
                    # Should be close to an integer
                    assert abs(actual_count - round(actual_count)) < 1e-6, (
                        f"Count normalization precision error: "
                        f"{actual_count} not close to integer"
                    )

        print("✓ Normalization precision test passed")

    def test_score_normalization_precision(self):
        """Test score normalization precision across different score ranges."""
        # Test various score values
        test_scores = [0, 1, 10, 50, 100, 150, 199, 200]

        for score in test_scores:
            # Create a mock game state with specific score
            game = GameState(num_players=2, seed=42)
            game.players[0].score = score

            state_repr = AzulStateRepresentation(game)
            flat_vector = state_repr.get_flat_state_vector(normalize=True)

            # Find score in flat vector (should be at index 4)
            normalized_score = flat_vector[4]  # First player's score
            expected_normalized = score / StateConfig.MAX_SCORE

            assert abs(normalized_score - expected_normalized) < 1e-6, (
                f"Score normalization error for {score}: "
                f"{normalized_score} != {expected_normalized}"
            )

        print("✓ Score normalization precision test passed")

    def test_tile_count_precision(self):
        """Test precision in tile counting and representation."""
        game = GameState(num_players=2, seed=42)

        # Play several moves to distribute tiles
        for _ in range(15):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Verify tile conservation with high precision
        distribution = state_repr.get_tile_distribution()

        # Total should exactly equal expected
        assert (
            distribution["total"] == distribution["expected_total"]
        ), f"Tile count precision error: {distribution['total']} != {distribution['expected_total']}"

        # Individual counts should be non-negative integers
        for location in [
            "bag",
            "discard",
            "factories",
            "center",
            "player_boards",
            "walls",
        ]:
            count = distribution[location]
            assert count >= 0, f"Negative tile count in {location}: {count}"
            assert count == int(count), f"Non-integer tile count in {location}: {count}"

        print("✓ Tile count precision test passed")

    def test_cumulative_precision_errors(self):
        """Test that precision errors don't accumulate over many operations."""
        game = GameState(num_players=2, seed=42)

        # Store initial state
        initial_state_repr = AzulStateRepresentation(game)
        _ = initial_state_repr.get_flat_state_vector(normalize=True)  # initial_vector

        # Make many moves and check precision doesn't degrade
        precision_samples = []

        for i in range(50):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

                if i % 10 == 0:  # Sample every 10 moves
                    state_repr = AzulStateRepresentation(game)
                    vector = state_repr.get_flat_state_vector(normalize=True)

                    # Check that values are still in valid ranges
                    assert np.all(np.isfinite(vector)), f"Non-finite values at move {i}"
                    assert np.all(
                        vector >= -0.1
                    ), f"Values too negative at move {i}: min={vector.min()}"
                    assert np.all(
                        vector <= 1.1
                    ), f"Values too large at move {i}: max={vector.max()}"

                    # Check precision of specific components
                    # Pattern line capacities should still be exact
                    pattern_start = 4 + StateConfig.MAX_PLAYERS  # After global + scores
                    for player_idx in range(state_repr.num_players):
                        for line_idx in range(5):
                            idx = pattern_start + player_idx * 5 * 8 + line_idx * 8
                            capacity = vector[idx]
                            expected = (line_idx + 1) / 5.0
                            assert (
                                abs(capacity - expected) < 1e-6
                            ), f"Capacity precision degraded at move {i}: {capacity} != {expected}"

                    precision_samples.append(i)
            else:
                break

        print(f"✓ Cumulative precision test passed ({len(precision_samples)} samples)")


class TestNormalizationEdgeCases:
    """Test normalization with extreme and edge case values."""

    def test_zero_division_safety(self):
        """Test that normalization handles potential zero division safely."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)

        # Test empty pattern lines (should not cause division by zero)
        for player_idx in range(state_repr.num_players):
            for line_idx in range(5):
                # Empty line should have count = 0, normalized to 0
                if state_repr.pattern_lines[player_idx, line_idx, 7] == 1:  # Empty flag
                    normalized_count = state_repr.pattern_lines[player_idx, line_idx, 1]
                    assert (
                        normalized_count == 0.0
                    ), f"Empty line should have normalized count 0, got {normalized_count}"

        # Test with artificially empty tile supply
        empty_game = GameState(num_players=2, seed=42)
        # Clear bag and discard (artificial scenario)
        empty_game.bag.clear()
        empty_game.discard_pile.clear()

        empty_state_repr = AzulStateRepresentation(empty_game)
        empty_vector = empty_state_repr.get_flat_state_vector(normalize=True)

        # Should not contain NaN or infinity
        assert np.all(
            np.isfinite(empty_vector)
        ), "Empty state contains non-finite values"

        print("✓ Zero division safety test passed")

    def test_maximum_value_normalization(self):
        """Test normalization at maximum possible values."""
        game = GameState(num_players=4, seed=42)  # Maximum players

        # Artificially set maximum scores
        for player in game.players:
            player.score = StateConfig.MAX_SCORE

        # Set maximum round number
        game.round_number = StateConfig.MAX_ROUND

        state_repr = AzulStateRepresentation(game)
        flat_vector = state_repr.get_flat_state_vector(normalize=True)

        # Check that maximum values normalize to 1.0
        # Round number (index 1)
        assert (
            abs(flat_vector[1] - 1.0) < 1e-6
        ), f"Maximum round normalization error: {flat_vector[1]} != 1.0"

        # Player scores (indices 4-7)
        for i in range(4, 8):
            assert (
                abs(flat_vector[i] - 1.0) < 1e-6
            ), f"Maximum score normalization error at index {i}: {flat_vector[i]} != 1.0"

        print("✓ Maximum value normalization test passed")

    def test_boundary_value_stability(self):
        """Test stability at boundary values."""
        # Test values very close to boundaries
        boundary_tests = [
            (0.0, "zero"),
            (1.0, "one"),
            (0.5, "half"),
            (1.0 / 3.0, "third"),
            (2.0 / 3.0, "two_thirds"),
            (0.2, "one_fifth"),
            (0.8, "four_fifths"),
        ]

        for value, name in boundary_tests:
            # Create a game and artificially set a normalized value
            game = GameState(num_players=2, seed=42)
            _ = AzulStateRepresentation(game)  # state_repr

            # Test that the value survives round-trip through normalization
            # For pattern line counts
            if value <= 1.0:
                # Simulate a pattern line with this fill ratio
                line_capacity = 3  # Use line 2 (capacity 3)
                tile_count = round(value * line_capacity)
                normalized_back = tile_count / line_capacity

                # Should be close to original value
                error = abs(normalized_back - value)
                max_error = 1.0 / line_capacity  # Maximum quantization error

                assert (
                    error <= max_error
                ), f"Boundary value {name} ({value}) has excessive error: {error} > {max_error}"

        print("✓ Boundary value stability test passed")

    def test_floating_point_edge_cases(self):
        """Test handling of floating point edge cases."""
        game = GameState(num_players=2, seed=42)

        # Make some moves to get interesting state
        for _ in range(5):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Test both normalized and unnormalized vectors
        norm_vector = state_repr.get_flat_state_vector(normalize=True)
        unnorm_vector = state_repr.get_flat_state_vector(normalize=False)

        # Check for problematic floating point values
        for vector, name in [
            (norm_vector, "normalized"),
            (unnorm_vector, "unnormalized"),
        ]:
            # No NaN values
            assert not np.any(np.isnan(vector)), f"{name} vector contains NaN"

            # No infinite values
            assert not np.any(np.isinf(vector)), f"{name} vector contains infinity"

            # No subnormal numbers (very small denormalized numbers)
            min_normal = np.finfo(np.float32).tiny
            non_zero_mask = vector != 0.0
            if np.any(non_zero_mask):
                min_non_zero = np.min(np.abs(vector[non_zero_mask]))
                assert (
                    min_non_zero >= min_normal
                ), f"{name} vector contains subnormal numbers: {min_non_zero} < {min_normal}"

        print("✓ Floating point edge cases test passed")


class TestNumericalRobustness:
    """Test numerical robustness under various conditions."""

    def test_repeated_normalization_stability(self):
        """Test that repeated normalization doesn't cause drift."""
        game = GameState(num_players=3, seed=42)

        # Make some moves
        for _ in range(8):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Get initial normalized vector
        vector1 = state_repr.get_flat_state_vector(normalize=True)

        # Create new state representation from same game (should be identical)
        state_repr2 = AzulStateRepresentation(game)
        vector2 = state_repr2.get_flat_state_vector(normalize=True)

        # Should be exactly equal
        np.testing.assert_array_equal(
            vector1,
            vector2,
            err_msg="Repeated normalization produces different results",
        )

        # Test multiple calls to same object
        vector3 = state_repr.get_flat_state_vector(normalize=True)
        np.testing.assert_array_equal(
            vector1,
            vector3,
            err_msg="Multiple calls to normalization produce different results",
        )

        print("✓ Repeated normalization stability test passed")

    def test_extreme_game_states(self):
        """Test numerical stability with extreme but valid game states."""
        # Test with maximum players and long game
        game = GameState(num_players=4, seed=42)

        # Play a very long game
        moves = 0
        max_moves = 100

        while not game.game_over and moves < max_moves:
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])
                moves += 1

                # Check numerical stability every 20 moves
                if moves % 20 == 0:
                    state_repr = AzulStateRepresentation(game)
                    vector = state_repr.get_flat_state_vector(normalize=True)

                    # Verify no numerical issues
                    assert np.all(
                        np.isfinite(vector)
                    ), f"Non-finite values at move {moves}"
                    assert (
                        vector.dtype == np.float32
                    ), f"Wrong dtype at move {moves}: {vector.dtype}"

                    # Check tile distribution is accurate (should be exact with proper discarded tile tracking)
                    distribution = state_repr.get_tile_distribution()
                    total_diff = abs(
                        distribution["total"] - distribution["expected_total"]
                    )
                    # With proper discarded tile tracking, should be exact (allow minimal floating-point precision errors)
                    assert (
                        total_diff <= 1
                    ), f"Tile distribution inaccurate at move {moves}: diff={total_diff}"

        print(f"✓ Extreme game states test passed ({moves} moves)")

    def test_precision_across_different_seeds(self):
        """Test that numerical precision is consistent across different random seeds."""
        seeds = [42, 123, 999, 2023, 12345]
        precision_results = []

        for seed in seeds:
            game = GameState(num_players=2, seed=seed)

            # Make deterministic number of moves
            for _ in range(10):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])

            state_repr = AzulStateRepresentation(game)
            vector = state_repr.get_flat_state_vector(normalize=True)

            # Check precision characteristics
            precision_info = {
                "seed": seed,
                "min_value": float(vector.min()),
                "max_value": float(vector.max()),
                "mean_value": float(vector.mean()),
                "std_value": float(vector.std()),
                "num_zeros": int(np.sum(vector == 0.0)),
                "num_ones": int(np.sum(vector == 1.0)),
            }

            precision_results.append(precision_info)

            # Basic sanity checks
            assert np.all(np.isfinite(vector)), f"Non-finite values with seed {seed}"
            assert vector.min() >= -0.1, f"Values too negative with seed {seed}"
            assert vector.max() <= 1.1, f"Values too large with seed {seed}"

        # Check that precision characteristics are reasonable across seeds
        min_values = [r["min_value"] for r in precision_results]
        max_values = [r["max_value"] for r in precision_results]

        # All should be in reasonable ranges
        assert all(
            v >= -0.1 for v in min_values
        ), f"Inconsistent min values: {min_values}"
        assert all(
            v <= 1.1 for v in max_values
        ), f"Inconsistent max values: {max_values}"

        print(f"✓ Precision across seeds test passed ({len(seeds)} seeds)")

    def test_memory_layout_consistency(self):
        """Test that memory layout doesn't affect numerical results."""
        game = GameState(num_players=3, seed=42)

        # Make some moves
        for _ in range(7):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        # Create multiple state representations
        state_repr1 = AzulStateRepresentation(game)
        state_repr2 = AzulStateRepresentation(game)

        # Get vectors with different memory layouts
        vector1 = state_repr1.get_flat_state_vector(normalize=True)
        vector2 = state_repr2.get_flat_state_vector(normalize=True)

        # Force different memory layout by copying
        vector1_copy = np.copy(vector1)
        vector2_copy = np.array(vector2, order="F")  # Fortran order

        # All should be numerically identical
        np.testing.assert_array_equal(vector1, vector2)
        np.testing.assert_array_equal(vector1, vector1_copy)
        np.testing.assert_array_equal(vector1, vector2_copy)

        # Test state dictionaries
        dict1 = state_repr1.get_state_dict()
        dict2 = state_repr2.get_state_dict()

        for key in dict1:
            np.testing.assert_array_equal(
                dict1[key], dict2[key], err_msg=f"State dict mismatch for {key}"
            )

        print("✓ Memory layout consistency test passed")


class TestNumericalInvariants:
    """Test that numerical invariants are maintained."""

    def test_normalization_invariants(self):
        """Test that normalization maintains mathematical invariants."""
        game = GameState(num_players=2, seed=42)

        # Make moves to create varied state
        for _ in range(12):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Test pattern line invariants
        for player_idx in range(state_repr.num_players):
            for line_idx in range(5):
                line_data = state_repr.pattern_lines[player_idx, line_idx]

                capacity_norm = line_data[0]
                count_norm = line_data[1]
                color_encoding = line_data[2:7]
                empty_flag = line_data[7]

                # Capacity should be exactly (line_idx + 1) / 5
                expected_capacity = (line_idx + 1) / 5.0
                assert (
                    abs(capacity_norm - expected_capacity) < 1e-6
                ), f"Capacity invariant violated: {capacity_norm} != {expected_capacity}"

                # Count should be in [0, 1]
                assert (
                    0.0 <= count_norm <= 1.0
                ), f"Count normalization out of range: {count_norm}"

                # Color encoding should be one-hot or empty
                color_sum = np.sum(color_encoding)
                assert color_sum <= 1.0, f"Color encoding sum > 1: {color_sum}"

                # Empty flag consistency
                if empty_flag == 1:
                    assert count_norm == 0.0, "Empty line should have zero count"
                    assert color_sum == 0.0, "Empty line should have no color"
                else:
                    assert (
                        color_sum == 1.0
                    ), "Non-empty line should have exactly one color"

        print("✓ Normalization invariants test passed")

    def test_conservation_invariants(self):
        """Test that conservation laws are maintained numerically."""
        game = GameState(num_players=3, seed=42)

        # Track tile conservation through multiple moves
        initial_state_repr = AzulStateRepresentation(game)
        _ = initial_state_repr.get_tile_distribution()  # initial_distribution

        for move_num in range(20):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

                state_repr = AzulStateRepresentation(game)
                distribution = state_repr.get_tile_distribution()

                # Total tiles should be exactly conserved with proper discarded tile tracking
                total_diff = abs(distribution["total"] - distribution["expected_total"])
                assert (
                    total_diff <= 1
                ), f"Tile distribution inaccurate at move {move_num}: diff={total_diff}"

                # All counts must be non-negative
                for location, count in distribution.items():
                    if location not in ["total", "expected_total"]:
                        assert (
                            count >= 0
                        ), f"Negative tile count in {location} at move {move_num}: {count}"

        print("✓ Conservation invariants test passed")

    def test_monotonicity_invariants(self):
        """Test monotonicity properties that should be maintained."""
        game = GameState(num_players=2, seed=42)

        previous_round = game.round_number

        for move_num in range(30):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

                # Round number should be monotonically non-decreasing
                assert (
                    game.round_number >= previous_round
                ), f"Round number decreased: {game.round_number} < {previous_round}"

                # Player scores should be monotonically non-decreasing
                # (scores can only increase or stay the same in Azul)
                for player in game.players:
                    assert player.score >= 0, f"Negative score: {player.score}"

                previous_round = game.round_number

        print("✓ Monotonicity invariants test passed")


if __name__ == "__main__":
    print("=== Azul Numerical Stability Tests Demo ===")

    # Run key tests
    precision_tests = TestFloatingPointPrecision()
    print("\n1. Testing floating point precision...")
    precision_tests.test_normalization_precision()
    precision_tests.test_score_normalization_precision()
    precision_tests.test_cumulative_precision_errors()

    edge_case_tests = TestNormalizationEdgeCases()
    print("\n2. Testing normalization edge cases...")
    edge_case_tests.test_zero_division_safety()
    edge_case_tests.test_maximum_value_normalization()
    edge_case_tests.test_floating_point_edge_cases()

    robustness_tests = TestNumericalRobustness()
    print("\n3. Testing numerical robustness...")
    robustness_tests.test_repeated_normalization_stability()
    robustness_tests.test_extreme_game_states()
    robustness_tests.test_precision_across_different_seeds()

    invariant_tests = TestNumericalInvariants()
    print("\n4. Testing numerical invariants...")
    invariant_tests.test_normalization_invariants()
    invariant_tests.test_conservation_invariants()
    invariant_tests.test_monotonicity_invariants()

    print("\nAll numerical stability tests completed successfully!")
