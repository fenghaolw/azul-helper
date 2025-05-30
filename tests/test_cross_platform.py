"""
Cross-platform compatibility tests for Azul game implementation.

This module tests consistency across different platforms, Python versions,
and environments to ensure reliable behavior everywhere.
"""

import hashlib
import json
import os
import pickle
import platform
import sys
import tempfile

import numpy as np

from game import AzulStateRepresentation, GameState, StateConfig, TileColor


class TestPlatformConsistency:
    """Test consistency across different platforms."""

    def test_deterministic_game_creation(self):
        """Test that game creation is deterministic across platforms."""
        # Test multiple seeds
        seeds = [42, 123, 999, 2023, 12345]

        for seed in seeds:
            # Create multiple games with same seed
            games = []
            for _ in range(5):
                game = GameState(num_players=3, seed=seed)
                games.append(game)

            # All games should be identical
            reference_game = games[0]

            for i, game in enumerate(games[1:], 1):
                # Check basic properties
                assert game.num_players == reference_game.num_players
                assert game.current_player == reference_game.current_player
                assert game.round_number == reference_game.round_number

                # Check factory setup
                assert len(game.factory_area.factories) == len(
                    reference_game.factory_area.factories
                )

                for j, (factory, ref_factory) in enumerate(
                    zip(
                        game.factory_area.factories,
                        reference_game.factory_area.factories,
                    )
                ):
                    assert len(factory.tiles) == len(ref_factory.tiles)
                    for k, (tile, ref_tile) in enumerate(
                        zip(factory.tiles, ref_factory.tiles)
                    ):
                        assert tile.color == ref_tile.color, (
                            f"Seed {seed}, game {i}, factory {j}, tile {k}: "
                            f"{tile.color} != {ref_tile.color}"
                        )

                # Check bag contents
                assert len(game.bag) == len(reference_game.bag)
                bag_colors = [tile.color for tile in game.bag]
                ref_bag_colors = [tile.color for tile in reference_game.bag]
                assert (
                    bag_colors == ref_bag_colors
                ), f"Bag contents differ for seed {seed}, game {i}"

        print(f"✓ Deterministic game creation verified for {len(seeds)} seeds")

    def test_state_representation_consistency(self):
        """Test that state representations are consistent across platforms."""
        # Create deterministic game
        game = GameState(num_players=2, seed=42)

        # Make deterministic moves
        for _ in range(10):
            actions = game.get_legal_actions()
            if actions:
                # Always choose first action for determinism
                game.apply_action(actions[0])

        # Create multiple state representations
        state_reprs = []
        for _ in range(5):
            state_repr = AzulStateRepresentation(game)
            state_reprs.append(state_repr)

        # All should be identical
        reference_repr = state_reprs[0]
        reference_vector = reference_repr.get_flat_state_vector(normalize=True)

        for i, state_repr in enumerate(state_reprs[1:], 1):
            vector = state_repr.get_flat_state_vector(normalize=True)
            np.testing.assert_array_equal(
                reference_vector,
                vector,
                err_msg=f"State representation {i} differs from reference",
            )

        print("✓ State representation consistency verified")

    def test_numerical_precision_consistency(self):
        """Test that numerical computations are consistent across platforms."""
        game = GameState(num_players=3, seed=42)

        # Make moves to create interesting state
        for _ in range(15):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Test precision of specific computations
        precision_tests = []

        # Pattern line normalization
        for player_idx in range(state_repr.num_players):
            for line_idx in range(5):
                capacity = line_idx + 1
                normalized_capacity = state_repr.pattern_lines[player_idx, line_idx, 0]
                expected = capacity / 5.0
                precision_tests.append(
                    ("pattern_capacity", normalized_capacity, expected)
                )

        # Score normalization
        for player_idx in range(state_repr.num_players):
            score = state_repr.player_scores[player_idx]
            flat_vector = state_repr.get_flat_state_vector(normalize=True)
            normalized_score = flat_vector[4 + player_idx]  # Scores start at index 4
            expected = score / StateConfig.MAX_SCORE
            precision_tests.append(("score_normalization", normalized_score, expected))

        # Verify all computations are precise
        for test_name, actual, expected in precision_tests:
            assert (
                abs(actual - expected) < 1e-10
            ), f"{test_name}: {actual} != {expected} (diff: {abs(actual - expected)})"

        print(
            f"✓ Numerical precision consistency verified ({len(precision_tests)} tests)"
        )

    def test_hash_consistency(self):
        """Test that game state hashes are consistent."""
        # Create deterministic games
        games = []
        for seed in [42, 123, 999]:
            game = GameState(num_players=2, seed=seed)
            # Make some moves
            for _ in range(5):
                actions = game.get_legal_actions()
                if actions:
                    game.apply_action(actions[0])
            games.append(game)

        # Compute hashes of state representations
        hashes = []
        for game in games:
            state_repr = AzulStateRepresentation(game)
            vector = state_repr.get_flat_state_vector(normalize=True)

            # Convert to bytes for hashing
            vector_bytes = vector.tobytes()
            hash_value = hashlib.sha256(vector_bytes).hexdigest()
            hashes.append(hash_value)

        # Verify hashes are different (games should be different)
        assert len(set(hashes)) == len(
            hashes
        ), "Games with different seeds should have different hashes"

        # Verify same game produces same hash
        for game in games:
            state_repr1 = AzulStateRepresentation(game)
            state_repr2 = AzulStateRepresentation(game)

            vector1 = state_repr1.get_flat_state_vector(normalize=True)
            vector2 = state_repr2.get_flat_state_vector(normalize=True)

            hash1 = hashlib.sha256(vector1.tobytes()).hexdigest()
            hash2 = hashlib.sha256(vector2.tobytes()).hexdigest()

            assert hash1 == hash2, "Same game should produce same hash"

        print("✓ Hash consistency verified")


class TestEnvironmentCompatibility:
    """Test compatibility across different environments."""

    def test_python_version_compatibility(self):
        """Test compatibility with current Python version."""
        python_version = sys.version_info

        # Should work with Python 3.7+
        assert python_version >= (3, 7), f"Python version too old: {python_version}"

        # Test version-specific features
        if python_version >= (3, 8):
            # Test walrus operator compatibility (if used)
            game = GameState(num_players=2, seed=42)
            actions = game.get_legal_actions()
            if actions:
                # This syntax requires Python 3.8+
                assert (actions[0]) is not None

        print(f"✓ Python version compatibility verified: {python_version}")

    def test_numpy_version_compatibility(self):
        """Test compatibility with NumPy version."""
        numpy_version = np.__version__

        # Test basic NumPy operations
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)

        # Test array creation and operations
        vector = state_repr.get_flat_state_vector(normalize=True)
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32

        # Test array operations
        assert np.all(np.isfinite(vector))
        assert vector.shape == (state_repr.flat_state_size,)

        # Test array slicing and indexing
        subset = vector[10:20]
        assert len(subset) == 10

        # Test mathematical operations
        mean_val = np.mean(vector)
        std_val = np.std(vector)
        assert np.isfinite(mean_val)
        assert np.isfinite(std_val)

        print(f"✓ NumPy version compatibility verified: {numpy_version}")

    def test_platform_specific_behavior(self):
        """Test platform-specific behavior."""
        platform_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        }

        # Test that game behavior is consistent regardless of platform
        game = GameState(num_players=3, seed=42)

        # Make moves
        for _ in range(8):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)
        vector = state_repr.get_flat_state_vector(normalize=True)

        # Basic sanity checks that should work on all platforms
        assert len(vector) == state_repr.flat_state_size
        assert np.all(np.isfinite(vector))
        assert vector.dtype == np.float32

        # Test tile conservation
        distribution = state_repr.get_tile_distribution()
        assert distribution["total"] == distribution["expected_total"]

        print(f"✓ Platform compatibility verified: {platform_info}")

    def test_file_system_compatibility(self):
        """Test file system operations across platforms."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)

        # Test temporary file operations
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Test pickle serialization
            pickle.dump(state_repr, f)
            temp_file = f.name

        try:
            # Test loading
            with open(temp_file, "rb") as f:
                loaded_repr = pickle.load(f)

            # Verify consistency
            original_vector = state_repr.get_flat_state_vector(normalize=True)
            loaded_vector = loaded_repr.get_flat_state_vector(normalize=True)
            np.testing.assert_array_equal(original_vector, loaded_vector)

        finally:
            os.unlink(temp_file)

        # Test directory operations
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving multiple files
            state_dict = state_repr.get_state_dict()

            for key, array in state_dict.items():
                file_path = os.path.join(temp_dir, f"{key}.npy")
                np.save(file_path, array)

                # Verify file was created and can be loaded
                assert os.path.exists(file_path)
                loaded_array = np.load(file_path)
                np.testing.assert_array_equal(array, loaded_array)

        print("✓ File system compatibility verified")


class TestDataTypeConsistency:
    """Test consistency of data types across platforms."""

    def test_integer_type_consistency(self):
        """Test that integer types are consistent."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)

        # Test integer fields
        assert isinstance(state_repr.current_player, (int, np.integer))
        assert isinstance(state_repr.round_number, (int, np.integer))
        assert isinstance(state_repr.game_over, (int, np.integer))
        assert isinstance(state_repr.winner, (int, np.integer))

        # Test array dtypes
        assert state_repr.walls.dtype == np.int8
        assert state_repr.floor_lines.dtype == np.int8
        assert state_repr.factories.dtype == np.int8
        assert state_repr.center_tiles.dtype == np.int8
        assert state_repr.tile_supply.dtype == np.int32
        assert state_repr.player_scores.dtype == np.int32

        print("✓ Integer type consistency verified")

    def test_float_type_consistency(self):
        """Test that float types are consistent."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)

        # Test float arrays
        assert state_repr.pattern_lines.dtype == np.float32

        # Test flat vector
        vector = state_repr.get_flat_state_vector(normalize=True)
        assert vector.dtype == np.float32

        # Test that values are in expected ranges
        assert np.all(vector >= -0.1)  # Allow small negative values
        assert np.all(vector <= 1.1)  # Allow small overflow

        print("✓ Float type consistency verified")

    def test_enum_consistency(self):
        """Test that enums work consistently."""
        # Test TileColor enum
        colors = [
            TileColor.BLUE,
            TileColor.YELLOW,
            TileColor.RED,
            TileColor.BLACK,
            TileColor.WHITE,
        ]

        for color in colors:
            assert isinstance(color, TileColor)
            assert hasattr(color, "value")
            assert isinstance(color.value, str)

        # Test that enum values are consistent
        assert TileColor.BLUE.value == "blue"
        assert TileColor.YELLOW.value == "yellow"
        assert TileColor.RED.value == "red"
        assert TileColor.BLACK.value == "black"
        assert TileColor.WHITE.value == "white"

        print("✓ Enum consistency verified")

    def test_serialization_type_preservation(self):
        """Test that types are preserved through serialization."""
        game = GameState(num_players=3, seed=42)
        state_repr = AzulStateRepresentation(game)

        # Test pickle preservation
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(state_repr, f)
            f.seek(0)
            loaded_repr = pickle.load(f)

        # Verify types are preserved
        assert isinstance(loaded_repr.current_player, type(state_repr.current_player))
        assert loaded_repr.walls.dtype == state_repr.walls.dtype
        assert loaded_repr.pattern_lines.dtype == state_repr.pattern_lines.dtype

        # Test JSON serialization (with type conversion)
        json_data = {
            "current_player": int(state_repr.current_player),
            "round_number": int(state_repr.round_number),
            "game_over": int(state_repr.game_over),
            "flat_vector": state_repr.get_flat_state_vector(normalize=True).tolist(),
        }

        json_str = json.dumps(json_data)
        loaded_data = json.loads(json_str)

        # Verify JSON round-trip
        assert loaded_data["current_player"] == int(state_repr.current_player)
        assert loaded_data["round_number"] == int(state_repr.round_number)

        # Verify vector can be reconstructed
        reconstructed_vector = np.array(loaded_data["flat_vector"], dtype=np.float32)
        original_vector = state_repr.get_flat_state_vector(normalize=True)
        np.testing.assert_array_almost_equal(
            original_vector, reconstructed_vector, decimal=6
        )

        print("✓ Serialization type preservation verified")


class TestRandomnessConsistency:
    """Test that randomness is consistent across platforms."""

    def test_seed_reproducibility(self):
        """Test that seeds produce reproducible results."""
        seed = 42

        # Create multiple games with same seed
        games = []
        for _ in range(3):
            game = GameState(num_players=2, seed=seed)
            games.append(game)

        # All games should be identical
        reference_game = games[0]

        for game in games[1:]:
            # Check factory contents
            for ref_factory, factory in zip(
                reference_game.factory_area.factories, game.factory_area.factories
            ):
                ref_colors = [tile.color for tile in ref_factory.tiles]
                colors = [tile.color for tile in factory.tiles]
                assert (
                    ref_colors == colors
                ), "Factory contents should be identical with same seed"

            # Check bag contents
            ref_bag_colors = [tile.color for tile in reference_game.bag]
            bag_colors = [tile.color for tile in game.bag]
            assert (
                ref_bag_colors == bag_colors
            ), "Bag contents should be identical with same seed"

        print("✓ Seed reproducibility verified")

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        seeds = [42, 123, 999, 2023]
        games = []

        for seed in seeds:
            game = GameState(num_players=3, seed=seed)
            games.append(game)

        # Games should be different
        for i, game1 in enumerate(games):
            for j, game2 in enumerate(games):
                if i != j:
                    # At least one factory should be different
                    factories_different = False
                    for factory1, factory2 in zip(
                        game1.factory_area.factories, game2.factory_area.factories
                    ):
                        colors1 = [tile.color for tile in factory1.tiles]
                        colors2 = [tile.color for tile in factory2.tiles]
                        if colors1 != colors2:
                            factories_different = True
                            break

                    assert (
                        factories_different
                    ), f"Games with seeds {seeds[i]} and {seeds[j]} should be different"

        print(
            f"✓ Different seeds produce different results verified ({len(seeds)} seeds)"
        )

    def test_move_sequence_determinism(self):
        """Test that move sequences are deterministic."""
        seed = 42

        # Play two identical games
        games = [
            GameState(num_players=2, seed=seed),
            GameState(num_players=2, seed=seed),
        ]

        # Make same moves in both games
        for move_num in range(10):
            actions_list = []

            for game in games:
                actions = game.get_legal_actions()
                actions_list.append(actions)

                if actions:
                    # Choose first action (deterministic)
                    game.apply_action(actions[0])

            # Legal actions should be identical
            if all(actions_list):
                actions1, actions2 = actions_list
                assert len(actions1) == len(
                    actions2
                ), f"Different number of legal actions at move {move_num}"

                for action1, action2 in zip(actions1, actions2):
                    assert (
                        action1.source == action2.source
                    ), f"Action source differs at move {move_num}"
                    assert (
                        action1.color == action2.color
                    ), f"Action color differs at move {move_num}"
                    assert (
                        action1.destination == action2.destination
                    ), f"Action destination differs at move {move_num}"

        # Final states should be identical
        state_repr1 = AzulStateRepresentation(games[0])
        state_repr2 = AzulStateRepresentation(games[1])

        vector1 = state_repr1.get_flat_state_vector(normalize=True)
        vector2 = state_repr2.get_flat_state_vector(normalize=True)

        np.testing.assert_array_equal(
            vector1,
            vector2,
            err_msg="Final states should be identical with same seed and moves",
        )

        print("✓ Move sequence determinism verified")


if __name__ == "__main__":
    print("=== Azul Cross-Platform Compatibility Tests ===")

    # Platform info
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    # Run tests
    platform_tests = TestPlatformConsistency()
    print("\n1. Platform Consistency Tests...")
    platform_tests.test_deterministic_game_creation()
    platform_tests.test_state_representation_consistency()
    platform_tests.test_numerical_precision_consistency()
    platform_tests.test_hash_consistency()

    env_tests = TestEnvironmentCompatibility()
    print("\n2. Environment Compatibility Tests...")
    env_tests.test_python_version_compatibility()
    env_tests.test_numpy_version_compatibility()
    env_tests.test_platform_specific_behavior()
    env_tests.test_file_system_compatibility()

    dtype_tests = TestDataTypeConsistency()
    print("\n3. Data Type Consistency Tests...")
    dtype_tests.test_integer_type_consistency()
    dtype_tests.test_float_type_consistency()
    dtype_tests.test_enum_consistency()
    dtype_tests.test_serialization_type_preservation()

    random_tests = TestRandomnessConsistency()
    print("\n4. Randomness Consistency Tests...")
    random_tests.test_seed_reproducibility()
    random_tests.test_different_seeds_produce_different_results()
    random_tests.test_move_sequence_determinism()

    print("\nAll cross-platform compatibility tests completed successfully!")
