"""
Tests for serialization and persistence of Azul game states.

This module tests the ability to save and load game states and state representations
in various formats for persistence and cross-platform compatibility.
"""

import json
import os
import pickle
import tempfile
import time

import numpy as np
import pytest

from azul_rl.game import AzulStateRepresentation, GameState


class TestStateSerialization:
    """Test serialization of game states and state representations."""

    def test_game_state_pickle_serialization(self):
        """Test pickle serialization of GameState objects."""
        # Create a game and make some moves
        original_game = GameState(num_players=3, seed=42)

        # Make several moves to create interesting state
        for _ in range(5):
            actions = original_game.get_legal_actions()
            if actions:
                original_game.apply_action(actions[0])

        # Serialize with pickle
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(original_game, f)
            temp_file = f.name

        try:
            # Deserialize
            with open(temp_file, "rb") as f:
                loaded_game = pickle.load(f)

            # Verify state matches
            assert loaded_game.num_players == original_game.num_players
            assert loaded_game.current_player == original_game.current_player
            assert loaded_game.round_number == original_game.round_number
            assert loaded_game.game_over == original_game.game_over

            # Verify player states
            for i, (orig_player, loaded_player) in enumerate(
                zip(original_game.players, loaded_game.players)
            ):
                assert orig_player.score == loaded_player.score
                assert len(orig_player.floor_line) == len(loaded_player.floor_line)

                # Check pattern lines
                for j, (orig_line, loaded_line) in enumerate(
                    zip(orig_player.pattern_lines, loaded_player.pattern_lines)
                ):
                    assert len(orig_line.tiles) == len(loaded_line.tiles)
                    assert orig_line.color == loaded_line.color

                # Check wall
                for row in range(5):
                    for col in range(5):
                        assert (
                            orig_player.wall.filled[row][col]
                            == loaded_player.wall.filled[row][col]
                        )

            # Verify factories
            for orig_factory, loaded_factory in zip(
                original_game.factory_area.factories, loaded_game.factory_area.factories
            ):
                assert len(orig_factory.tiles) == len(loaded_factory.tiles)
                for orig_tile, loaded_tile in zip(
                    orig_factory.tiles, loaded_factory.tiles
                ):
                    assert orig_tile.color == loaded_tile.color

            # Verify center
            assert len(original_game.factory_area.center.tiles) == len(
                loaded_game.factory_area.center.tiles
            )
            assert (
                original_game.factory_area.center.has_first_player_marker
                == loaded_game.factory_area.center.has_first_player_marker
            )

            print("✓ Pickle serialization test passed")

        finally:
            os.unlink(temp_file)

    def test_state_representation_numpy_serialization(self):
        """Test NumPy array serialization of state representations."""
        # Create game and state representation
        game = GameState(num_players=2, seed=42)

        # Make some moves
        for _ in range(3):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        original_state_repr = AzulStateRepresentation(game)

        # Serialize state components to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all state components
            state_dict = original_state_repr.get_state_dict()

            saved_files = {}
            for key, array in state_dict.items():
                file_path = os.path.join(temp_dir, f"{key}.npy")
                np.save(file_path, array)
                saved_files[key] = file_path

            # Save metadata
            metadata = {
                "num_players": original_state_repr.num_players,
                "num_factories": original_state_repr.num_factories,
                "current_player": original_state_repr.current_player,
                "round_number": original_state_repr.round_number,
                "game_over": original_state_repr.game_over,
                "winner": original_state_repr.winner,
            }

            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Load back and verify
            loaded_state_dict = {}
            for key, file_path in saved_files.items():
                loaded_state_dict[key] = np.load(file_path)

            with open(metadata_path, "r") as f:
                loaded_metadata = json.load(f)

            # Verify all arrays match
            for key in state_dict:
                np.testing.assert_array_equal(
                    state_dict[key],
                    loaded_state_dict[key],
                    err_msg=f"Mismatch in {key} array",
                )

            # Verify metadata
            for key, value in metadata.items():
                assert loaded_metadata[key] == value, f"Metadata mismatch for {key}"

            print("✓ NumPy serialization test passed")

    def test_flat_state_vector_serialization(self):
        """Test serialization of flattened state vectors."""
        game = GameState(num_players=4, seed=123)

        # Make moves to create varied state
        for _ in range(8):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Get both normalized and unnormalized vectors
        normalized_vector = state_repr.get_flat_state_vector(normalize=True)
        unnormalized_vector = state_repr.get_flat_state_vector(normalize=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save vectors
            norm_path = os.path.join(temp_dir, "normalized.npy")
            unnorm_path = os.path.join(temp_dir, "unnormalized.npy")

            np.save(norm_path, normalized_vector)
            np.save(unnorm_path, unnormalized_vector)

            # Load and verify
            loaded_norm = np.load(norm_path)
            loaded_unnorm = np.load(unnorm_path)

            np.testing.assert_array_equal(normalized_vector, loaded_norm)
            np.testing.assert_array_equal(unnormalized_vector, loaded_unnorm)

            # Verify properties
            assert len(loaded_norm) == state_repr.flat_state_size
            assert len(loaded_unnorm) == state_repr.flat_state_size
            assert loaded_norm.dtype == np.float32

            print("✓ Flat vector serialization test passed")

    def test_json_serialization_compatibility(self):
        """Test JSON serialization for cross-platform compatibility."""
        game = GameState(num_players=2, seed=42)

        # Make some moves
        for _ in range(4):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Convert to JSON-serializable format
        json_data = {
            "metadata": {
                "num_players": state_repr.num_players,
                "num_factories": state_repr.num_factories,
                "current_player": int(state_repr.current_player),
                "round_number": int(state_repr.round_number),
                "game_over": int(state_repr.game_over),
                "winner": int(state_repr.winner),
                "flat_state_size": state_repr.flat_state_size,
            },
            "state_arrays": {},
            "flat_vector": state_repr.get_flat_state_vector(normalize=True).tolist(),
        }

        # Convert arrays to lists for JSON
        state_dict = state_repr.get_state_dict()
        for key, array in state_dict.items():
            json_data["state_arrays"][key] = array.tolist()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f, indent=2)
            temp_file = f.name

        try:
            # Load and verify
            with open(temp_file, "r") as f:
                loaded_data = json.load(f)

            # Verify metadata
            for key, value in json_data["metadata"].items():
                assert loaded_data["metadata"][key] == value

            # Verify arrays can be reconstructed
            for key, original_list in json_data["state_arrays"].items():
                loaded_list = loaded_data["state_arrays"][key]
                original_array = np.array(original_list)
                loaded_array = np.array(loaded_list)
                np.testing.assert_array_equal(original_array, loaded_array)

            # Verify flat vector
            original_flat = np.array(json_data["flat_vector"])
            loaded_flat = np.array(loaded_data["flat_vector"])
            np.testing.assert_array_almost_equal(original_flat, loaded_flat, decimal=6)

            print("✓ JSON serialization test passed")

        finally:
            os.unlink(temp_file)

    def test_cross_session_consistency(self):
        """Test that serialized states remain consistent across sessions."""
        # Create deterministic game
        game1 = GameState(num_players=3, seed=999)

        # Make specific sequence of moves
        move_sequence = []
        for _ in range(6):
            actions = game1.get_legal_actions()
            if actions:
                chosen_action = actions[0]  # Always choose first action for determinism
                game1.apply_action(chosen_action)
                move_sequence.append(
                    {
                        "source": chosen_action.source,
                        "color": chosen_action.color.value,
                        "destination": chosen_action.destination,
                    }
                )

        state_repr1 = AzulStateRepresentation(game1)
        flat_vector1 = state_repr1.get_flat_state_vector(normalize=True)

        # Serialize the game state
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(game1, f)
            temp_file = f.name

        try:
            # In a "new session", load the game
            with open(temp_file, "rb") as f:
                game2 = pickle.load(f)

            # Create state representation from loaded game
            state_repr2 = AzulStateRepresentation(game2)
            flat_vector2 = state_repr2.get_flat_state_vector(normalize=True)

            # Verify complete consistency
            np.testing.assert_array_equal(flat_vector1, flat_vector2)

            # Verify we can continue the game consistently
            actions1 = game1.get_legal_actions()
            actions2 = game2.get_legal_actions()

            assert len(actions1) == len(actions2)
            for a1, a2 in zip(actions1, actions2):
                assert a1.source == a2.source
                assert a1.color == a2.color
                assert a1.destination == a2.destination

            print("✓ Cross-session consistency test passed")

        finally:
            os.unlink(temp_file)

    def test_partial_state_reconstruction(self):
        """Test reconstructing partial game information from state representation."""
        game = GameState(num_players=2, seed=42)

        # Play until we have interesting state
        for _ in range(10):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)

        # Extract key information that should be reconstructible
        original_info = {
            "player_scores": [p.score for p in game.players],
            "current_player": game.current_player,
            "round_number": game.round_number,
            "game_over": game.game_over,
            "factory_tile_counts": [len(f.tiles) for f in game.factory_area.factories],
            "center_tile_count": len(game.factory_area.center.tiles),
            "center_has_first_player": game.factory_area.center.has_first_player_marker,
        }

        # Serialize state representation
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(state_repr, f)
            temp_file = f.name

        try:
            # Load state representation
            with open(temp_file, "rb") as f:  # type: ignore
                loaded_state_repr = pickle.load(f)

            # Reconstruct information
            reconstructed_info = {
                "player_scores": loaded_state_repr.player_scores[
                    : loaded_state_repr.num_players
                ].tolist(),
                "current_player": loaded_state_repr.current_player,
                "round_number": loaded_state_repr.round_number,
                "game_over": bool(loaded_state_repr.game_over),
                "factory_tile_counts": [
                    int(np.sum(loaded_state_repr.factories[i, :, 0]))
                    for i in range(loaded_state_repr.num_factories)
                ],
                "center_tile_count": int(np.sum(loaded_state_repr.center_tiles[:, 0])),
                "center_has_first_player": bool(
                    loaded_state_repr.center_first_player_marker
                ),
            }

            # Verify reconstruction accuracy
            for key in original_info:
                assert (
                    original_info[key] == reconstructed_info[key]
                ), f"Mismatch in {key}: {original_info[key]} != {reconstructed_info[key]}"

            print("✓ Partial state reconstruction test passed")

        finally:
            os.unlink(temp_file)


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_empty_game_serialization(self):
        """Test serialization of newly created games."""
        game = GameState(num_players=2, seed=42)
        state_repr = AzulStateRepresentation(game)

        # Serialize empty game
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump((game, state_repr), f)
            temp_file = f.name

        try:
            with open(temp_file, "rb") as f:
                loaded_game, loaded_state_repr = pickle.load(f)

            # Verify empty state properties
            assert loaded_game.round_number == 1
            assert loaded_game.current_player == 0
            assert not loaded_game.game_over
            assert all(p.score == 0 for p in loaded_game.players)

            # Verify state representation
            assert loaded_state_repr.round_number == 1
            assert loaded_state_repr.current_player == 0
            assert loaded_state_repr.game_over == 0

            print("✓ Empty game serialization test passed")

        finally:
            os.unlink(temp_file)

    def test_completed_game_serialization(self):
        """Test serialization of completed games."""
        game = GameState(num_players=2, seed=42)

        # Play until game ends
        max_moves = 200  # Safety limit
        moves = 0
        while not game.game_over and moves < max_moves:
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])
            moves += 1

        if game.game_over:
            state_repr = AzulStateRepresentation(game)

            with tempfile.NamedTemporaryFile(delete=False) as f:
                pickle.dump((game, state_repr), f)
                temp_file = f.name

            try:
                with open(temp_file, "rb") as f:
                    loaded_game, loaded_state_repr = pickle.load(f)

                # Verify completed game properties
                assert loaded_game.game_over
                assert loaded_state_repr.game_over == 1
                assert loaded_game.winner is not None
                assert loaded_state_repr.winner >= 0

                print("✓ Completed game serialization test passed")

            finally:
                os.unlink(temp_file)
        else:
            print(
                "⚠ Game didn't complete within move limit, skipping completed game test"
            )

    def test_large_state_serialization(self):
        """Test serialization of games with maximum players."""
        game = GameState(num_players=4, seed=42)  # Maximum players

        # Make many moves to create complex state
        for _ in range(15):
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(actions[0])

        state_repr = AzulStateRepresentation(game)
        flat_vector = state_repr.get_flat_state_vector(normalize=True)

        # Test that large states can be serialized efficiently
        with tempfile.NamedTemporaryFile(delete=False) as f:
            start_time = time.time()
            pickle.dump(state_repr, f)
            serialize_time = time.time() - start_time
            temp_file = f.name

        try:
            # Test loading time
            start_time = time.time()
            with open(temp_file, "rb") as f:
                loaded_state_repr = pickle.load(f)
            load_time = time.time() - start_time

            # Verify correctness
            loaded_flat_vector = loaded_state_repr.get_flat_state_vector(normalize=True)
            np.testing.assert_array_equal(flat_vector, loaded_flat_vector)

            # Performance should be reasonable (less than 1 second each)
            assert (
                serialize_time < 1.0
            ), f"Serialization too slow: {serialize_time:.3f}s"
            assert load_time < 1.0, f"Loading too slow: {load_time:.3f}s"

            print(
                f"✓ Large state serialization test passed (serialize: {serialize_time:.3f}s, load: {load_time:.3f}s)"
            )

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    print("=== Azul Serialization Tests Demo ===")

    # Run a quick demo
    test_suite = TestStateSerialization()

    print("\n1. Testing game state pickle serialization...")
    test_suite.test_game_state_pickle_serialization()

    print("\n2. Testing state representation numpy serialization...")
    test_suite.test_state_representation_numpy_serialization()

    print("\n3. Testing JSON compatibility...")
    test_suite.test_json_serialization_compatibility()

    print("\n4. Testing cross-session consistency...")
    test_suite.test_cross_session_consistency()

    print("\nAll serialization tests completed successfully!")
