#!/usr/bin/env python3
"""
Comprehensive tests for Azul action encoding and decoding.

This test suite verifies that our optimized action encoding works correctly
for all possible action combinations and edge cases.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

import random

from game.game_state import Action, TileColor

# Set random seed for reproducible tests
random.seed(42)

# Color to index mapping for efficient conversion (FIRST_PLAYER excluded from encoding)
_COLOR_TO_IDX = {
    TileColor.BLUE: 0,
    TileColor.YELLOW: 1,
    TileColor.RED: 2,
    TileColor.BLACK: 3,
    TileColor.WHITE: 4,
    # TileColor.FIRST_PLAYER: not included in encoding (never explicitly chosen)
}

# Index to color mapping for reverse conversion
_IDX_TO_COLOR = {v: k for k, v in _COLOR_TO_IDX.items()}


def _int_to_azul_action(action_int: int) -> Action:
    """Convert integer action to Azul Action using optimized encoding."""
    # Input validation
    if not (0 <= action_int < 180):
        raise ValueError(f"Action integer {action_int} is out of valid range [0, 179]")
    # Reverse the optimized formula: source_idx * 30 + color_idx * 6 + dest_idx
    source_idx = action_int // 30
    remainder = action_int % 30
    color_idx = remainder // 6
    dest_idx = remainder % 6

    source = source_idx - 1  # 0 becomes -1, 1-5 becomes 0-4
    color = _IDX_TO_COLOR[color_idx]
    destination = dest_idx - 1  # 0 becomes -1, 1-5 becomes 0-4

    return Action(source, color, destination)


def _azul_action_to_int(action: Action) -> int:
    """Convert Azul Action to integer using optimized encoding."""
    source_idx = action.source + 1  # -1 becomes 0, 0-4 becomes 1-5
    dest_idx = action.destination + 1  # -1 becomes 0, 0-4 becomes 1-5
    color_idx = _COLOR_TO_IDX[action.color]
    # Optimized encoding: source_idx * 30 + color_idx * 6 + dest_idx
    return source_idx * 30 + color_idx * 6 + dest_idx


def test_encoding_formula():
    """Test the basic encoding formula properties."""
    # Test all possible combinations
    collisions = {}
    max_action = 0
    min_action = float("inf")

    total_combinations = 0
    valid_combinations = 0

    for source in range(-1, 5):  # -1 (center) to 4 (factory 4)
        for color_idx in range(5):  # 0-4 (excluding FIRST_PLAYER)
            for dest in range(-1, 5):  # -1 (floor) to 4 (pattern line 4)
                total_combinations += 1

                # Encode
                source_idx = source + 1  # 0-5
                dest_idx = dest + 1  # 0-5
                action_int = source_idx * 30 + color_idx * 6 + dest_idx

                # Track statistics
                max_action = max(max_action, action_int)
                min_action = min(min_action, action_int)
                valid_combinations += 1

                # Check for collisions
                key = (source, color_idx, dest)
                assert (
                    action_int not in collisions
                ), f"COLLISION! Action {action_int} maps to both {collisions.get(action_int)} and {key}"
                collisions[action_int] = key

    # Verify properties
    assert (
        total_combinations == 180
    ), f"Expected 180 combinations, got {total_combinations}"
    assert (
        valid_combinations == 180
    ), f"Expected 180 valid combinations, got {valid_combinations}"
    assert min_action == 0, f"Expected min action 0, got {min_action}"
    assert max_action == 179, f"Expected max action 179, got {max_action}"
    assert max_action < 180, f"Action {max_action} exceeds action space limit"


def test_roundtrip_encoding():
    """Test that encoding and decoding are perfect inverses."""
    test_cases = []

    # Test all possible valid actions
    for source in range(-1, 5):
        for color_idx in range(5):  # Only tile colors, not FIRST_PLAYER
            for dest in range(-1, 5):
                color = [
                    TileColor.BLUE,
                    TileColor.YELLOW,
                    TileColor.RED,
                    TileColor.BLACK,
                    TileColor.WHITE,
                ][color_idx]
                action = Action(source, color, dest)
                test_cases.append(action)

    assert len(test_cases) == 180, f"Expected 180 test cases, got {len(test_cases)}"

    failures = []
    for original_action in test_cases:
        try:
            # Encode the action
            encoded = _azul_action_to_int(original_action)

            # Decode it back
            decoded_action = _int_to_azul_action(encoded)

            # Check if they match
            if (
                decoded_action.source != original_action.source
                or decoded_action.color != original_action.color
                or decoded_action.destination != original_action.destination
            ):
                failures.append(
                    {
                        "original": original_action,
                        "encoded": encoded,
                        "decoded": decoded_action,
                    }
                )
        except Exception as e:
            failures.append({"original": original_action, "error": str(e)})

    assert not failures, f"Found {len(failures)} roundtrip failures: {failures[:3]}"


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test minimum action (center, blue, floor)
    min_action = Action(-1, TileColor.BLUE, -1)
    encoded_min = _azul_action_to_int(min_action)

    assert encoded_min == 0, f"Expected min encoded action 0, got {encoded_min}"
    decoded_min = _int_to_azul_action(encoded_min)
    assert decoded_min.source == min_action.source
    assert decoded_min.color == min_action.color
    assert decoded_min.destination == min_action.destination

    # Test maximum action (factory 4, white, pattern line 4)
    max_action = Action(4, TileColor.WHITE, 4)
    encoded_max = _azul_action_to_int(max_action)

    assert encoded_max == 179, f"Expected max encoded action 179, got {encoded_max}"
    decoded_max = _int_to_azul_action(encoded_max)
    assert decoded_max.source == max_action.source
    assert decoded_max.color == max_action.color
    assert decoded_max.destination == max_action.destination

    # Test all five colors
    colors = [
        TileColor.BLUE,
        TileColor.YELLOW,
        TileColor.RED,
        TileColor.BLACK,
        TileColor.WHITE,
    ]
    for i, color in enumerate(colors):
        test_action = Action(0, color, 0)  # factory 0, color, pattern line 0
        encoded = _azul_action_to_int(test_action)

        decoded = _int_to_azul_action(encoded)
        assert decoded.source == test_action.source
        assert decoded.color == test_action.color
        assert decoded.destination == test_action.destination


def test_invalid_actions():
    """Test handling of invalid action integers."""
    # Test out-of-range actions
    invalid_actions = [-1, 180, 181, 200, 999]

    for invalid_action in invalid_actions:
        try:
            decoded = _int_to_azul_action(invalid_action)
            # If decoding succeeds, check if the result makes sense
            assert hasattr(
                decoded, "source"
            ), f"Invalid action {invalid_action} produced invalid result"
            assert hasattr(
                decoded, "color"
            ), f"Invalid action {invalid_action} produced invalid result"
            assert hasattr(
                decoded, "destination"
            ), f"Invalid action {invalid_action} produced invalid result"

            # The decoded action should be mathematically valid even if not game-legal
            # For negative actions, we expect them to be handled by the exception
            if invalid_action < 0:
                assert (
                    False
                ), f"Decoding negative action {invalid_action} should have raised an exception"
            # For actions >= 180, we expect them to be handled by the exception
            elif invalid_action >= 180:
                assert (
                    False
                ), f"Decoding out-of-range action {invalid_action} should have raised an exception"
            else:
                assert (
                    -1 <= decoded.source <= 4
                ), f"Invalid source {decoded.source} from action {invalid_action}"
                assert decoded.color in [
                    TileColor.BLUE,
                    TileColor.YELLOW,
                    TileColor.RED,
                    TileColor.BLACK,
                    TileColor.WHITE,
                ]
                assert (
                    -1 <= decoded.destination <= 4
                ), f"Invalid destination {decoded.destination} from action {invalid_action}"

        except (IndexError, KeyError, ValueError) as e:
            # It's acceptable for invalid actions to raise exceptions
            # For negative actions, we expect them to raise an exception
            if invalid_action < 0:
                assert isinstance(
                    e, (IndexError, ValueError)
                ), f"Expected IndexError or ValueError for negative action {invalid_action}, got {type(e)}"
            # For actions >= 180, we expect them to raise an exception
            elif invalid_action >= 180:
                assert isinstance(
                    e, (IndexError, ValueError)
                ), f"Expected IndexError or ValueError for out-of-range action {invalid_action}, got {type(e)}"


def test_no_first_player_conflicts():
    """Test that FIRST_PLAYER token doesn't conflict with valid actions."""
    # FIRST_PLAYER should not be in the color mapping
    assert (
        TileColor.FIRST_PLAYER not in _COLOR_TO_IDX
    ), "FIRST_PLAYER should not be in color mapping"

    # Verify only 5 colors are mapped (excluding FIRST_PLAYER)
    assert (
        len(_COLOR_TO_IDX) == 5
    ), f"Expected 5 colors in mapping, got {len(_COLOR_TO_IDX)}"

    # Verify the reverse mapping has the same size
    assert (
        len(_IDX_TO_COLOR) == 5
    ), f"Expected 5 colors in reverse mapping, got {len(_IDX_TO_COLOR)}"

    # Verify all tile colors (except FIRST_PLAYER) are mapped
    expected_colors = {
        TileColor.BLUE,
        TileColor.YELLOW,
        TileColor.RED,
        TileColor.BLACK,
        TileColor.WHITE,
    }
    mapped_colors = set(_COLOR_TO_IDX.keys())
    assert (
        mapped_colors == expected_colors
    ), f"Color mapping mismatch. Expected {expected_colors}, got {mapped_colors}"
