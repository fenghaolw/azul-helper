#!/usr/bin/env python3
"""
Demonstration: Tile Counting and Accessibility

This script demonstrates that factory tiles (and all other tiles) are easily
accessible and countable in the Azul game state representation.
"""
import os
import sys

# Add parent directory to path so we can import from game
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game import AzulStateRepresentation, GameState


def demonstrate_tile_counting():
    """Demonstrate various ways to count tiles in the game state."""
    print("=== Azul Tile Counting Demonstration ===\n")

    # Create a game
    game = GameState(num_players=2, seed=42)
    state_repr = AzulStateRepresentation(game)

    print("1. INITIAL GAME STATE")
    print("=" * 40)

    # Method 1: Use the built-in helper method
    distribution = state_repr.get_tile_distribution()
    print("Complete tile distribution:")
    for location, count in distribution.items():
        print(f"  {location}: {count}")
    print()

    # Method 2: Manual counting of factory tiles
    print("2. MANUAL FACTORY TILE COUNTING")
    print("=" * 40)

    total_factory_tiles = 0
    for factory_idx in range(state_repr.num_factories):
        factory_array = state_repr.factories[factory_idx]
        tiles_in_factory = int(factory_array[:, 0].sum())  # Count has_tile indicators
        print(f"  Factory {factory_idx}: {tiles_in_factory} tiles")
        total_factory_tiles += tiles_in_factory

    print(f"  Total factory tiles: {total_factory_tiles}")
    print()

    print("CONCLUSION: Factory tiles are easily accessible!")


if __name__ == "__main__":
    demonstrate_tile_counting()
