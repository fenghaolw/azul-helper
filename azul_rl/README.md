# Azul Reinforcement Learning

A Python implementation of the Azul board game designed for reinforcement learning research and agent training.

## Overview

This project provides a complete implementation of the Azul board game with:
- Full game mechanics and rules
- State representation suitable for ML models
- Action space definition
- Game state management
- Foundation for RL agent development

## Project Structure

```
azul_rl/
├── game/                   # Core game implementation
│   ├── __init__.py
│   ├── tile.py            # Tile and color definitions
│   ├── player_board.py    # Player board, pattern lines, wall
│   ├── factory.py         # Factory displays and center area
│   └── game_state.py      # Main game state and logic
├── agents/                # RL agents (future development)
├── training/              # Training scripts and utilities
├── utils/                 # Helper utilities
├── tests/                 # Unit tests
├── example_game.py        # Example usage
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Game Components

### Core Classes

- **`Tile`**: Represents individual tiles with colors (blue, yellow, red, black, white)
- **`PlayerBoard`**: Manages pattern lines, wall, floor line, and scoring
- **`Factory`**: Represents factory displays that hold 4 tiles
- **`CenterArea`**: Central area where leftover tiles accumulate
- **`GameState`**: Main game controller managing all game logic
- **`Action`**: Represents player actions (source, color, destination)

### Key Features

- **Complete Azul Rules**: Implements all standard Azul game mechanics
- **State Vector**: Numerical representation of game state for ML models
- **Legal Action Generation**: Automatic generation of valid moves
- **Scoring System**: Full implementation of Azul scoring rules
- **Game State Copying**: Deep copy support for tree search algorithms

## Quick Start

### Installation

```bash
# Clone or copy the azul_rl directory
cd azul_rl

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from azul_rl import create_game, Action, TileColor

# Create a new game
game = create_game(num_players=2, seed=42)

# Get legal actions for current player
legal_actions = game.get_legal_actions()

# Create and apply an action
action = Action(source=0, color=TileColor.BLUE, destination=0)
success = game.apply_action(action)

# Get game state as numerical vector for ML
state_vector = game.get_state_vector()

# Check if game is over
if game.game_over:
    print(f"Winner: Player {game.winner}")
    print(f"Final scores: {game.get_scores()}")
```

### Running the Example

```bash
cd azul_rl
python example_game.py
```

This will run a complete game between two random agents and demonstrate the basic functionality.

## Game Rules Summary

Azul is a tile-laying game where players:

1. **Take tiles** from factory displays or the center area
2. **Place tiles** on pattern lines or floor line
3. **Score points** by moving completed pattern lines to the wall
4. **End game** when any player completes a horizontal row

### Scoring
- Points for placing tiles on wall (based on adjacent tiles)
- Penalties for tiles on floor line
- End-game bonuses for completed rows, columns, and colors

## State Representation

The `get_state_vector()` method returns a normalized numerical representation including:

- Game metadata (current player, round, game over status)
- Player scores
- Current player's board state (pattern lines, wall, floor line)
- Factory states (tile counts by color)
- Center area state
- First player marker location

## Action Space

Actions are represented as `Action(source, color, destination)`:
- **Source**: Factory index (0-4 for 2 players) or -1 for center
- **Color**: One of the 5 tile colors
- **Destination**: Pattern line index (0-4) or -1 for floor line

## Future Development

This foundation supports development of:
- Deep Q-Networks (DQN)
- Policy Gradient methods
- Monte Carlo Tree Search (MCTS)
- Multi-agent reinforcement learning
- Self-play training

## Testing

```bash
cd azul_rl
python -m pytest tests/
```

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add appropriate type hints
3. Include docstrings for public methods
4. Add unit tests for new functionality
5. Update this README if needed

## License

This project is intended for educational and research purposes. 