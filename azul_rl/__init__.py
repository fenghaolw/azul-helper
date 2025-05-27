"""
Azul Reinforcement Learning Package

A Python implementation of the Azul board game for reinforcement learning research.
"""

from .game.factory import CenterArea, Factory, FactoryArea
from .game.game_state import Action, GameState, create_game
from .game.player_board import PlayerBoard
from .game.tile import Tile, TileColor

__version__ = "0.1.0"
__all__ = [
    "Action",
    "CenterArea",
    "Factory",
    "FactoryArea",
    "GameState",
    "PlayerBoard",
    "Tile",
    "TileColor",
    "create_game",
]
