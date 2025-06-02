"""
Training module for Azul RL.

🚀 **Primary Training Method**: OpenSpiel AlphaZero
Use `python training/openspiel_alphazero_training.py` for all training needs.

The custom training infrastructure has been removed in favor of OpenSpiel's
superior AlphaZero implementation.
"""

# Only keep utilities that are still useful
from training.eta_tracker import ETATracker

# Note: OpenSpiel AlphaZero training is available as a script:
# python training/openspiel_alphazero_training.py

__all__ = [
    "ETATracker",
]
