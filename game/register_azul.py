"""
Register Azul game with OpenSpiel.

This module registers our custom Azul game implementation with OpenSpiel
so it can be loaded using pyspiel.load_game("azul").
"""

import pyspiel

from game.azul_openspiel import AzulGame


def register_azul_game():
    """Register the Azul game with OpenSpiel."""
    # Create a game type for Azul
    game_type = pyspiel.GameType(
        short_name="azul",
        long_name="Azul",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.GENERAL_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=4,
        min_num_players=2,
        provides_information_state_string=True,
        provides_information_state_tensor=True,
        provides_observation_string=True,
        provides_observation_tensor=True,
        provides_factored_observation_string=False,
        parameter_specification={},  # Simplified - no parameters for now
        default_loadable=True,
    )

    # Register the game with a factory function
    def azul_game_factory(params):
        """Factory function to create AzulGame instances."""
        # Use default parameters since we removed parameter specification
        default_params = {"players": 2}
        if params:
            default_params.update(params)
        return AzulGame(default_params)

    pyspiel.register_game(game_type, azul_game_factory)
    print("Azul game registered with OpenSpiel!")


def is_azul_registered():
    """Check if Azul game is already registered."""
    return "azul" in pyspiel.registered_names()


# Auto-register when module is imported
if not is_azul_registered():
    register_azul_game()
