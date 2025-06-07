#include <iostream>

#include "agent_evaluator.h"
#include "evaluation_config.h"

namespace {
void force_azul_registration() {
  // Reference symbols from azul namespace to force linking
  (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}
}  // namespace

auto main() -> int {
  std::cout << "=== AZUL ALPHAZERO MCTS vs MINIMAX EVALUATION DEMO ===" << '\n';

  try {
    // Force registration
    force_azul_registration();

    // Verify Azul game loads
    auto game = open_spiel::LoadGame("azul");
    if (!game) {
      std::cerr << "❌ Failed to load Azul game" << '\n';
      return 1;
    }
    std::cout << "✅ Azul game loaded successfully" << '\n';

    // Create a new state
    auto state = game->NewInitialState();
    std::cout << "\nInitial game state:\n" << state->ToString() << '\n';

    const auto& azul_state =
        static_cast<const open_spiel::azul::AzulState&>(*state);

    // Create the agents
    auto alphazero_agent = azul::create_alphazero_mcts_evaluation_agent(
        "models/libtorch_alphazero_azul/checkpoint--1", 400, 1.4, 42,
        "AlphaZero_MCTS_400");
    auto minimax_agent = azul::create_minimax_evaluation_agent(5, "Minimax_D4");

    // Play the game
    while (!state->IsTerminal()) {
      // Get current player
      int current_player = state->CurrentPlayer();

      // Select agent based on current player
      auto& agent = (current_player == 0) ? *alphazero_agent : *minimax_agent;

      // Get action from agent
      auto action = agent.get_action(*state, current_player);

      // Print the action being taken
      std::cout << "\nPlayer " << current_player << " ("
                << (current_player == 0 ? "AlphaZero MCTS" : "Minimax")
                << ") takes action: "
                << state->ActionToString(current_player, action) << '\n';

      // Apply the action
      state->ApplyAction(action);

      // Print the new state
      std::cout << "\nGame state after move:\n" << state->ToString() << '\n';
    }

    // Print final results
    std::cout << "\n=== GAME OVER ===\n";
    std::cout << "Final scores:\n";
    for (int player = 0; player < game->NumPlayers(); ++player) {
      std::cout << "Player " << player << " ("
                << (player == 0 ? "AlphaZero MCTS" : "Minimax")
                << "): " << state->PlayerReturn(player) << '\n';
    }

  } catch (const std::exception& e) {
    std::cerr << "❌ Error during evaluation: " << e.what() << '\n';
    return 1;
  }

  return 0;
}