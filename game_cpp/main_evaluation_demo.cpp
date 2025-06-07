#include <iostream>
#include <memory>
#include <string>

#include "agent_evaluator.h"
#include "cxxopts.hpp"
#include "evaluation_config.h"

namespace {
void force_azul_registration() {
  // Reference symbols from azul namespace to force linking
  (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}

void detailed_game_evaluation(std::shared_ptr<const open_spiel::Game> game,
                              std::unique_ptr<azul::EvaluationAgent>& agent1,
                              std::unique_ptr<azul::EvaluationAgent>& agent2,
                              const std::string& agent1_name,
                              const std::string& agent2_name) {
  auto state = game->NewInitialState();
  std::cout << "\nInitial game state:\n" << state->ToString() << '\n';

  while (!state->IsTerminal()) {
    int current_player = state->CurrentPlayer();
    auto& agent = (current_player == 0) ? *agent1 : *agent2;
    std::string agent_name = (current_player == 0) ? agent1_name : agent2_name;

    auto action = agent.get_action(*state, current_player);

    std::cout << "\nPlayer " << current_player << " (" << agent_name
              << ") takes action: "
              << state->ActionToString(current_player, action) << '\n';

    state->ApplyAction(action);
    std::cout << "\nGame state after move:\n" << state->ToString() << '\n';
  }

  std::cout << "\n=== GAME OVER ===\n";
  std::cout << "Final scores:\n";
  for (int player = 0; player < game->NumPlayers(); ++player) {
    std::string agent_name = (player == 0) ? agent1_name : agent2_name;
    std::cout << "Player " << player << " (" << agent_name
              << "): " << state->PlayerReturn(player) << '\n';
  }
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  std::cout << "=== AZUL AGENT EVALUATION DEMO ===" << '\n';

  try {
    // Parse command line options
    cxxopts::Options options("azul_evaluation", "Azul Agent Evaluation Demo");
    options.add_options()(
        "m,mode", "Evaluation mode: 'detailed' or 'tournament'",
        cxxopts::value<std::string>()->default_value("tournament"))(
        "g,games", "Number of games per matchup in tournament mode",
        cxxopts::value<int>()->default_value("10"))(
        "v,verbose", "Enable verbose output",
        cxxopts::value<bool>()->default_value("true"))("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }

    std::string mode = result["mode"].as<std::string>();
    if (mode != "detailed" && mode != "tournament") {
      std::cerr << "❌ Invalid mode: " << mode
                << ". Must be 'detailed' or 'tournament'" << '\n';
      return 1;
    }

    force_azul_registration();

    auto game = open_spiel::LoadGame("azul");
    if (!game) {
      std::cerr << "❌ Failed to load Azul game" << '\n';
      return 1;
    }
    std::cout << "✅ Azul game loaded successfully" << '\n';

    // Create agents
    auto mcts_agent =
        azul::create_mcts_evaluation_agent(2000, 1.4, 42, "MCTS_2000");
    auto minimax_agent = azul::create_minimax_evaluation_agent(3, "Minimax_D3");
    auto random_agent = azul::create_random_evaluation_agent(42, "Random");

    if (mode == "detailed") {
      detailed_game_evaluation(game, mcts_agent, minimax_agent,
                               "MCTS (2000 sims)", "Minimax (D3)");
    } else {
      // Run tournament using the existing Tournament class
      azul::EvaluationConfig config;
      config.verbose = result["verbose"].as<bool>();
      config.num_games = result["games"].as<int>();

      azul::Tournament tournament(config);
      tournament.add_agent(std::move(mcts_agent));
      tournament.add_agent(std::move(minimax_agent));
      tournament.add_agent(std::move(random_agent));

      auto tournament_result = tournament.run_tournament();
    }

  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "❌ Error parsing options: " << e.what() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "❌ Error during evaluation: " << e.what() << '\n';
    return 1;
  }

  return 0;
}