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
  std::cout << "=== AZUL AGENT EVALUATION & TOURNAMENT DEMO ===" << '\n';

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

    // Configure evaluation
    azul::EvaluationConfig eval_config;
    eval_config.num_games = 50;
    eval_config.verbose = true;
    eval_config.swap_player_positions = true;
    eval_config.timeout_per_move = 10.0;

    azul::AgentEvaluator evaluator(eval_config);

    // // Test 1: Minimax vs MCTS
    // std::cout << "\n--- Evaluation 1: Minimax vs MCTS ---" << '\n';
    // auto result1 = evaluator.evaluate_agent(*minimax_agent, *mcts_agent);
    // std::cout << result1.summary() << '\n';

    // // Test 2: Minimax vs Random (should win easily)
    // std::cout << "\n--- Evaluation 2: Minimax vs Random ---" << '\n';
    // auto result2 = evaluator.evaluate_agent(*minimax_agent, *random_agent);
    // std::cout << result2.summary() << '\n';

    // // Test 3: MCTS vs Random (should win easily)
    // std::cout << "\n--- Evaluation 3: MCTS vs Random ---" << '\n';
    // auto result3 = evaluator.evaluate_agent(*mcts_agent, *random_agent);
    // std::cout << result3.summary() << '\n';

    // Test 4: AlphaZero MCTS vs Random Rollout MCTS
    // std::cout << "\n--- Evaluation 4: AlphaZero MCTS vs Random Rollout MCTS
    // ---"
    //           << '\n';
    // std::string checkpoint_path =
    //     "models/libtorch_alphazero_azul/checkpoint-0.pt";
    // auto alphazero_mcts_agent = azul::create_alphazero_mcts_evaluation_agent(
    //     checkpoint_path, 400, 1.4, 42, "AlphaZero_MCTS_400");
    // auto result4 = evaluator.evaluate_agent(*alphazero_mcts_agent,
    // *mcts_agent); std::cout << result4.summary() << '\n';

    // =================================================================
    // PART 2: TOURNAMENT EVALUATION
    // =================================================================

    // Configure tournament
    azul::EvaluationConfig tournament_config;
    tournament_config.num_games =
        30;  // 30 games per matchup for reliable results
    tournament_config.verbose = false;  // Less verbose for tournament
    tournament_config.swap_player_positions = true;

    azul::Tournament tournament(tournament_config);

    // Add the 3 core agents to tournament
    tournament.add_agent(
        azul::create_minimax_evaluation_agent(4, "Minimax_D4"));
    tournament.add_agent(
        azul::create_mcts_evaluation_agent(1000, 1.4, 42, "MCTS_1000"));
    // Add AlphaZero MCTS agent to tournament
    tournament.add_agent(azul::create_alphazero_mcts_evaluation_agent(
        "models/libtorch_alphazero_azul/checkpoint--1", 400, 1.4, 42,
        "AlphaZero_MCTS_400"));

    std::cout << "Starting tournament with " << tournament.get_num_agents()
              << " agents..." << '\n';

    // Run tournament
    auto tournament_result = tournament.run_tournament();

    // Display results
    std::cout << "\nTOURNAMENT RESULTS:" << '\n';
    std::cout << tournament_result.summary() << '\n';

  } catch (const std::exception& e) {
    std::cerr << "❌ Error during evaluation: " << e.what() << '\n';
    return 1;
  }

  return 0;
}