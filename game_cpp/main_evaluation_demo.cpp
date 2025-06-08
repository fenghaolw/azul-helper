#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "agent_evaluator.h"
#include "azul.h"
#include "cxxopts.hpp"
#include "evaluation_config.h"
#include "nlohmann/json.hpp"
#include "open_spiel/spiel.h"

namespace {
void force_azul_registration() {
  // Reference symbols from azul namespace to force linking
  (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}

// Helper function to convert game state to JSON format
nlohmann::json state_to_json(const open_spiel::State* state) {
  const auto* azul_state =
      dynamic_cast<const open_spiel::azul::AzulState*>(state);
  if (!azul_state) {
    throw std::runtime_error("Invalid state type");
  }

  nlohmann::json state_json;
  state_json["currentPlayer"] = azul_state->CurrentPlayer();
  state_json["roundNumber"] = azul_state->round_number_;
  state_json["gameEnded"] = azul_state->IsTerminal();
  state_json["firstPlayerTileAvailable"] = azul_state->HasFirstPlayerTile();

  // Convert factories
  state_json["factories"] = nlohmann::json::array();
  for (const auto& factory : azul_state->Factories()) {
    nlohmann::json factory_json;
    for (int color = 0; color < open_spiel::azul::kNumTileColors; ++color) {
      factory_json[open_spiel::azul::TileColorToString(
          static_cast<open_spiel::azul::TileColor>(color))] =
          factory.tiles[color];
    }
    state_json["factories"].push_back(factory_json);
  }

  // Convert center
  state_json["center"] = nlohmann::json::object();
  for (int color = 0; color < open_spiel::azul::kNumTileColors; ++color) {
    state_json["center"][open_spiel::azul::TileColorToString(
        static_cast<open_spiel::azul::TileColor>(color))] =
        azul_state->CenterPile().tiles[color];
  }

  // Convert player boards
  state_json["playerBoards"] = nlohmann::json::array();
  for (int player = 0; player < azul_state->num_players_; ++player) {
    const auto& board = azul_state->GetPlayerBoard(player);
    nlohmann::json board_json;
    board_json["score"] = board.score;

    // Convert pattern lines
    board_json["patternLines"] = nlohmann::json::array();
    for (int line = 0; line < open_spiel::azul::kNumPatternLines; ++line) {
      nlohmann::json line_json;
      line_json["color"] =
          open_spiel::azul::TileColorToString(board.pattern_lines[line].color);
      line_json["count"] = board.pattern_lines[line].count;
      board_json["patternLines"].push_back(line_json);
    }

    // Convert wall
    board_json["wall"] = nlohmann::json::array();
    for (int row = 0; row < open_spiel::azul::kWallSize; ++row) {
      nlohmann::json row_json = nlohmann::json::array();
      for (int col = 0; col < open_spiel::azul::kWallSize; ++col) {
        row_json.push_back(board.wall[row][col] ? 1 : 0);
      }
      board_json["wall"].push_back(row_json);
    }

    // Convert floor
    board_json["floor"] = nlohmann::json::array();
    for (const auto& tile : board.floor_line) {
      board_json["floor"].push_back(open_spiel::azul::TileColorToString(tile));
    }

    state_json["playerBoards"].push_back(board_json);
  }

  return state_json;
}

void detailed_game_evaluation(
    const std::shared_ptr<const open_spiel::Game>& game,
    std::unique_ptr<azul::EvaluationAgent>& agent1,
    std::unique_ptr<azul::EvaluationAgent>& agent2,
    const std::string& agent1_name, const std::string& agent2_name,
    const std::string& output_file = "") {
  auto state = game->NewInitialState();
  std::cout << "\nInitial game state:\n" << state->ToString() << '\n';

  // Create JSON structure for game replay
  nlohmann::json game_replay;
  game_replay["agent1_name"] = agent1_name;
  game_replay["agent2_name"] = agent2_name;
  game_replay["moves"] = nlohmann::json::array();
  game_replay["initial_state"] = state_to_json(state.get());

  while (!state->IsTerminal()) {
    int current_player = state->CurrentPlayer();
    auto& agent = (current_player == 0) ? *agent1 : *agent2;
    std::string agent_name = (current_player == 0) ? agent1_name : agent2_name;

    auto action = agent.get_action(*state, current_player);
    std::string action_str = state->ActionToString(current_player, action);

    std::cout << "\nPlayer " << current_player << " (" << agent_name
              << ") takes action: " << action_str << '\n';

    // Record move in JSON
    nlohmann::json move;
    move["player"] = current_player;
    move["agent"] = agent_name;
    move["action"] = action_str;
    state->ApplyAction(action);
    move["state_after"] = state_to_json(state.get());
    game_replay["moves"].push_back(move);

    std::cout << "\nGame state after move:\n" << state->ToString() << '\n';
  }

  // Record final state and scores
  game_replay["final_state"] = state_to_json(state.get());
  game_replay["final_scores"] = nlohmann::json::array();
  for (int player = 0; player < game->NumPlayers(); ++player) {
    game_replay["final_scores"].push_back(state->PlayerReturn(player));
  }

  std::cout << "\n=== GAME OVER ===\n";
  std::cout << "Final scores:\n";
  for (int player = 0; player < game->NumPlayers(); ++player) {
    std::string agent_name = (player == 0) ? agent1_name : agent2_name;
    std::cout << "Player " << player << " (" << agent_name
              << "): " << state->PlayerReturn(player) << '\n';
  }

  // Save game replay to file if requested
  if (!output_file.empty()) {
    std::ofstream out(output_file);
    out << game_replay.dump(2);
    std::cout << "\nGame replay saved to: " << output_file << '\n';
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
        cxxopts::value<bool>()->default_value("true"))(
        "o,output", "Output file for game replay (JSON format)",
        cxxopts::value<std::string>()->default_value(""))("h,help",
                                                          "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") != 0U) {
      std::cout << options.help() << '\n';
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
    auto alphazero_mcts_agent = azul::create_alphazero_mcts_evaluation_agent(
        "models/libtorch_alphazero_azul/checkpoint--1", 400, 1.4, 42,
        "AlphaZero_MCTS_400");
    auto minimax_agent = azul::create_minimax_evaluation_agent(3, "Minimax_D3");
    auto random_agent = azul::create_random_evaluation_agent(42, "Random");

    if (mode == "detailed") {
      detailed_game_evaluation(game, alphazero_mcts_agent, minimax_agent,
                               "AlphaZero MCTS (400 sims)", "Minimax (D3)",
                               result["output"].as<std::string>());
    } else {
      // Run tournament using the existing Tournament class
      azul::EvaluationConfig config;
      config.verbose = result["verbose"].as<bool>();
      config.num_games = result["games"].as<int>();

      azul::Tournament tournament(config);
      tournament.add_agent(std::move(alphazero_mcts_agent));
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