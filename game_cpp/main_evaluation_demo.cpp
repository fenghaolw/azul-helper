#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
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

// Helper function to generate random seeds
int generate_random_seed() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> dist(
      1, 2147483647); // avoid negative seeds
  return dist(gen);
}

// Helper function to convert game state to JSON format
nlohmann::json state_to_json(const open_spiel::State *state) {
  const auto *azul_state =
      dynamic_cast<const open_spiel::azul::AzulState *>(state);
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
  for (const auto &factory : azul_state->Factories()) {
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
    const auto &board = azul_state->GetPlayerBoard(player);
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
    for (const auto &tile : board.floor_line) {
      board_json["floor"].push_back(open_spiel::azul::TileColorToString(tile));
    }

    state_json["playerBoards"].push_back(board_json);
  }

  return state_json;
}

// Helper function to display game state in compact format
std::string compact_state_display(const open_spiel::State *state) {
  const auto *azul_state =
      dynamic_cast<const open_spiel::azul::AzulState *>(state);
  if (!azul_state) {
    return "Invalid state";
  }

  std::ostringstream oss;

  // Header with round and current player
  oss << "Round " << azul_state->round_number_ << " | Player "
      << azul_state->CurrentPlayer() << "'s turn";
  if (azul_state->HasFirstPlayerTile()) {
    oss << " | First player tile available";
  }
  oss << "\n";

  // Factories in a single line
  oss << "Factories: ";
  const auto &factories = azul_state->Factories();
  for (size_t i = 0; i < factories.size(); ++i) {
    if (i > 0)
      oss << " | ";
    oss << "F" << i << ":";
    bool has_tiles = false;
    for (int color = 0; color < open_spiel::azul::kNumTileColors; ++color) {
      if (factories[i].tiles[color] > 0) {
        if (has_tiles)
          oss << ",";
        oss << open_spiel::azul::TileColorToString(
                   static_cast<open_spiel::azul::TileColor>(color))[0]
            << factories[i].tiles[color];
        has_tiles = true;
      }
    }
    if (!has_tiles)
      oss << "Empty";
  }
  oss << "\n";

  // Center pile
  oss << "Center: ";
  const auto &center = azul_state->CenterPile();
  bool has_center_tiles = false;
  for (int color = 0; color < open_spiel::azul::kNumTileColors; ++color) {
    if (center.tiles[color] > 0) {
      if (has_center_tiles)
        oss << ",";
      oss << open_spiel::azul::TileColorToString(
                 static_cast<open_spiel::azul::TileColor>(color))[0]
          << center.tiles[color];
      has_center_tiles = true;
    }
  }
  if (!has_center_tiles)
    oss << "Empty";
  oss << "\n";

  // Helper function to generate board display for a player
  auto generate_player_board = [](const open_spiel::azul::PlayerBoard &board,
                                  int player_num) -> std::vector<std::string> {
    std::vector<std::string> lines;

    // Header line
    lines.push_back("Player " + std::to_string(player_num) +
                    " (Score: " + std::to_string(board.score) + "):");

    // Pattern lines and wall
    for (int row = 0; row < open_spiel::azul::kWallSize; ++row) {
      std::string line;

      // Pattern line (right-aligned based on row number)
      // Row 0 can hold 1 tile, row 1 can hold 2 tiles, etc.
      std::string pattern_part;

      if (board.pattern_lines[row].count > 0) {
        char color_char = open_spiel::azul::TileColorToString(
            board.pattern_lines[row].color)[0];
        pattern_part = std::string(board.pattern_lines[row].count, color_char);
      }

      // Create the pattern section (8 characters total, right-aligned)
      std::string pattern_section(8, ' ');
      if (!pattern_part.empty()) {
        int start_pos = 8 - pattern_part.length();
        if (start_pos < 0)
          start_pos = 0;
        for (size_t i = 0; i < pattern_part.length() && (start_pos + i) < 8;
             ++i) {
          pattern_section[start_pos + i] = pattern_part[i];
        }
      }

      line += pattern_section;

      // Separator
      line += " | ";

      // Wall row
      for (int col = 0; col < open_spiel::azul::kWallSize; ++col) {
        if (board.wall[row][col]) {
          // Get the color that should be at this position using the actual wall
          // pattern
          line += open_spiel::azul::TileColorToString(
              open_spiel::azul::kWallPattern[row][col])[0];
        } else {
          line += '.';
        }
      }

      lines.push_back(line);
    }

    // Floor line
    std::string floor_line = "Floor: ";
    if (!board.floor_line.empty()) {
      for (const auto &tile : board.floor_line) {
        floor_line += open_spiel::azul::TileColorToString(tile)[0];
      }
    }
    lines.push_back(floor_line);

    return lines;
  };

  // Player boards side by side
  oss << "\nPlayer Boards:\n";
  if (azul_state->num_players_ >= 2) {
    auto player0_lines =
        generate_player_board(azul_state->GetPlayerBoard(0), 0);
    auto player1_lines =
        generate_player_board(azul_state->GetPlayerBoard(1), 1);

    // Find max width for player 0's lines for proper spacing
    size_t max_width = 0;
    for (const auto &line : player0_lines) {
      max_width = std::max(max_width, line.length());
    }

    // Display lines side by side
    size_t max_lines = std::max(player0_lines.size(), player1_lines.size());
    for (size_t i = 0; i < max_lines; ++i) {
      std::string left_line =
          (i < player0_lines.size()) ? player0_lines[i] : "";
      std::string right_line =
          (i < player1_lines.size()) ? player1_lines[i] : "";

      // Pad left line to consistent width (ensure minimum spacing)
      const size_t min_spacing = 4;
      if (left_line.length() < max_width) {
        left_line += std::string(max_width - left_line.length(), ' ');
      }

      oss << left_line << std::string(min_spacing, ' ') << right_line << "\n";
    }
  } else {
    // Fallback for single player (shouldn't happen in Azul but just in case)
    auto player0_lines =
        generate_player_board(azul_state->GetPlayerBoard(0), 0);
    for (const auto &line : player0_lines) {
      oss << line << "\n";
    }
  }

  return oss.str();
}

void detailed_game_evaluation(
    const std::shared_ptr<const open_spiel::Game> &game,
    std::unique_ptr<azul::EvaluationAgent> &agent1,
    std::unique_ptr<azul::EvaluationAgent> &agent2,
    const std::string &output_file = "") {
  auto state = game->NewInitialState();
  std::cout << "\n=== INITIAL GAME STATE ===\n"
            << compact_state_display(state.get()) << '\n';

  // Create JSON structure for game replay
  nlohmann::json game_replay;
  game_replay["agent1_name"] = agent1->get_name();
  game_replay["agent2_name"] = agent2->get_name();
  game_replay["moves"] = nlohmann::json::array();
  game_replay["initial_state"] = state_to_json(state.get());

  while (!state->IsTerminal()) {
    int current_player = state->CurrentPlayer();
    auto &agent = (current_player == 0) ? *agent1 : *agent2;
    std::string agent_name =
        (current_player == 0) ? agent1->get_name() : agent2->get_name();

    auto action = agent.get_action(*state, current_player);
    std::string action_str = state->ActionToString(current_player, action);

    std::cout << "\n--- Player " << current_player << " (" << agent_name
              << ") takes action: " << action_str << " ---\n";

    // Record move in JSON
    nlohmann::json move;
    move["player"] = current_player;
    move["agent"] = agent_name;
    move["action"] = action_str;
    state->ApplyAction(action);
    move["state_after"] = state_to_json(state.get());
    game_replay["moves"].push_back(move);

    std::cout << compact_state_display(state.get()) << '\n';
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
    std::string agent_name =
        (player == 0) ? agent1->get_name() : agent2->get_name();
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

} // namespace

auto main(int argc, char *argv[]) -> int {
  std::cout << "=== AZUL AGENT EVALUATION DEMO ===" << '\n';

  try {
    // Parse command line options
    cxxopts::Options options("azul_evaluation", "Azul Agent Evaluation Demo");
    options.add_options()(
        "m,mode", "Evaluation mode: 'detailed' or 'tournament'",
        cxxopts::value<std::string>()->default_value("tournament"))(
        "g,games", "Number of games per matchup in tournament mode",
        cxxopts::value<int>()->default_value("10"))(
        "s,sims", "Number of simulations for AlphaZero MCTS agent",
        cxxopts::value<int>()->default_value("1000"))(
        "d,depth", "Search depth for Minimax agent",
        cxxopts::value<int>()->default_value("3"))(
        "v,verbose", "Enable verbose output",
        cxxopts::value<bool>()->default_value("true"))(
        "o,output", "Output file for game replay (JSON format)",
        cxxopts::value<std::string>()->default_value(""))(
        "r,random-seeds", "Use random seeds for each game (default: true)",
        cxxopts::value<bool>()->default_value("true"))(
        "seed", "Fixed seed to use when random-seeds is false",
        cxxopts::value<int>()->default_value("42"))("h,help", "Print usage");

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

    bool use_random_seeds = result["random-seeds"].as<bool>();
    int fixed_seed = result["seed"].as<int>();

    // For AlphaZero, seed is only used for fallback random rollouts or
    // tie breaking. The main MCTS search is based on the neural network
    // weights, not the seed. We use a fixed seed here since it does not matter.
    auto az_agent = azul::create_alphazero_mcts_evaluation_agent(
        "models/libtorch_alphazero_azul/checkpoint--1",
        result["sims"].as<int>(), 1.4, 42,
        "AlphaZero_MCTS_" + std::to_string(result["sims"].as<int>()));
    auto minimax_agent = azul::create_minimax_evaluation_agent(
        result["depth"].as<int>(),
        "Minimax_D" + std::to_string(result["depth"].as<int>()));

    if (use_random_seeds) {
      std::cout << "✅ Using random seeds for each game\n";
    } else {
      std::cout << "✅ Using fixed seed " << fixed_seed << " for all games\n";
    }

    if (mode == "detailed") {
      detailed_game_evaluation(game, az_agent, minimax_agent,
                               result["output"].as<std::string>());
    } else {
      // Use AgentEvaluator for direct comparison
      azul::EvaluationConfig config;
      config.verbose = result["verbose"].as<bool>();
      config.num_games = result["games"].as<int>();
      config.use_fixed_seeds = !use_random_seeds;
      config.random_seed = fixed_seed;

      azul::AgentEvaluator evaluator(config);
      auto eval_result = evaluator.evaluate_agent(*az_agent, *minimax_agent);

      // Print evaluation results
      std::cout << "\n=== EVALUATION RESULTS ===\n";
      std::cout << "Agent 1: " << eval_result.test_agent_name << "\n";
      std::cout << "Agent 2: " << eval_result.baseline_agent_name << "\n";
      std::cout << "Games played: " << eval_result.games_played << "\n";
      std::cout << eval_result.test_agent_name
                << " wins: " << eval_result.test_agent_wins << "\n";
      std::cout << "Win rate: " << (eval_result.test_agent_win_rate * 100.0)
                << "%\n";
      std::cout << "Statistical significance: "
                << (eval_result.is_statistically_significant ? "Yes" : "No")
                << "\n";
      if (eval_result.confidence_interval.first != 0 ||
          eval_result.confidence_interval.second != 0) {
        std::cout << "95% Confidence interval: ["
                  << eval_result.confidence_interval.first * 100 << "%, "
                  << eval_result.confidence_interval.second * 100 << "%]\n";
      }
    }

  } catch (const cxxopts::exceptions::exception &e) {
    std::cerr << "❌ Error parsing options: " << e.what() << '\n';
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "❌ Error during evaluation: " << e.what() << '\n';
    return 1;
  }

  return 0;
}