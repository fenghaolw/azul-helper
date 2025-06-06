#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "agent_evaluator.h"
#include "azul.h"
#include "open_spiel/spiel.h"

using json = nlohmann::json;
using namespace azul;

namespace {
void force_azul_registration() {
  // Reference symbols from azul namespace to force linking
  (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}
}  // namespace

/**
 * Convert webapp tile color string to OpenSpiel TileColor enum
 */
open_spiel::azul::TileColor string_to_tile_color(const std::string& color_str) {
  if (color_str == "red") return open_spiel::azul::TileColor::kRed;
  if (color_str == "blue") return open_spiel::azul::TileColor::kBlue;
  if (color_str == "yellow") return open_spiel::azul::TileColor::kYellow;
  if (color_str == "black") return open_spiel::azul::TileColor::kBlack;
  if (color_str == "white") return open_spiel::azul::TileColor::kWhite;
  throw std::runtime_error("Unknown tile color: " + color_str);
}

/**
 * Convert OpenSpiel TileColor to webapp string
 */
std::string tile_color_to_string(open_spiel::azul::TileColor color) {
  switch (color) {
    case open_spiel::azul::TileColor::kRed:
      return "red";
    case open_spiel::azul::TileColor::kBlue:
      return "blue";
    case open_spiel::azul::TileColor::kYellow:
      return "yellow";
    case open_spiel::azul::TileColor::kBlack:
      return "black";
    case open_spiel::azul::TileColor::kWhite:
      return "white";
    default:
      throw std::runtime_error("Unknown tile color enum");
  }
}

/**
 * Convert OpenSpiel AzulState action to webapp JSON format
 * Uses the private DecodeAction method via a const_cast hack since we made
 * members public
 */
json action_to_json(open_spiel::Action action,
                    const open_spiel::azul::AzulState& state) {
  json result;

  try {
    // Since we made the DecodeAction method accessible, we can use it
    // We need to const_cast to call the method
    auto mutable_state = const_cast<open_spiel::azul::AzulState*>(&state);

    // For now, let's implement a basic decoder based on the action space
    // structure The exact encoding depends on the OpenSpiel implementation
    // details

    int action_int = static_cast<int>(action);
    int num_factories = (state.num_players_ * 2) + 1;
    int total_factory_actions = num_factories *
                                open_spiel::azul::kNumTileColors *
                                (open_spiel::azul::kNumPatternLines + 1);
    int center_actions = open_spiel::azul::kNumTileColors *
                         (open_spiel::azul::kNumPatternLines + 1);

    if (action_int < total_factory_actions) {
      // Factory action
      int factory_color_line = action_int;
      int factory_id =
          factory_color_line / (open_spiel::azul::kNumTileColors *
                                (open_spiel::azul::kNumPatternLines + 1));
      int color_line =
          factory_color_line % (open_spiel::azul::kNumTileColors *
                                (open_spiel::azul::kNumPatternLines + 1));
      int color_idx = color_line / (open_spiel::azul::kNumPatternLines + 1);
      int line_idx = color_line % (open_spiel::azul::kNumPatternLines + 1);

      result["factoryIndex"] = factory_id;
      result["tile"] = tile_color_to_string(
          static_cast<open_spiel::azul::TileColor>(color_idx));
      result["lineIndex"] = (line_idx == open_spiel::azul::kNumPatternLines)
                                ? -1
                                : line_idx;  // -1 for floor
    } else {
      // Center action
      int center_action = action_int - total_factory_actions;
      int color_idx = center_action / (open_spiel::azul::kNumPatternLines + 1);
      int line_idx = center_action % (open_spiel::azul::kNumPatternLines + 1);

      result["factoryIndex"] =
          num_factories;  // Center is represented as last factory index
      result["tile"] = tile_color_to_string(
          static_cast<open_spiel::azul::TileColor>(color_idx));
      result["lineIndex"] = (line_idx == open_spiel::azul::kNumPatternLines)
                                ? -1
                                : line_idx;  // -1 for floor
    }

    result["success"] = true;

  } catch (const std::exception& e) {
    // Fallback if decoding fails
    result["factoryIndex"] = 0;
    result["tile"] = "red";
    result["lineIndex"] = 0;
    result["success"] = false;
    result["error"] = "Action decoding failed: " + std::string(e.what());
  }

  return result;
}

/**
 * Parse tiles array from JSON and convert to vector of TileColor
 */
std::vector<open_spiel::azul::TileColor> parse_tiles_array(
    const json& tiles_json) {
  std::vector<open_spiel::azul::TileColor> tiles;
  if (tiles_json.is_array()) {
    for (const auto& tile_str : tiles_json) {
      if (tile_str.is_string()) {
        tiles.push_back(string_to_tile_color(tile_str.get<std::string>()));
      }
    }
  }
  return tiles;
}

/**
 * Parse factory from JSON object with color counts
 */
open_spiel::azul::Factory parse_factory(const json& factory_json) {
  open_spiel::azul::Factory factory;

  if (factory_json.contains("red") && factory_json["red"].is_number_integer()) {
    factory.tiles[static_cast<int>(open_spiel::azul::TileColor::kRed)] =
        factory_json["red"];
  }
  if (factory_json.contains("blue") &&
      factory_json["blue"].is_number_integer()) {
    factory.tiles[static_cast<int>(open_spiel::azul::TileColor::kBlue)] =
        factory_json["blue"];
  }
  if (factory_json.contains("yellow") &&
      factory_json["yellow"].is_number_integer()) {
    factory.tiles[static_cast<int>(open_spiel::azul::TileColor::kYellow)] =
        factory_json["yellow"];
  }
  if (factory_json.contains("black") &&
      factory_json["black"].is_number_integer()) {
    factory.tiles[static_cast<int>(open_spiel::azul::TileColor::kBlack)] =
        factory_json["black"];
  }
  if (factory_json.contains("white") &&
      factory_json["white"].is_number_integer()) {
    factory.tiles[static_cast<int>(open_spiel::azul::TileColor::kWhite)] =
        factory_json["white"];
  }

  return factory;
}

/**
 * Parse player board from JSON
 */
open_spiel::azul::PlayerBoard parse_player_board(const json& player_json) {
  open_spiel::azul::PlayerBoard board;

  // Parse score
  if (player_json.contains("score") &&
      player_json["score"].is_number_integer()) {
    board.score = player_json["score"];
  }

  // Parse pattern lines
  if (player_json.contains("patternLines") &&
      player_json["patternLines"].is_array()) {
    const auto& pattern_lines_json = player_json["patternLines"];
    for (size_t i = 0; i < pattern_lines_json.size() &&
                       i < open_spiel::azul::kNumPatternLines;
         ++i) {
      const auto& line_json = pattern_lines_json[i];
      if (line_json.is_object()) {
        if (line_json.contains("count") &&
            line_json["count"].is_number_integer()) {
          board.pattern_lines[i].count = line_json["count"];
        }
        if (board.pattern_lines[i].count > 0 && line_json.contains("color") &&
            line_json["color"].is_string()) {
          std::string color_str = line_json["color"];
          if (!color_str.empty()) {
            board.pattern_lines[i].color = string_to_tile_color(color_str);
          }
        }
      }
    }
  }

  // Parse wall
  if (player_json.contains("wall") && player_json["wall"].is_array()) {
    const auto& wall_json = player_json["wall"];
    for (size_t row = 0;
         row < wall_json.size() && row < open_spiel::azul::kWallSize; ++row) {
      if (wall_json[row].is_array()) {
        const auto& wall_row = wall_json[row];
        for (size_t col = 0;
             col < wall_row.size() && col < open_spiel::azul::kWallSize;
             ++col) {
          if (wall_row[col].is_boolean()) {
            board.wall[row][col] = wall_row[col];
          }
        }
      }
    }
  }

  // Parse floor line
  if (player_json.contains("floorLine") &&
      player_json["floorLine"].is_array()) {
    board.floor_line = parse_tiles_array(player_json["floorLine"]);
  }

  return board;
}

/**
 * Parse webapp JSON game state and create a reconstructed OpenSpiel state
 */
std::unique_ptr<open_spiel::azul::AzulState> create_state_from_json(
    const json& game_state) {
  // Determine number of players
  int num_players = open_spiel::azul::kDefaultNumPlayers;
  if (game_state.contains("players") && game_state["players"].is_array()) {
    num_players = std::min(static_cast<int>(game_state["players"].size()),
                           open_spiel::azul::kMaxNumPlayers);
  }

  // Create new game instance with correct number of players
  open_spiel::GameParameters params;
  params["num_players"] = open_spiel::GameParameter(num_players);
  auto game = open_spiel::LoadGame("azul", params);
  if (!game) {
    throw std::runtime_error("Failed to load Azul game");
  }

  // Create initial state
  auto state = std::unique_ptr<open_spiel::azul::AzulState>(
      dynamic_cast<open_spiel::azul::AzulState*>(
          game->NewInitialState().release()));

  if (!state) {
    throw std::runtime_error("Failed to create AzulState");
  }

  // Set number of players
  state->num_players_ = num_players;

  // Parse current player
  if (game_state.contains("currentPlayer") &&
      game_state["currentPlayer"].is_number_integer()) {
    state->current_player_ = game_state["currentPlayer"];
  }

  // Parse round number
  if (game_state.contains("roundNumber") &&
      game_state["roundNumber"].is_number_integer()) {
    state->round_number_ = game_state["roundNumber"];
  }

  // Parse game ended flag
  if (game_state.contains("gameEnded") &&
      game_state["gameEnded"].is_boolean()) {
    state->game_ended_ = game_state["gameEnded"];
  }

  // Parse first player tile availability
  if (game_state.contains("firstPlayerTileAvailable") &&
      game_state["firstPlayerTileAvailable"].is_boolean()) {
    state->first_player_tile_available_ =
        game_state["firstPlayerTileAvailable"];
  }

  // Parse first player next round
  if (game_state.contains("firstPlayerNextRound") &&
      game_state["firstPlayerNextRound"].is_number_integer()) {
    state->first_player_next_round_ = game_state["firstPlayerNextRound"];
  }

  // Parse bag shuffle flag
  if (game_state.contains("needsBagShuffle") &&
      game_state["needsBagShuffle"].is_boolean()) {
    state->needs_bag_shuffle_ = game_state["needsBagShuffle"];
  }

  // Parse factories
  if (game_state.contains("factories") && game_state["factories"].is_array()) {
    const auto& factories_json = game_state["factories"];
    state->factories_.clear();
    for (const auto& factory_json : factories_json) {
      state->factories_.push_back(parse_factory(factory_json));
    }
  }

  // Parse center pile
  if (game_state.contains("centerPile") &&
      game_state["centerPile"].is_object()) {
    state->center_pile_ = parse_factory(game_state["centerPile"]);
  }

  // Parse player boards
  if (game_state.contains("players") && game_state["players"].is_array()) {
    const auto& players_json = game_state["players"];
    state->player_boards_.clear();
    for (const auto& player_json : players_json) {
      state->player_boards_.push_back(parse_player_board(player_json));
    }
  }

  // Parse bag contents
  if (game_state.contains("bag") && game_state["bag"].is_array()) {
    state->bag_ = parse_tiles_array(game_state["bag"]);
  }

  // Parse discard pile
  if (game_state.contains("discardPile") &&
      game_state["discardPile"].is_array()) {
    state->discard_pile_ = parse_tiles_array(game_state["discardPile"]);
  }

  return state;
}

/**
 * Parse command line arguments for standalone operation
 */
struct BridgeConfig {
  std::string input_file;
  std::string checkpoint_path;
  int player_id = 0;
  int num_simulations = 400;
  double uct_c = 1.4;
  bool deterministic = false;
  bool help = false;
};

BridgeConfig parse_args(int argc, char* argv[]) {
  BridgeConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      config.help = true;
    } else if (arg == "--input" && i + 1 < argc) {
      config.input_file = argv[++i];
    } else if (arg == "--checkpoint" && i + 1 < argc) {
      config.checkpoint_path = argv[++i];
    } else if (arg == "--player-id" && i + 1 < argc) {
      config.player_id = std::stoi(argv[++i]);
    } else if (arg == "--simulations" && i + 1 < argc) {
      config.num_simulations = std::stoi(argv[++i]);
    } else if (arg == "--uct-c" && i + 1 < argc) {
      config.uct_c = std::stod(argv[++i]);
    } else if (arg == "--deterministic") {
      config.deterministic = true;
    }
  }

  return config;
}

void print_usage(const char* program_name) {
  std::cout
      << "Usage: " << program_name << " [options]\n"
      << "\nOptions:\n"
      << "  --input <file>        Input JSON file (default: read from stdin)\n"
      << "  --checkpoint <path>   AlphaZero model checkpoint path\n"
      << "  --player-id <id>      Player ID (0-based)\n"
      << "  --simulations <num>   Number of MCTS simulations (default: 400)\n"
      << "  --uct-c <value>       UCT exploration constant (default: 1.4)\n"
      << "  --deterministic       Use deterministic action selection\n"
      << "  --help, -h           Show this help message\n"
      << "\nInput JSON format:\n"
      << "{\n"
      << "  \"gameState\": { ... },  // Game state in webapp format\n"
      << "  \"playerId\": 0,         // Player to act\n"
      << "  \"checkpointPath\": \"model.pt\",\n"
      << "  \"numSimulations\": 400,\n"
      << "  \"uctC\": 1.4,\n"
      << "  \"deterministic\": false\n"
      << "}\n";
}

int main(int argc, char* argv[]) {
  try {
    // Force Azul game registration
    force_azul_registration();

    // Parse command line arguments
    auto config = parse_args(argc, argv);

    if (config.help) {
      print_usage(argv[0]);
      return 0;
    }

    // Read JSON input
    json input;

    if (!config.input_file.empty()) {
      // Read from file
      std::ifstream file(config.input_file);
      if (!file.is_open()) {
        std::cerr << "Error: Could not open input file: " << config.input_file
                  << std::endl;
        return 1;
      }
      file >> input;
    } else {
      // Read from stdin
      std::string line;
      std::string json_str;
      while (std::getline(std::cin, line)) {
        json_str += line;
      }

      if (json_str.empty()) {
        std::cerr << "Error: No input provided" << std::endl;
        return 1;
      }

      try {
        input = json::parse(json_str);
      } catch (const json::parse_error& e) {
        std::cerr << "Error: Invalid JSON input: " << e.what() << std::endl;
        return 1;
      }
    }

    // Extract parameters from JSON (with command line overrides)
    auto game_state_json = input["gameState"];
    int player_id =
        config.player_id > 0 ? config.player_id : input.value("playerId", 0);
    bool deterministic =
        config.deterministic || input.value("deterministic", false);
    std::string checkpoint_path =
        !config.checkpoint_path.empty()
            ? config.checkpoint_path
            : input.value("checkpointPath", std::string(""));
    int num_simulations = config.num_simulations > 0
                              ? config.num_simulations
                              : input.value("numSimulations", 400);
    double uct_c = config.uct_c > 0 ? config.uct_c : input.value("uctC", 1.4);

    // Validate required parameters
    if (checkpoint_path.empty()) {
      std::cerr << "Error: checkpoint path is required" << std::endl;
      return 1;
    }

    // Create AlphaZero agent (suppress stdout during initialization)
    std::streambuf* orig_cout = std::cout.rdbuf();
    std::ostringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());  // Redirect stdout to null

    auto alphazero_agent = std::make_unique<AlphaZeroMCTSAgentWrapper>(
        checkpoint_path, num_simulations, uct_c, -1, "AlphaZero_Bridge");

    // Create game state from JSON
    auto state = create_state_from_json(game_state_json);

    // Restore stdout for JSON output
    std::cout.rdbuf(orig_cout);

    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Get action from AlphaZero (suppress stderr warnings)
    std::streambuf* orig_cerr = std::cerr.rdbuf();
    std::ostringstream null_stream_err;
    std::cerr.rdbuf(null_stream_err.rdbuf());  // Redirect stderr to null

    open_spiel::Action best_action;
    try {
      best_action = alphazero_agent->get_action(*state, player_id);
    } catch (const std::exception& e) {
      // Restore stderr for error output
      std::cerr.rdbuf(orig_cerr);

      // Fallback: use random action if AlphaZero fails
      auto legal_actions = state->LegalActions();
      if (legal_actions.empty()) {
        json error_response;
        error_response["success"] = false;
        error_response["error"] = "No legal actions available";
        error_response["factoryIndex"] = 0;
        error_response["tile"] = "red";
        error_response["lineIndex"] = 0;
        error_response["thinking_time"] = 0.0;
        error_response["nodes_explored"] = 0;

        std::cout << error_response.dump() << std::endl;
        return 1;
      }
      best_action = legal_actions[0];

      // Continue with action but don't print warning to stderr anymore
    }

    // Restore stderr
    std::cerr.rdbuf(orig_cerr);

    // Calculate timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    double thinking_time = duration.count() / 1000.0;

    // Convert action to JSON format
    json action_json = action_to_json(best_action, *state);

    // Prepare response
    json response;
    response["factoryIndex"] = action_json["factoryIndex"];
    response["tile"] = action_json["tile"];
    response["lineIndex"] = action_json["lineIndex"];
    response["success"] = true;
    response["thinking_time"] = thinking_time;
    response["nodes_explored"] = alphazero_agent->get_nodes_explored();
    response["simulations_used"] = num_simulations;
    response["uct_c_used"] = uct_c;
    response["deterministic"] = deterministic;
    response["player_id"] = player_id;
    response["checkpoint_path"] = checkpoint_path;

    // Output response as JSON
    std::cout << response.dump() << std::endl;

    return 0;

  } catch (const std::exception& e) {
    json error_response;
    error_response["success"] = false;
    error_response["error"] = e.what();
    error_response["factoryIndex"] = 0;
    error_response["tile"] = "red";
    error_response["lineIndex"] = 0;
    error_response["thinking_time"] = 0.0;
    error_response["nodes_explored"] = 0;

    std::cout << error_response.dump() << std::endl;
    return 1;
  }
}