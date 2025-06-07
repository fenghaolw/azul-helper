#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// HTTP server library (header-only)
#include <httplib.h>

// JSON library
#include <nlohmann/json.hpp>

// OpenSpiel and Azul includes
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
 * C++ API Server for Azul Agents
 *
 * This server provides a direct C++ HTTP API that:
 * 1. Receives JSON requests from the webapp
 * 2. Converts JSON gamestate to OpenSpiel AzulState
 * 3. Calls the selected agent directly (no subprocess overhead)
 * 4. Returns JSON responses
 *
 * Supports multiple agent types: AlphaZero, Random, MCTS, Minimax
 */
class AzulApiServer {
 private:
  httplib::Server server_;
  std::unique_ptr<azul::EvaluationAgent> current_agent_;
  std::string agent_type_;
  std::string checkpoint_path_;
  int num_simulations_;
  double uct_c_;
  int port_;
  int seed_;

 public:
  AzulApiServer(const std::string& agent_type = "alphazero",
                const std::string& checkpoint_path = "",
                int num_simulations = 2000, double uct_c = 1.4, int port = 5001,
                int seed = -1)
      : agent_type_(agent_type),
        checkpoint_path_(checkpoint_path),
        num_simulations_(num_simulations),
        uct_c_(uct_c),
        port_(port),
        seed_(seed) {
    // Force registration
    force_azul_registration();

    // Initialize the selected agent
    initialize_agent();
    setup_routes();
  }

 private:
  void initialize_agent() {
    std::cout << "🚀 Initializing " << agent_type_ << " agent..." << std::endl;

    if (agent_type_ == "alphazero") {
      if (checkpoint_path_.empty()) {
        throw std::runtime_error("AlphaZero agent requires checkpoint path");
      }
      current_agent_ = azul::create_alphazero_mcts_evaluation_agent(
          checkpoint_path_, num_simulations_, uct_c_, seed_,
          "AlphaZero_API_Server");

    } else if (agent_type_ == "random") {
      current_agent_ =
          azul::create_random_evaluation_agent(seed_, "Random_API_Server");

    } else if (agent_type_ == "mcts") {
      current_agent_ = azul::create_mcts_evaluation_agent(
          num_simulations_, uct_c_, seed_, "MCTS_API_Server");

    } else if (agent_type_ == "minimax") {
      // For minimax, use num_simulations as depth (clamped to reasonable range)
      int depth = std::max(1, std::min(6, num_simulations_ / 100));
      current_agent_ =
          azul::create_minimax_evaluation_agent(depth, "Minimax_API_Server");

    } else {
      throw std::runtime_error(
          "Unknown agent type: " + agent_type_ +
          ". Supported types: alphazero, random, mcts, minimax");
    }

    std::cout << "✅ " << current_agent_->get_name() << " initialized"
              << std::endl;
  }

  void setup_routes() {
    // Enable CORS for webapp integration
    server_.set_pre_routing_handler(
        [](const httplib::Request& req, httplib::Response& res) {
          res.set_header("Access-Control-Allow-Origin", "*");
          res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
          res.set_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization");
          return httplib::Server::HandlerResponse::Unhandled;
        });

    // Handle preflight OPTIONS requests
    server_.Options(
        ".*", [](const httplib::Request&, httplib::Response& res) { return; });

    // Simple health check endpoint
    server_.Get("/health",
                [this](const httplib::Request&, httplib::Response& res) {
                  json response;
                  response["status"] = "healthy";
                  response["agent_type"] = agent_type_;
                  response["agent_name"] = current_agent_->get_name();
                  res.set_content(response.dump(), "application/json");
                });

    // Main move endpoint - get best move from current agent
    server_.Post("/agent/move", [this](const httplib::Request& req,
                                       httplib::Response& res) {
      try {
        // Parse JSON request
        json request_json = json::parse(req.body);

        // Extract parameters
        auto game_state_json = request_json["gameState"];
        int player_id = request_json.value("playerId", 0);

        // Create OpenSpiel state from JSON
        auto state = create_state_from_json(game_state_json);

        // Record start time
        auto start_time = std::chrono::high_resolution_clock::now();

        // Get action from current agent
        open_spiel::Action best_action =
            current_agent_->get_action(*state, player_id);

        // Calculate timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        double thinking_time = duration.count() / 1000.0;

        // Convert action to webapp JSON format
        json action_json = action_to_json(best_action, *state);

        // Prepare response
        json response;
        response["success"] = true;
        response["move"] = action_json;
        response["stats"] = {
            {"agent_type", agent_type_},
            {"agent_name", current_agent_->get_name()},
            {"nodesEvaluated", current_agent_->get_nodes_explored()},
            {"searchTime", thinking_time}};

        res.set_content(response.dump(), "application/json");

      } catch (const std::exception& e) {
        json error_response;
        error_response["success"] = false;
        error_response["error"] = e.what();

        res.status = 400;
        res.set_content(error_response.dump(), "application/json");
      }
    });
  }

 public:
  void start() {
    std::cout << "🌐 Starting Azul C++ API Server on port " << port_
              << std::endl;
    std::cout << "🧠 Agent: " << current_agent_->get_name() << " ("
              << agent_type_ << ", " << num_simulations_
              << " simulations, UCT=" << uct_c_ << ")" << std::endl;
    if (!checkpoint_path_.empty()) {
      std::cout << "📁 Checkpoint: " << checkpoint_path_ << std::endl;
    }
    std::cout << "🎯 Available endpoints:" << std::endl;
    std::cout << "  GET  /health - Health check" << std::endl;
    std::cout << "  POST /agent/move - Get best move" << std::endl;
    std::cout << "✅ Server ready!" << std::endl;

    server_.listen("0.0.0.0", port_);
  }

  void stop() { server_.stop(); }

 private:
  /**
   * Convert webapp tile color string to OpenSpiel TileColor enum
   */
  open_spiel::azul::TileColor string_to_tile_color(
      const std::string& color_str) {
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

    if (factory_json.contains("red") &&
        factory_json["red"].is_number_integer()) {
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
   * Create OpenSpiel AzulState from webapp JSON gamestate with full
   * reconstruction
   */
  std::unique_ptr<open_spiel::azul::AzulState> create_state_from_json(
      const json& game_state_json) {
    // Determine number of players
    int num_players = open_spiel::azul::kDefaultNumPlayers;
    if (game_state_json.contains("players") &&
        game_state_json["players"].is_array()) {
      num_players =
          std::min(static_cast<int>(game_state_json["players"].size()),
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
    if (game_state_json.contains("currentPlayer") &&
        game_state_json["currentPlayer"].is_number_integer()) {
      state->current_player_ = game_state_json["currentPlayer"];
    }

    // Parse round number
    if (game_state_json.contains("roundNumber") &&
        game_state_json["roundNumber"].is_number_integer()) {
      state->round_number_ = game_state_json["roundNumber"];
    }

    // Parse game ended flag
    if (game_state_json.contains("gameEnded") &&
        game_state_json["gameEnded"].is_boolean()) {
      state->game_ended_ = game_state_json["gameEnded"];
    }

    // Parse first player tile availability
    if (game_state_json.contains("firstPlayerTileAvailable") &&
        game_state_json["firstPlayerTileAvailable"].is_boolean()) {
      state->first_player_tile_available_ =
          game_state_json["firstPlayerTileAvailable"];
    }

    // Parse first player next round
    if (game_state_json.contains("firstPlayerNextRound") &&
        game_state_json["firstPlayerNextRound"].is_number_integer()) {
      state->first_player_next_round_ = game_state_json["firstPlayerNextRound"];
    }

    // Parse bag shuffle flag
    if (game_state_json.contains("needsBagShuffle") &&
        game_state_json["needsBagShuffle"].is_boolean()) {
      state->needs_bag_shuffle_ = game_state_json["needsBagShuffle"];
    }

    // Parse factories
    if (game_state_json.contains("factories") &&
        game_state_json["factories"].is_array()) {
      const auto& factories_json = game_state_json["factories"];
      state->factories_.clear();
      for (const auto& factory_json : factories_json) {
        state->factories_.push_back(parse_factory(factory_json));
      }
    }

    // Parse center pile
    if (game_state_json.contains("centerPile") &&
        game_state_json["centerPile"].is_object()) {
      state->center_pile_ = parse_factory(game_state_json["centerPile"]);
    }

    // Parse player boards
    if (game_state_json.contains("players") &&
        game_state_json["players"].is_array()) {
      const auto& players_json = game_state_json["players"];
      state->player_boards_.clear();
      for (const auto& player_json : players_json) {
        state->player_boards_.push_back(parse_player_board(player_json));
      }
    }

    // Parse bag contents
    if (game_state_json.contains("bag") && game_state_json["bag"].is_array()) {
      state->bag_ = parse_tiles_array(game_state_json["bag"]);
    }

    // Parse discard pile
    if (game_state_json.contains("discardPile") &&
        game_state_json["discardPile"].is_array()) {
      state->discard_pile_ = parse_tiles_array(game_state_json["discardPile"]);
    }

    return state;
  }

  /**
   * Convert OpenSpiel action to webapp JSON format using AzulState's
   * DecodeAction
   */
  json action_to_json(open_spiel::Action action,
                      const open_spiel::azul::AzulState& state) {
    json action_json;

    try {
      // Use the proper DecodeAction method from AzulState
      auto decoded = state.DecodeAction(action);

      // Map to webapp JSON format
      if (decoded.from_center) {
        // Center pile is represented as -1 in webapp
        action_json["factoryIndex"] = -1;
      } else {
        action_json["factoryIndex"] = decoded.factory_id;
      }

      // Convert tile color to webapp string format
      action_json["tile"] = tile_color_to_string(decoded.color);
      action_json["lineIndex"] =
          decoded.destination;  // Already -1 for floor line

    } catch (const std::exception& e) {
      // Fallback if decoding fails
      action_json["factoryIndex"] = 0;
      action_json["tile"] = "red";
      action_json["lineIndex"] = 0;
      action_json["error"] = "Action decoding failed: " + std::string(e.what());
    }

    return action_json;
  }
};

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --agent <type>       Agent type: alphazero, mcts, random, "
               "minimax (default: alphazero)\n";
  std::cout << "  --checkpoint <path>  AlphaZero checkpoint path (required for "
               "alphazero)\n";
  std::cout << "  --port <n>           Server port (default: 5001)\n";
  std::cout
      << "  --simulations <n>    Number of MCTS simulations (default: 800)\n";
  std::cout
      << "  --uct <c>            UCT exploration constant (default: 1.4)\n";
  std::cout << "  --seed <n>           Random seed (default: -1 for random)\n";
  std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
  std::string agent_type = "alphazero";  // Default to AlphaZero
  std::string checkpoint_path;
  int port = 5001;
  int num_simulations = 800;
  double uct_c = 1.4;
  int seed = -1;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--agent" && i + 1 < argc) {
      agent_type = argv[++i];
    } else if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint_path = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (arg == "--simulations" && i + 1 < argc) {
      num_simulations = std::stoi(argv[++i]);
    } else if (arg == "--uct" && i + 1 < argc) {
      uct_c = std::stod(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoi(argv[++i]);
    }
  }

  // Validate required parameters
  if (agent_type == "alphazero" && checkpoint_path.empty()) {
    std::cerr
        << "❌ Error: --checkpoint parameter is required for alphazero agent"
        << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  try {
    // Create and start the server
    AzulApiServer server(agent_type, checkpoint_path, num_simulations, uct_c,
                         port, seed);

    // Handle Ctrl+C gracefully
    std::signal(SIGINT, [](int) {
      std::cout << "\n🛑 Shutting down server..." << std::endl;
      std::exit(0);
    });

    server.start();

  } catch (const std::exception& e) {
    std::cerr << "❌ Server error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}