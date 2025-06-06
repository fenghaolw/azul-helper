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
 * C++ API Server for Azul AlphaZero Agent
 *
 * This server provides a direct C++ HTTP API that:
 * 1. Receives JSON requests from the webapp
 * 2. Converts JSON gamestate to OpenSpiel AzulState
 * 3. Calls AlphaZero agent directly (no subprocess overhead)
 * 4. Returns JSON responses
 */
class AzulApiServer {
 private:
  httplib::Server server_;
  std::unique_ptr<AlphaZeroMCTSAgentWrapper> alphazero_agent_;
  std::string checkpoint_path_;
  int num_simulations_;
  double uct_c_;
  int port_;

 public:
  AzulApiServer(const std::string& checkpoint_path, int num_simulations = 800,
                double uct_c = 1.4, int port = 5001)
      : checkpoint_path_(checkpoint_path),
        num_simulations_(num_simulations),
        uct_c_(uct_c),
        port_(port) {
    // Force registration
    force_azul_registration();

    // Initialize AlphaZero agent
    std::cout << "🚀 Initializing AlphaZero agent..." << std::endl;
    alphazero_agent_ = std::make_unique<AlphaZeroMCTSAgentWrapper>(
        checkpoint_path_, num_simulations_, uct_c_, -1, "AlphaZero_API_Server");
    std::cout << "✅ AlphaZero agent initialized" << std::endl;

    setup_routes();
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

    // Health check endpoint
    server_.Get("/health",
                [this](const httplib::Request&, httplib::Response& res) {
                  json response;
                  response["status"] = "healthy";
                  response["agent_type"] = "alphazero";
                  response["checkpoint_path"] = checkpoint_path_;
                  response["num_simulations"] = num_simulations_;
                  response["uct_c"] = uct_c_;
                  response["port"] = port_;

                  res.set_content(response.dump(), "application/json");
                });

    // Agent info endpoint
    server_.Get("/agent/info",
                [this](const httplib::Request&, httplib::Response& res) {
                  json response;
                  response["agent_type"] = "alphazero";
                  response["algorithm"] = "AlphaZero MCTS (" +
                                          std::to_string(num_simulations_) +
                                          " simulations)";
                  response["checkpoint_path"] = checkpoint_path_;
                  response["num_simulations"] = num_simulations_;
                  response["uct_c"] = uct_c_;
                  response["version"] = "C++ Direct API";
                  response["description"] =
                      "Direct C++ AlphaZero agent without Python wrapper";

                  res.set_content(response.dump(), "application/json");
                });

    // Available agent types endpoint
    server_.Get("/agent/types",
                [](const httplib::Request&, httplib::Response& res) {
                  json response;
                  response["available_types"] = json::array({"alphazero"});
                  response["current_type"] = "alphazero";
                  response["description"] = "C++ Direct AlphaZero API Server";

                  res.set_content(response.dump(), "application/json");
                });

    // Main move endpoint - get best move from AlphaZero
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

        // Get action from AlphaZero
        open_spiel::Action best_action =
            alphazero_agent_->get_action(*state, player_id);

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
            {"agent_type", "alphazero"},
            {"algorithm", "AlphaZero MCTS (" +
                              std::to_string(num_simulations_) +
                              " simulations)"},
            {"nodesEvaluated", alphazero_agent_->get_nodes_explored()},
            {"searchTime", thinking_time}};

        res.set_content(response.dump(), "application/json");

      } catch (const std::exception& e) {
        json error_response;
        error_response["success"] = false;
        error_response["error"] = e.what();
        error_response["agent_type"] = "alphazero";

        res.status = 400;
        res.set_content(error_response.dump(), "application/json");
      }
    });

    // Configuration endpoint
    server_.Post("/agent/configure", [this](const httplib::Request& req,
                                            httplib::Response& res) {
      try {
        json request_json = json::parse(req.body);

        // Update configuration
        if (request_json.contains("num_simulations")) {
          num_simulations_ = request_json["num_simulations"];
        }
        if (request_json.contains("uct_c")) {
          uct_c_ = request_json["uct_c"];
        }

        // Recreate agent with new parameters
        alphazero_agent_ = std::make_unique<AlphaZeroMCTSAgentWrapper>(
            checkpoint_path_, num_simulations_, uct_c_, -1,
            "AlphaZero_API_Server");

        json response;
        response["success"] = true;
        response["message"] = "AlphaZero agent reconfigured";
        response["num_simulations"] = num_simulations_;
        response["uct_c"] = uct_c_;

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

  void start() {
    std::cout << "🌐 Starting Azul C++ API Server on port " << port_
              << std::endl;
    std::cout << "🧠 Agent: AlphaZero (" << num_simulations_
              << " simulations, UCT=" << uct_c_ << ")" << std::endl;
    std::cout << "📁 Checkpoint: " << checkpoint_path_ << std::endl;
    std::cout << "🎯 Available endpoints:" << std::endl;
    std::cout << "  GET  /health - Health check" << std::endl;
    std::cout << "  GET  /agent/info - Agent information" << std::endl;
    std::cout << "  GET  /agent/types - Available agent types" << std::endl;
    std::cout << "  POST /agent/move - Get best move" << std::endl;
    std::cout << "  POST /agent/configure - Configure agent" << std::endl;
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
   * Create OpenSpiel AzulState from webapp JSON gamestate
   */
  std::unique_ptr<open_spiel::State> create_state_from_json(
      const json& game_state_json) {
    // For now, create initial state - full reconstruction requires OpenSpiel
    // modifications to expose private members or add state reconstruction API
    open_spiel::GameParameters params;
    params["num_players"] =
        open_spiel::GameParameter(game_state_json.value("numPlayers", 2));

    auto game = open_spiel::LoadGame("azul", params);
    auto state = game->NewInitialState();

    // Log the incoming gamestate for debugging
    std::cout << "Received gamestate JSON for analysis (using initial state "
                 "for now):\n";
    std::cout << game_state_json.dump(2) << std::endl;

    return state;
  }

  /**
   * Decode OpenSpiel action to understand its components
   */
  struct DecodedAction {
    bool from_center;
    int factory_id;
    open_spiel::azul::TileColor color;
    int destination;  // Pattern line (0-4) or -1 for floor
  };

  DecodedAction decode_action(open_spiel::Action action, int num_players) {
    DecodedAction decoded;

    // Replicate OpenSpiel's action decoding logic
    // Constants from OpenSpiel Azul implementation
    const int kNumTileColors = 5;
    const int kNumPatternLines = 5;

    int max_factories = (2 * num_players) + 1;
    int factory_pattern_actions =
        max_factories * kNumTileColors * kNumPatternLines;
    int center_pattern_actions = kNumTileColors * kNumPatternLines;
    int factory_floor_actions = max_factories * kNumTileColors;

    if (action < factory_pattern_actions) {
      // Factory to pattern line
      decoded.from_center = false;
      decoded.factory_id = action / (kNumTileColors * kNumPatternLines);
      int remainder = action % (kNumTileColors * kNumPatternLines);
      decoded.color = static_cast<open_spiel::azul::TileColor>(
          remainder / kNumPatternLines);
      decoded.destination = remainder % kNumPatternLines;
    } else if (action < factory_pattern_actions + center_pattern_actions) {
      // Center to pattern line
      decoded.from_center = true;
      decoded.factory_id = -1;
      int remainder = action - factory_pattern_actions;
      decoded.color = static_cast<open_spiel::azul::TileColor>(
          remainder / kNumPatternLines);
      decoded.destination = remainder % kNumPatternLines;
    } else if (action < factory_pattern_actions + center_pattern_actions +
                            factory_floor_actions) {
      // Factory to floor
      decoded.from_center = false;
      int remainder = action - factory_pattern_actions - center_pattern_actions;
      decoded.factory_id = remainder / kNumTileColors;
      decoded.color =
          static_cast<open_spiel::azul::TileColor>(remainder % kNumTileColors);
      decoded.destination = -1;
    } else {
      // Center to floor
      decoded.from_center = true;
      decoded.factory_id = -1;
      int remainder = action - factory_pattern_actions -
                      center_pattern_actions - factory_floor_actions;
      decoded.color = static_cast<open_spiel::azul::TileColor>(remainder);
      decoded.destination = -1;
    }

    return decoded;
  }

  /**
   * Convert OpenSpiel action to webapp JSON format
   */
  json action_to_json(open_spiel::Action action,
                      const open_spiel::State& state) {
    // Cast to AzulState to access number of players
    const auto& azul_state =
        static_cast<const open_spiel::azul::AzulState&>(state);

    // Use our own action decoding implementation
    auto decoded = decode_action(action, azul_state.num_players_);

    json action_json;

    // Convert source: center = -1, factories = 0-4
    if (decoded.from_center) {
      action_json["factoryIndex"] = -1;
    } else {
      action_json["factoryIndex"] = decoded.factory_id;
    }

    // Convert tile color to webapp format (integer)
    action_json["tile"] = static_cast<int>(decoded.color);

    // Convert destination: floor = -1, pattern lines = 0-4
    action_json["lineIndex"] = decoded.destination;

    return action_json;
  }
};

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --checkpoint <path>   AlphaZero checkpoint path (required)\n";
  std::cout << "  --port <n>           Server port (default: 5001)\n";
  std::cout
      << "  --simulations <n>    Number of MCTS simulations (default: 800)\n";
  std::cout
      << "  --uct <c>            UCT exploration constant (default: 1.4)\n";
  std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
  std::string checkpoint_path;
  int port = 5001;
  int num_simulations = 800;
  double uct_c = 1.4;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint_path = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (arg == "--simulations" && i + 1 < argc) {
      num_simulations = std::stoi(argv[++i]);
    } else if (arg == "--uct" && i + 1 < argc) {
      uct_c = std::stod(argv[++i]);
    }
  }

  // Validate required parameters
  if (checkpoint_path.empty()) {
    std::cerr << "❌ Error: --checkpoint parameter is required" << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  try {
    // Create and start the server
    AzulApiServer server(checkpoint_path, num_simulations, uct_c, port);

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