#include <chrono>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// HTTP server library (header-only)
#include <httplib.h>

// JSON library
#include <nlohmann/json.hpp>

// Command line parsing
#include <cxxopts.hpp>

// Logging library
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

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
} // namespace

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
  int minimax_depth_;
  std::shared_ptr<spdlog::logger> logger_;

public:
  AzulApiServer(const std::string &agent_type = "alphazero",
                const std::string &checkpoint_path = "",
                int num_simulations = 2000, double uct_c = 1.4, int port = 5000,
                int seed = -1, int minimax_depth = 4)
      : agent_type_(agent_type), checkpoint_path_(checkpoint_path),
        num_simulations_(num_simulations), uct_c_(uct_c), port_(port),
        seed_(seed), minimax_depth_(minimax_depth) {
    // Initialize logger with rotating file sink
    // 5MB size max and 3 rotated files
    logger_ = spdlog::rotating_logger_mt("azul_api", "azul_api.log",
                                         5 * 1024 * 1024, 3);
    logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    logger_->set_level(spdlog::level::debug);

    // Force registration
    force_azul_registration();

    // Initialize the selected agent
    initialize_agent();
    setup_routes();
  }

private:
  void initialize_agent() {
    logger_->info("🚀 Initializing {} agent...", agent_type_);

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
      current_agent_ = azul::create_minimax_evaluation_agent(
          minimax_depth_, "Minimax_API_Server");

    } else {
      throw std::runtime_error(
          "Unknown agent type: " + agent_type_ +
          ". Supported types: alphazero, random, mcts, minimax");
    }

    logger_->info("✅ {} initialized", current_agent_->get_name());
  }

  void setup_routes() {
    // Enable CORS for webapp integration
    server_.set_pre_routing_handler([](const httplib::Request &req,
                                       httplib::Response &res) {
      // Only allow requests from the webapp origin
      res.set_header("Access-Control-Allow-Origin", "http://localhost:3000");
      res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.set_header("Access-Control-Allow-Headers",
                     "Content-Type, Authorization");
      return httplib::Server::HandlerResponse::Unhandled;
    });

    // Handle preflight OPTIONS requests
    server_.Options(
        ".*", [](const httplib::Request &, httplib::Response &res) { return; });

    // Simple health check endpoint
    server_.Get("/health",
                [this](const httplib::Request &, httplib::Response &res) {
                  json response;
                  response["status"] = "healthy";
                  response["agentType"] = agent_type_;
                  response["agentName"] = current_agent_->get_name();
                  res.set_content(response.dump(), "application/json");
                });

    // Main move endpoint - get best move from current agent
    server_.Post("/agent/move", [this](const httplib::Request &req,
                                       httplib::Response &res) {
      try {
        // Parse JSON request
        json request_json = json::parse(req.body);

        // Extract parameters
        auto game_state_json = request_json["gameState"];
        int player_id = request_json.value("playerId", 0);

        // Debug log the incoming game state
        logger_->debug("📥 Received game state:");
        logger_->debug("  Current player: {}",
                       game_state_json.value("currentPlayer", -1));
        logger_->debug("  Round number: {}",
                       game_state_json.value("roundNumber", -1));
        logger_->debug("  Game ended: {}",
                       game_state_json.value("gameEnded", false));
        logger_->debug(
            "  First player tile available: {}",
            game_state_json.value("firstPlayerTileAvailable", false));
        logger_->debug("  First player next round: {}",
                       game_state_json.value("firstPlayerNextRound", -1));
        logger_->debug("  Needs bag shuffle: {}",
                       game_state_json.value("needsBagShuffle", false));

        // Log factories
        if (game_state_json.contains("factories") &&
            game_state_json["factories"].is_array()) {
          logger_->debug("  Factories:");
          for (const auto &factory : game_state_json["factories"]) {
            std::string factory_str = "    Factory: ";
            for (const auto &[color, count] : factory.items()) {
              factory_str += fmt::format("{}={} ", color, count.get<int>());
            }
            logger_->debug(factory_str);
          }
        }

        // Log center pile
        if (game_state_json.contains("centerPile") &&
            game_state_json["centerPile"].is_object()) {
          std::string center_str = "  Center pile: ";
          for (const auto &[color, count] :
               game_state_json["centerPile"].items()) {
            center_str += fmt::format("{}={} ", color, count.get<int>());
          }
          logger_->debug(center_str);
        }

        // Log player boards
        if (game_state_json.contains("players") &&
            game_state_json["players"].is_array()) {
          logger_->debug("  Player boards:");
          for (size_t i = 0; i < game_state_json["players"].size(); ++i) {
            const auto &player = game_state_json["players"][i];
            logger_->debug("    Player {}:", i);
            logger_->debug("      Score: {}", player.value("score", 0));

            // Log pattern lines
            if (player.contains("patternLines") &&
                player["patternLines"].is_array()) {
              logger_->debug("      Pattern lines:");
              for (const auto &line : player["patternLines"]) {
                std::string line_str = fmt::format("        Line: count={}",
                                                   line.value("count", 0));
                if (line.contains("color") && line["color"].is_string()) {
                  line_str += fmt::format(", color={}",
                                          line["color"].get<std::string>());
                }
                logger_->debug(line_str);
              }
            }

            // Log wall
            if (player.contains("wall") && player["wall"].is_array()) {
              logger_->debug("      Wall:");
              for (const auto &row : player["wall"]) {
                std::string row_str = "        Row: ";
                for (const auto &cell : row) {
                  row_str += cell.get<bool>() ? "1 " : "0 ";
                }
                logger_->debug(row_str);
              }
            }

            // Log floor line
            if (player.contains("floorLine") &&
                player["floorLine"].is_array()) {
              std::string floor_str = "      Floor line: ";
              for (const auto &tile : player["floorLine"]) {
                floor_str += tile.get<std::string>() + " ";
              }
              logger_->debug(floor_str);
            }
          }
        }

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

        // Log detailed move information
        logger_->info("🤖 Selected move:");
        logger_->info("  Factory index: {}",
                      action_json["factoryIndex"].get<int>());
        logger_->info("  Tile color: {}",
                      action_json["tile"].get<std::string>());
        logger_->info("  Line index: {}", action_json["lineIndex"].get<int>());
        logger_->info("  Raw action value: {}", best_action);

        // Decode and log the raw action value
        auto decoded = state->DecodeAction(best_action);
        logger_->info("  Decoded action:");
        logger_->info("    From center: {}",
                      decoded.from_center ? "yes" : "no");
        logger_->info("    Factory ID: {}", decoded.factory_id);
        logger_->info("    Color: {}", static_cast<int>(decoded.color));
        logger_->info("    Destination: {}", decoded.destination);

        // Prepare response
        json response;
        response["success"] = true;
        response["move"] = action_json;
        response["stats"] = {
            {"agentType", agent_type_},
            {"agentName", current_agent_->get_name()},
            {"nodesEvaluated", current_agent_->get_nodes_explored()},
            {"searchTime", thinking_time}};

        res.set_content(response.dump(), "application/json");

      } catch (const std::exception &e) {
        logger_->error("❌ Error processing move request: {}", e.what());
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
    logger_->info("🌐 Starting Azul C++ API Server on port {}", port_);
    logger_->info("🧠 Agent: {} ({}, {} simulations, UCT={})",
                  current_agent_->get_name(), agent_type_, num_simulations_,
                  uct_c_);
    if (!checkpoint_path_.empty()) {
      logger_->info("📁 Checkpoint: {}", checkpoint_path_);
    }
    logger_->info("🎯 Available endpoints:");
    logger_->info("  GET  /health - Health check");
    logger_->info("  POST /agent/move - Get best move");
    logger_->info("✅ Server ready!");

    server_.listen("0.0.0.0", port_);
  }

  void stop() { server_.stop(); }

private:
  /**
   * Convert webapp tile color string to OpenSpiel TileColor enum
   */
  static auto string_to_tile_color(const std::string &color_str)
      -> open_spiel::azul::TileColor {
    // Handle single-letter color codes
    if (color_str == "B" || color_str == "Blue" || color_str == "blue")
      return open_spiel::azul::TileColor::kBlue;
    if (color_str == "Y" || color_str == "Yellow" || color_str == "yellow")
      return open_spiel::azul::TileColor::kYellow;
    if (color_str == "R" || color_str == "Red" || color_str == "red")
      return open_spiel::azul::TileColor::kRed;
    if (color_str == "K" || color_str == "Black" || color_str == "black")
      return open_spiel::azul::TileColor::kBlack;
    if (color_str == "W" || color_str == "White" || color_str == "white")
      return open_spiel::azul::TileColor::kWhite;
    if (color_str == "F" || color_str == "FirstPlayer" ||
        color_str == "firstPlayer")
      return open_spiel::azul::TileColor::kFirstPlayer;
    throw std::runtime_error("Unknown tile color: " + color_str);
  }

  /**
   * Convert OpenSpiel TileColor to webapp string
   */
  static auto tile_color_to_string(open_spiel::azul::TileColor color)
      -> std::string {
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
    case open_spiel::azul::TileColor::kFirstPlayer:
      return "F";
    default:
      throw std::runtime_error("Unknown tile color enum");
    }
  }

  /**
   * Parse tiles array from JSON and convert to vector of TileColor
   */
  static auto parse_tiles_array(const json &tiles_json)
      -> std::vector<open_spiel::azul::TileColor> {
    std::vector<open_spiel::azul::TileColor> tiles;
    if (tiles_json.is_array()) {
      for (const auto &tile_str : tiles_json) {
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
  static auto parse_factory(const json &factory_json)
      -> open_spiel::azul::Factory {
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
  static auto parse_player_board(const json &player_json)
      -> open_spiel::azul::PlayerBoard {
    open_spiel::azul::PlayerBoard board;

    // Parse score
    if (player_json.contains("score") &&
        player_json["score"].is_number_integer()) {
      board.score = player_json["score"];
    }

    // Parse pattern lines
    if (player_json.contains("patternLines") &&
        player_json["patternLines"].is_array()) {
      const auto &pattern_lines_json = player_json["patternLines"];
      for (size_t i = 0; i < pattern_lines_json.size() &&
                         i < open_spiel::azul::kNumPatternLines;
           ++i) {
        const auto &line_json = pattern_lines_json[i];
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
      const auto &wall_json = player_json["wall"];
      for (size_t row = 0;
           row < wall_json.size() && row < open_spiel::azul::kWallSize; ++row) {
        if (wall_json[row].is_array()) {
          const auto &wall_row = wall_json[row];
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
  static auto create_state_from_json(const json &game_state_json)
      -> std::unique_ptr<open_spiel::azul::AzulState> {
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
        dynamic_cast<open_spiel::azul::AzulState *>(
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
      const auto &factories_json = game_state_json["factories"];
      state->factories_.clear();
      for (const auto &factory_json : factories_json) {
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
      const auto &players_json = game_state_json["players"];
      state->player_boards_.clear();
      for (const auto &player_json : players_json) {
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
  static auto action_to_json(open_spiel::Action action,
                             const open_spiel::azul::AzulState &state) -> json {
    json action_json;

    try {
      // Use the proper DecodeAction method from AzulState
      auto decoded = state.DecodeAction(action);

      // Map to webapp JSON format
      if (decoded.from_center) {
        // Center pile is always represented as -1 in the webapp
        action_json["factoryIndex"] = -1;
      } else {
        action_json["factoryIndex"] = decoded.factory_id;
      }

      // Convert tile color to webapp string format
      action_json["tile"] = tile_color_to_string(decoded.color);
      // Line index is already 0-based in the C++ server
      action_json["lineIndex"] = decoded.destination;

    } catch (const std::exception &e) {
      // Fallback if decoding fails
      action_json["factoryIndex"] = 0;
      action_json["tile"] = "red";
      action_json["lineIndex"] = 0;
      action_json["error"] = "Action decoding failed: " + std::string(e.what());
    }

    return action_json;
  }
};

auto main(int argc, char *argv[]) -> int {
  try {
    cxxopts::Options options("azul_api_server", "Azul C++ API Server");
    options.add_options()(
        "a,agent", "Agent type (alphazero, mcts, random, minimax)",
        cxxopts::value<std::string>()->default_value("alphazero"))(
        "c,checkpoint", "AlphaZero checkpoint path",
        cxxopts::value<std::string>())(
        "p,port", "Server port", cxxopts::value<int>()->default_value("5001"))(
        "s,simulations", "Number of MCTS simulations",
        cxxopts::value<int>()->default_value("1600"))(
        "u,uct", "UCT exploration constant",
        cxxopts::value<double>()->default_value("1.4"))(
        "seed", "Random seed", cxxopts::value<int>()->default_value("-1"))(
        "d,depth", "Minimax search depth",
        cxxopts::value<int>()->default_value("4"))("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") != 0U) {
      std::cout << options.help() << '\n';
      return 0;
    }

    std::string agent_type = result["agent"].as<std::string>();
    std::string checkpoint_path = (result.count("checkpoint") != 0U)
                                      ? result["checkpoint"].as<std::string>()
                                      : "";
    int port = result["port"].as<int>();
    int num_simulations = result["simulations"].as<int>();
    double uct_c = result["uct"].as<double>();
    int seed = result["seed"].as<int>();
    int minimax_depth = result["depth"].as<int>();

    // Validate required parameters
    if (agent_type == "alphazero" && checkpoint_path.empty()) {
      std::cerr
          << "❌ Error: --checkpoint parameter is required for alphazero agent"
          << '\n';
      std::cout << options.help() << '\n';
      return 1;
    }

    // Create and start the server
    AzulApiServer server(agent_type, checkpoint_path, num_simulations, uct_c,
                         port, seed, minimax_depth);

    // Handle Ctrl+C gracefully
    std::signal(SIGINT, [](int) {
      std::cout << "\n🛑 Shutting down server..." << '\n';
      std::exit(0);
    });

    server.start();

  } catch (const cxxopts::exceptions::exception &e) {
    std::cerr << "❌ Error parsing options: " << e.what() << '\n';
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "❌ Server error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}