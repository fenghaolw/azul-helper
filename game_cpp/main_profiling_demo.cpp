#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef PROFILING_ENABLED
#include <gperftools/profiler.h>
#endif

#include "agent_evaluator.h"
#include "agent_profiler.h"
#include "evaluation_config.h"

// Simple command line argument parsing
struct ProfilingConfig {
  std::string agent_type = "mcts";  // "mcts", "minimax", or "alphazero"
  int depth = 4;                    // for minimax
  int simulations = 1000;           // for mcts
  double uct_c = 1.4;               // for mcts
  int num_games = 5;                // number of profiling games
  int seed = 42;                    // random seed
  bool verbose = true;              // verbose output
  std::string checkpoint_path = ""; // for alphazero
};

namespace {
void force_azul_registration() {
  // Reference symbols from azul namespace to force linking
  (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  --agent <type>       Agent type: 'mcts', 'minimax', or "
               "'alphazero' (default: "
               "mcts)\n";
  std::cout << "  --depth <n>          Minimax depth (default: 4)\n";
  std::cout << "  --simulations <n>    MCTS simulations (default: 1000)\n";
  std::cout << "  --uct <c>            MCTS UCT constant (default: 1.4)\n";
  std::cout
      << "  --games <n>          Number of profiling games (default: 5)\n";
  std::cout << "  --seed <n>           Random seed (default: 42)\n";
  std::cout << "  --checkpoint <path>  AlphaZero checkpoint path (required for "
               "alphazero)\n";
  std::cout << "  --quiet              Reduce output verbosity\n";
  std::cout << "  --help               Show this help message\n";
}

ProfilingConfig parse_arguments(int argc, char *argv[]) {
  ProfilingConfig config;
  config.verbose = true;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      exit(0);
    } else if (arg == "--agent" && i + 1 < argc) {
      config.agent_type = argv[++i];
      if (config.agent_type != "mcts" && config.agent_type != "minimax" &&
          config.agent_type != "alphazero") {
        std::cerr << "Error: Invalid agent type. Use 'mcts', 'minimax', or "
                     "'alphazero'\n";
        exit(1);
      }
    } else if (arg == "--depth" && i + 1 < argc) {
      config.depth = std::stoi(argv[++i]);
      if (config.depth < 1 || config.depth > 10) {
        std::cerr << "Error: Depth must be between 1 and 10\n";
        exit(1);
      }
    } else if (arg == "--simulations" && i + 1 < argc) {
      config.simulations = std::stoi(argv[++i]);
      if (config.simulations < 10 || config.simulations > 10000) {
        std::cerr << "Error: Simulations must be between 10 and 10000\n";
        exit(1);
      }
    } else if (arg == "--uct" && i + 1 < argc) {
      config.uct_c = std::stod(argv[++i]);
      if (config.uct_c < 0.1 || config.uct_c > 5.0) {
        std::cerr << "Error: UCT constant must be between 0.1 and 5.0\n";
        exit(1);
      }
    } else if (arg == "--games" && i + 1 < argc) {
      config.num_games = std::stoi(argv[++i]);
      if (config.num_games < 1 || config.num_games > 100) {
        std::cerr << "Error: Number of games must be between 1 and 100\n";
        exit(1);
      }
    } else if (arg == "--seed" && i + 1 < argc) {
      config.seed = std::stoi(argv[++i]);
    } else if (arg == "--checkpoint" && i + 1 < argc) {
      config.checkpoint_path = argv[++i];
    } else if (arg == "--quiet") {
      config.verbose = false;
    } else {
      std::cerr << "Error: Unknown argument " << arg << "\n";
      print_usage(argv[0]);
      exit(1);
    }
  }

  // Validate alphazero requirements
  if (config.agent_type == "alphazero" && config.checkpoint_path.empty()) {
    std::cerr << "Error: --checkpoint path is required for alphazero agent\n";
    exit(1);
  }

  return config;
}
} // namespace

auto main(int argc, char *argv[]) -> int {
  auto config = parse_arguments(argc, argv);

#ifdef PROFILING_ENABLED
  // Start CPU profiling
  std::string profile_file = "azul_profile.prof";
  ProfilerStart(profile_file.c_str());
#endif

  std::cout << "=== AZUL AGENT PROFILING DEMO ===" << '\n';
  std::cout << "Agent: " << config.agent_type;
  if (config.agent_type == "minimax") {
    std::cout << " (depth: " << config.depth << ")";
  } else if (config.agent_type == "alphazero") {
    std::cout << " (checkpoint: " << config.checkpoint_path << ")";
  } else {
    std::cout << " (simulations: " << config.simulations
              << ", UCT: " << config.uct_c << ")";
  }
  std::cout << "\nGames: " << config.num_games << " vs Random";
  std::cout << "\nSeed: " << config.seed << '\n';
  std::cout << '\n';

  try {
    // Force registration
    force_azul_registration();

    // Verify Azul game loads
    auto game = open_spiel::LoadGame("azul");
    if (!game) {
      std::cerr << "❌ Failed to load Azul game" << '\n';
      return 1;
    }
    if (config.verbose) {
      std::cout << "✅ Azul game loaded successfully" << '\n';
    }

    // Enable profiling
    auto &profiler = azul::AgentProfiler::instance();
    profiler.start_profiling();

    // Create profiled agent based on configuration
    std::unique_ptr<azul::ProfiledMinimaxAgent> profiled_minimax;
    std::unique_ptr<azul::ProfiledMCTSAgent> profiled_mcts;
    std::unique_ptr<azul::ProfiledAlphaZeroMCTSAgent> profiled_alphazero;

    if (config.agent_type == "minimax") {
      profiled_minimax = azul::create_profiled_minimax_agent(0, config.depth);
      if (config.verbose) {
        std::cout << "Created profiled Minimax agent (depth " << config.depth
                  << ")" << '\n';
      }
    } else if (config.agent_type == "alphazero") {
      try {
        profiled_alphazero = azul::create_profiled_alphazero_agent(
            config.checkpoint_path, config.simulations, config.uct_c,
            config.seed);
        if (config.verbose) {
          std::cout << "Created profiled AlphaZero agent (checkpoint: "
                    << config.checkpoint_path << ", " << config.simulations
                    << " sims, UCT " << config.uct_c << ")" << '\n';
        }
      } catch (const std::exception &e) {
        std::cerr << "❌ Failed to load AlphaZero model: " << e.what() << '\n';
        return 1;
      }
    } else {
      profiled_mcts = azul::create_profiled_mcts_agent(
          0, config.simulations, config.uct_c, config.seed);
      if (config.verbose) {
        std::cout << "Created profiled MCTS agent (" << config.simulations
                  << " sims, UCT " << config.uct_c << ")" << '\n';
      }
    }

    // Run profiling games against random agent
    if (config.verbose) {
      std::cout << "Running " << config.num_games
                << " profiling games against random agent..." << '\n';
    }

    auto test_state = game->NewInitialState();
    for (int game_num = 0; game_num < config.num_games; ++game_num) {
      auto state = test_state->Clone();
      int moves = 0;

      if (config.verbose &&
          (game_num % std::max(1, config.num_games / 5) == 0)) {
        std::cout << "  Game " << (game_num + 1) << "/" << config.num_games
                  << "..." << '\n';
      }

      while (!state->IsTerminal() &&
             moves < 50) { // Limit moves to prevent long games
        auto current_player = state->CurrentPlayer();

        // Handle chance events
        if (current_player == -1) {
          auto chance_outcomes = state->ChanceOutcomes();
          if (!chance_outcomes.empty()) {
            state->ApplyAction(chance_outcomes[0].first);
            continue;
          }
        }

        open_spiel::Action action;
        if (current_player == 0) {
          // Profiled agent
          if (config.agent_type == "minimax") {
            action = profiled_minimax->get_action(*state);
          } else if (config.agent_type == "alphazero") {
            action = profiled_alphazero->get_action(*state);
          } else {
            action = profiled_mcts->get_action(*state);
          }
        } else {
          // Random agent for fast profiling
          auto legal_actions = state->LegalActions();
          if (!legal_actions.empty()) {
            std::mt19937 rng(config.seed + moves);
            std::uniform_int_distribution<size_t> dist(0, legal_actions.size() -
                                                              1);
            action = legal_actions[dist(rng)];
          } else {
            break;
          }
        }

        state->ApplyAction(action);
        moves++;
      }
    }

#ifdef PROFILING_ENABLED
    // Stop CPU profiling
    ProfilerStop();

    // Print profiling results
    std::cout << "\n=== PROFILING RESULTS ===" << '\n';
    std::cout << "CPU profile saved to: " << profile_file << '\n';
    std::cout << "To analyze the profile, run: pprof --text " << profile_file
              << '\n';
    std::cout << "For a graphical view, run: pprof --pdf " << profile_file
              << " > profile.pdf" << '\n';
#else
    std::cout << "\n=== PROFILING RESULTS ===" << '\n';
    std::cout << "CPU profiling is disabled. Install gperftools to enable "
                 "detailed profiling."
              << '\n';
    std::cout << "On macOS: brew install gperftools" << '\n';
    std::cout << "On Ubuntu/Debian: sudo apt-get install google-perftools "
                 "libgoogle-perftools-dev"
              << '\n';
#endif

    profiler.stop_profiling();

    std::cout << "\n✅ PROFILING DEMO COMPLETED SUCCESSFULLY!" << '\n';

  } catch (const std::exception &e) {
#ifdef PROFILING_ENABLED
    ProfilerStop(); // Make sure to stop profiling even if there's an error
#endif
    std::cerr << "❌ Error during profiling: " << e.what() << '\n';
    return 1;
  }

  return 0;
}