#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mcts_agent.h"
#include "open_spiel/spiel.h"

namespace azul {

/**
 * Neural Network MCTS Training Configuration
 */
struct NeuralMCTSConfig {
  // Game settings
  std::string game_name = "azul";
  int num_players = 2;

  // Neural network architecture
  std::string nn_model = "resnet";  // "mlp", "conv2d", "resnet"
  int nn_width = 256;
  int nn_depth = 10;

  // Training parameters
  int max_steps = 10000;
  int actors = 4;
  int evaluators = 2;
  int max_simulations = 400;
  double learning_rate = 0.001;
  int train_batch_size = 32;
  int replay_buffer_size = 100000;
  int checkpoint_freq = 100;

  // MCTS parameters
  double uct_c = 1.4;
  double policy_alpha = 0.3;
  double policy_epsilon = 0.25;
  double temperature = 1.0;
  int temperature_drop = 20;

  // Training optimization
  double weight_decay = 1e-4;
  int replay_buffer_reuse = 10;
  int evaluation_window = 100;
  int eval_levels = 3;

  // Output settings
  std::string checkpoint_dir = "models/neural_mcts_azul";
  bool quiet = false;
  bool use_gpu = true;

  // Device settings
  std::string device = "auto";  // "cpu", "cuda", "mps", "auto"
  int cache_size_mb = 1000;
  int cache_shards = 1;
  int threads = 0;  // 0 = auto-detect
};

/**
 * Neural MCTS Trainer that orchestrates Python AlphaZero training
 * and provides C++ evaluation infrastructure
 */
class NeuralMCTSTrainer {
 public:
  explicit NeuralMCTSTrainer(NeuralMCTSConfig config)
      : config_(std::move(config)) {
    // Load the game
    game_ = open_spiel::LoadGame(config_.game_name);
    if (!game_) {
      throw std::runtime_error("Failed to load game: " + config_.game_name);
    }

    std::cout << "Loaded game: " << game_->GetType().short_name << '\n';
    std::cout << "Observation shape: ";
    auto obs_shape = game_->ObservationTensorShape();
    for (size_t i = 0; i < obs_shape.size(); ++i) {
      std::cout << obs_shape[i];
      if (i < obs_shape.size() - 1) {
        std::cout << "x";
      }
    }
    std::cout << '\n';
    std::cout << "Number of actions: " << game_->NumDistinctActions() << '\n';

    // Create checkpoint directory
    std::filesystem::create_directories(config_.checkpoint_dir);
  }

  /**
   * Start the training process
   */
  void Train() {
    std::cout << "\n=== Neural MCTS Training for Azul ===" << '\n';
    std::cout << "Model: " << config_.nn_model << '\n';
    std::cout << "Network: " << config_.nn_width << "x" << config_.nn_depth
              << '\n';
    std::cout << "Steps: " << config_.max_steps << '\n';
    std::cout << "Actors: " << config_.actors
              << ", Evaluators: " << config_.evaluators << '\n';
    std::cout << "Simulations per move: " << config_.max_simulations << '\n';
    std::cout << "Device: " << config_.device << '\n';
    std::cout << "Checkpoint dir: " << config_.checkpoint_dir << '\n';
    std::cout << '\n';

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
      // Generate training script and run Python AlphaZero
      GenerateTrainingScript();
      RunPythonTraining();

      // Evaluate trained models
      EvaluateTrainedModel();

    } catch (const std::exception& e) {
      std::cerr << "Training failed: " << e.what() << '\n';
      throw;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);

    std::cout << "\nTraining completed in " << duration.count() << " minutes!"
              << '\n';
    std::cout << "Model checkpoints saved in: " << config_.checkpoint_dir
              << '\n';
  }

 private:
  /**
   * Generate Python training script that uses OpenSpiel's AlphaZero
   */
  void GenerateTrainingScript() const {
    std::string script_path = config_.checkpoint_dir + "/run_training.py";
    std::ofstream script(script_path);

    if (!script.is_open()) {
      throw std::runtime_error("Failed to create training script: " +
                               script_path);
    }

    script << "#!/usr/bin/env python3\n";
    script << "# Auto-generated Neural MCTS training script\n";
    script << "import sys\n";
    script << "import os\n";
    script << "# Add the project root directory to Python path\n";
    script << "script_dir = os.path.dirname(os.path.abspath(__file__))\n";
    script << "project_root = os.path.join(script_dir, '../../..')\n";
    script << "sys.path.insert(0, os.path.abspath(project_root))\n\n";

    script << "from training.openspiel_alphazero_training import main\n";
    script << "from absl import app, flags\n\n";

    script << "# Override flags with C++ configuration\n";
    script << "FLAGS = flags.FLAGS\n";
    script << "FLAGS.checkpoint_dir = '" << config_.checkpoint_dir << "'\n";
    script << "FLAGS.max_steps = " << config_.max_steps << "\n";
    script << "FLAGS.actors = " << config_.actors << "\n";
    script << "FLAGS.evaluators = " << config_.evaluators << "\n";
    script << "FLAGS.max_simulations = " << config_.max_simulations << "\n";
    script << "FLAGS.learning_rate = " << config_.learning_rate << "\n";
    script << "FLAGS.train_batch_size = " << config_.train_batch_size << "\n";
    script << "FLAGS.replay_buffer_size = " << config_.replay_buffer_size
           << "\n";
    script << "FLAGS.checkpoint_freq = " << config_.checkpoint_freq << "\n";
    script << "FLAGS.uct_c = " << config_.uct_c << "\n";
    script << "FLAGS.policy_alpha = " << config_.policy_alpha << "\n";
    script << "FLAGS.policy_epsilon = " << config_.policy_epsilon << "\n";
    script << "FLAGS.temperature = " << config_.temperature << "\n";
    script << "FLAGS.temperature_drop = " << config_.temperature_drop << "\n";
    script << "FLAGS.nn_model = '" << config_.nn_model << "'\n";
    script << "FLAGS.nn_width = " << config_.nn_width << "\n";
    script << "FLAGS.nn_depth = " << config_.nn_depth << "\n";
    script << "FLAGS.weight_decay = " << config_.weight_decay << "\n";
    script << "FLAGS.evaluation_window = " << config_.evaluation_window << "\n";
    script << "FLAGS.eval_levels = " << config_.eval_levels << "\n";
    script << "FLAGS.replay_buffer_reuse = " << config_.replay_buffer_reuse
           << "\n";
    script << "FLAGS.quiet = " << (config_.quiet ? "True" : "False") << "\n\n";

    // Add device auto-detection for MPS support
    script << "# Device auto-detection with MPS support\n";
    script << "import torch\n";
    script << "requested_device = '" << config_.device << "'\n";
    script << "if requested_device == 'auto':\n";
    script << "    if torch.backends.mps.is_available():\n";
    script << "        actual_device = 'mps'\n";
    script << "        print(f'[INFO] MPS detected - using Apple Silicon GPU "
              "acceleration')\n";
    script << "    elif torch.cuda.is_available():\n";
    script << "        actual_device = 'cuda'\n";
    script << "        print(f'[INFO] CUDA detected - using NVIDIA GPU "
              "acceleration')\n";
    script << "    else:\n";
    script << "        actual_device = 'cpu'\n";
    script << "        print(f'[INFO] Using CPU (no GPU acceleration "
              "available)')\n";
    script << "else:\n";
    script << "    actual_device = requested_device\n";
    script << "    print(f'[INFO] Using explicitly requested device: "
              "{actual_device}')\n\n";
    script << "# Set environment variable for PyTorch MPS fallback if needed\n";
    script << "if actual_device == 'mps':\n";
    script << "    import os\n";
    script << "    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'\n\n";

    script << "if __name__ == '__main__':\n";
    script << "    app.run(main)\n";

    script.close();

    // Make script executable
    std::filesystem::permissions(script_path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec);

    std::cout << "Generated training script: " << script_path << '\n';
  }

  /**
   * Run the Python AlphaZero training
   */
  void RunPythonTraining() const {
    std::cout << "Starting Python AlphaZero training..." << '\n';

    std::string script_path = config_.checkpoint_dir + "/run_training.py";
    std::string command =
        "cd " + config_.checkpoint_dir + " && python3 run_training.py";

    std::cout << "Running: " << command << '\n';

    int result = std::system(command.c_str());
    if (result != 0) {
      throw std::runtime_error("Python training failed with exit code: " +
                               std::to_string(result));
    }

    std::cout << "Python training completed successfully!" << '\n';
  }

  /**
   * Evaluate the trained model using C++ MCTS agents
   */
  void EvaluateTrainedModel() const {
    std::cout << "Evaluating trained model..." << '\n';

    // Create baseline MCTS agent
    auto baseline_agent =
        create_mcts_agent(0, config_.max_simulations / 4, config_.uct_c);

    // TODO: Create neural network MCTS agent when checkpoint is available
    // For now, we'll just report that evaluation would happen here

    std::cout << "Model evaluation infrastructure ready." << '\n';
    std::cout << "Trained checkpoints available in: " << config_.checkpoint_dir
              << '\n';
    std::cout
        << "Use the Python evaluation tools to compare against MCTS baseline."
        << '\n';
  }

  NeuralMCTSConfig config_;
  std::shared_ptr<const open_spiel::Game> game_;
};

}  // namespace azul

/**
 * Command line argument parsing
 */
void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]" << '\n';
  std::cout << "Options:" << '\n';
  std::cout << "  --steps=N           Maximum training steps (default: 10000)"
            << '\n';
  std::cout << "  --actors=N          Number of actor threads (default: 4)"
            << '\n';
  std::cout << "  --evaluators=N      Number of evaluator threads (default: 2)"
            << '\n';
  std::cout << "  --simulations=N     MCTS simulations per move (default: 400)"
            << '\n';
  std::cout << "  --width=N           NN width (default: 256)" << '\n';
  std::cout << "  --depth=N           NN depth (default: 10)" << '\n';
  std::cout << "  --lr=FLOAT          Learning rate (default: 0.001)" << '\n';
  std::cout << "  --batch=N           Training batch size (default: 32)"
            << '\n';
  std::cout << "  --buffer=N          Replay buffer size (default: 100000)"
            << '\n';
  std::cout << "  --checkpoint_freq=N Checkpoint frequency (default: 100)"
            << '\n';
  std::cout << "  --checkpoint_dir=PATH Checkpoint directory (default: "
               "models/neural_mcts_azul)"
            << '\n';
  std::cout << "  --device=TYPE       Device (cpu|cuda|mps|auto, default: auto)"
            << '\n';
  std::cout << "  --quiet             Suppress training output" << '\n';
  std::cout << "  --help              Show this help message" << '\n';
}

azul::NeuralMCTSConfig ParseArguments(int argc, char** argv) {
  azul::NeuralMCTSConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help") {
      PrintUsage(argv[0]);
      exit(0);
    } else if (arg.find("--steps=") == 0) {
      config.max_steps = std::stoi(arg.substr(8));
    } else if (arg.find("--actors=") == 0) {
      config.actors = std::stoi(arg.substr(9));
    } else if (arg.find("--evaluators=") == 0) {
      config.evaluators = std::stoi(arg.substr(13));
    } else if (arg.find("--simulations=") == 0) {
      config.max_simulations = std::stoi(arg.substr(14));
    } else if (arg.find("--model=") == 0) {
      config.nn_model = arg.substr(8);
    } else if (arg.find("--width=") == 0) {
      config.nn_width = std::stoi(arg.substr(8));
    } else if (arg.find("--depth=") == 0) {
      config.nn_depth = std::stoi(arg.substr(8));
    } else if (arg.find("--lr=") == 0) {
      config.learning_rate = std::stod(arg.substr(5));
    } else if (arg.find("--batch=") == 0) {
      config.train_batch_size = std::stoi(arg.substr(8));
    } else if (arg.find("--buffer=") == 0) {
      config.replay_buffer_size = std::stoi(arg.substr(9));
    } else if (arg.find("--checkpoint_freq=") == 0) {
      config.checkpoint_freq = std::stoi(arg.substr(18));
    } else if (arg.find("--checkpoint_dir=") == 0) {
      config.checkpoint_dir = arg.substr(17);
    } else if (arg.find("--device=") == 0) {
      config.device = arg.substr(9);
      config.use_gpu = (config.device != "cpu");
    } else if (arg == "--quiet") {
      config.quiet = true;
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      PrintUsage(argv[0]);
      exit(1);
    }
  }

  return config;
}

/**
 * Main function
 */
int main(int argc, char** argv) {
  try {
    // Parse command line arguments
    auto config = ParseArguments(argc, argv);

    // Create and run trainer
    azul::NeuralMCTSTrainer trainer(config);
    trainer.Train();

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}