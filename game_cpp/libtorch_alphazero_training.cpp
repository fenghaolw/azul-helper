#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/thread.h"

namespace azul {

/**
 * LibTorch AlphaZero Training Configuration
 */
struct LibTorchAZConfig {
  // Game settings
  std::string game_name = "azul";

  // Training parameters
  int max_steps = 1000;
  int actors = 4;
  int evaluators = 1;
  int max_simulations = 400;

  // Neural network architecture
  std::string nn_model = "resnet";  // "mlp", "conv2d", "resnet"
  int nn_width = 128;
  int nn_depth = 6;

  // Learning parameters
  double learning_rate = 0.001;
  double weight_decay = 1e-4;
  int train_batch_size = 32;
  // This should ideally match the number of actors to avoid contention.
  int inference_batch_size = 4;
  int inference_threads = 1;
  // This is crucial for performance, since a full NN inference is costly and
  // MCTS generates millions of states to be evaluated. A small cache size will
  // significantly slow down training.
  int inference_cache = 2000000;
  int replay_buffer_size = 100000;
  int replay_buffer_reuse = 10;
  int checkpoint_freq = 100;
  int evaluation_window = 100;

  // MCTS parameters
  double uct_c = 1.4;
  double policy_alpha = 0.3;
  double policy_epsilon = 0.25;
  double temperature = 1.0;
  double temperature_drop = 20.0;
  double cutoff_probability = 0.0;
  double cutoff_value = 0.0;

  // Evaluation parameters
  int eval_levels = 3;

  // Output settings
  std::string checkpoint_dir = "models/libtorch_alphazero_azul";
  std::string device = "cpu";  // "cpu", "cuda", "mps"
  bool explicit_learning = true;
  bool resume_from_checkpoint =
      false;  // Whether to resume from existing checkpoint
};

/**
 * LibTorch AlphaZero Trainer - uses pure C++ implementation
 */
class LibTorchAZTrainer {
 public:
  explicit LibTorchAZTrainer(LibTorchAZConfig config)
      : config_(std::move(config)) {
    // Load the game - assume Azul is already registered
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
   * Start LibTorch AlphaZero training
   */
  void Train() {
    std::cout << "\n=== LibTorch AlphaZero Training for Azul ===\n";
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
      // Create AlphaZero configuration
      auto az_config = CreateAlphaZeroConfig();

      // Create stop token for graceful shutdown
      open_spiel::StopToken stop_token;

      // Start training (resuming if specified)
      bool success = open_spiel::algorithms::torch_az::AlphaZero(
          az_config, &stop_token, config_.resume_from_checkpoint);

      if (!success) {
        throw std::runtime_error("LibTorch AlphaZero training failed");
      }

    } catch (const std::exception& e) {
      std::cerr << "Training failed: " << e.what() << '\n';
      throw;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);

    std::cout << "\nTraining completed in " << duration.count()
              << " minutes!\n";
    std::cout << "Model checkpoints saved in: " << config_.checkpoint_dir
              << '\n';
  }

 private:
  /**
   * Create OpenSpiel AlphaZero configuration from our config
   */
  [[nodiscard]] auto CreateAlphaZeroConfig() const
      -> open_spiel::algorithms::torch_az::AlphaZeroConfig {
    open_spiel::algorithms::torch_az::AlphaZeroConfig az_config;

    // Game settings
    az_config.game = config_.game_name;
    az_config.path = config_.checkpoint_dir;

    // Neural network settings
    az_config.nn_model = config_.nn_model;
    az_config.nn_width = config_.nn_width;
    az_config.nn_depth = config_.nn_depth;
    az_config.devices = config_.device;

    // Training settings
    az_config.explicit_learning = config_.explicit_learning;
    az_config.learning_rate = config_.learning_rate;
    az_config.weight_decay = config_.weight_decay;
    az_config.train_batch_size = config_.train_batch_size;
    az_config.inference_batch_size = config_.inference_batch_size;
    az_config.inference_threads = config_.inference_threads;
    az_config.inference_cache = config_.inference_cache;
    az_config.replay_buffer_size = config_.replay_buffer_size;
    az_config.replay_buffer_reuse = config_.replay_buffer_reuse;
    az_config.checkpoint_freq = config_.checkpoint_freq;
    az_config.evaluation_window = config_.evaluation_window;

    // MCTS settings
    az_config.uct_c = config_.uct_c;
    az_config.max_simulations = config_.max_simulations;
    az_config.policy_alpha = config_.policy_alpha;
    az_config.policy_epsilon = config_.policy_epsilon;
    az_config.temperature = config_.temperature;
    az_config.temperature_drop = config_.temperature_drop;
    az_config.cutoff_probability = config_.cutoff_probability;
    az_config.cutoff_value = config_.cutoff_value;

    // Training control
    az_config.actors = config_.actors;
    az_config.evaluators = config_.evaluators;
    az_config.eval_levels = config_.eval_levels;
    az_config.max_steps = config_.max_steps;

    return az_config;
  }

  LibTorchAZConfig config_;
  std::shared_ptr<const open_spiel::Game> game_;
};

}  // namespace azul

/**
 * Command line argument parsing
 */
void PrintUsage(const char* program_name) {
  std::cout << "LibTorch AlphaZero Training for Azul" << '\n';
  std::cout << '\n';
  std::cout << "Usage: " << program_name << " [options]" << '\n';
  std::cout << '\n';
  std::cout << "Options:" << '\n';
  std::cout << "  --steps=N           Training steps (default: 1000)" << '\n';
  std::cout << "  --actors=N          Actor threads (default: 2)" << '\n';
  std::cout << "  --evaluators=N      Evaluator threads (default: 1)" << '\n';
  std::cout << "  --simulations=N     MCTS simulations per move (default: 400)"
            << '\n';
  std::cout
      << "  --model=TYPE        NN model: mlp|conv2d|resnet (default: resnet)"
      << '\n';
  std::cout << "  --width=N           NN width (default: 128)" << '\n';
  std::cout << "  --depth=N           NN depth (default: 6)" << '\n';
  std::cout << "  --lr=F              Learning rate (default: 0.001)" << '\n';
  std::cout << "  --batch=N           Batch size (default: 32)" << '\n';
  std::cout << "  --device=TYPE       Device: cpu|cuda|mps (default: cpu)"
            << '\n';
  std::cout << "  --dir=PATH          Checkpoint directory (default: "
               "models/libtorch_alphazero_azul)"
            << '\n';
  std::cout << "  --no-explicit-learning  Disable explicit learning (default: "
               "enabled)"
            << '\n';
  std::cout
      << "  --resume              Resume from existing checkpoint (default: "
         "not resuming)"
      << '\n';
  std::cout << "  --help              Show this help" << '\n';
  std::cout << '\n';
  std::cout << "Examples:" << '\n';
  std::cout << "  " << program_name
            << "                    # Quick training with defaults" << '\n';
  std::cout << "  " << program_name
            << " --steps=5000 --actors=4  # Longer training" << '\n';
  std::cout << "  " << program_name
            << " --device=mps        # Use Apple Silicon GPU" << '\n';
  std::cout << "  " << program_name
            << " --resume            # Resume from existing checkpoint" << '\n';
  std::cout << "  " << program_name
            << " --resume --steps=2000 # Resume and train for more steps"
            << '\n';
}

auto ParseArguments(int argc, char** argv) -> azul::LibTorchAZConfig {
  azul::LibTorchAZConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help") {
      PrintUsage(argv[0]);
      exit(0);
    } else if (arg.substr(0, 8) == "--steps=") {
      config.max_steps = std::stoi(arg.substr(8));
    } else if (arg.substr(0, 9) == "--actors=") {
      config.actors = std::stoi(arg.substr(9));
    } else if (arg.substr(0, 13) == "--evaluators=") {
      config.evaluators = std::stoi(arg.substr(13));
    } else if (arg.substr(0, 14) == "--simulations=") {
      config.max_simulations = std::stoi(arg.substr(14));
    } else if (arg.substr(0, 8) == "--model=") {
      config.nn_model = arg.substr(8);
    } else if (arg.substr(0, 8) == "--width=") {
      config.nn_width = std::stoi(arg.substr(8));
    } else if (arg.substr(0, 8) == "--depth=") {
      config.nn_depth = std::stoi(arg.substr(8));
    } else if (arg.substr(0, 5) == "--lr=") {
      config.learning_rate = std::stod(arg.substr(5));
    } else if (arg.substr(0, 8) == "--batch=") {
      config.train_batch_size = std::stoi(arg.substr(8));
    } else if (arg.substr(0, 9) == "--device=") {
      config.device = arg.substr(9);
    } else if (arg.substr(0, 6) == "--dir=") {
      config.checkpoint_dir = arg.substr(6);
    } else if (arg == "--no-explicit-learning") {
      config.explicit_learning = false;
    } else if (arg == "--resume") {
      config.resume_from_checkpoint = true;
    } else {
      std::cerr << "Unknown option: " << arg << '\n';
      PrintUsage(argv[0]);
      exit(1);
    }
  }

  return config;
}

int main(int argc, char** argv) {
  try {
    auto config = ParseArguments(argc, argv);
    azul::LibTorchAZTrainer trainer(config);
    trainer.Train();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}