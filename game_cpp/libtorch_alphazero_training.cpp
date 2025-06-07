#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/thread.h"

namespace azul {

/**
 * Throughput Monitor - parses actor logs and reports training throughput
 */
class ThroughputMonitor {
 public:
  explicit ThroughputMonitor(std::string log_directory)
      : log_directory_(std::move(log_directory)),
        stop_monitoring_(false),
        first_report_(true) {}

  ~ThroughputMonitor() { Stop(); }

  void Start() {
    stop_monitoring_ = false;
    monitor_thread_ = std::thread(&ThroughputMonitor::MonitorLoop, this);
  }

  void Stop() {
    stop_monitoring_ = true;
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();
    }
  }

 private:
  void MonitorLoop() {
    std::cout << "\n=== Throughput Monitor Started ===\n";
    std::cout << "Monitoring directory: " << log_directory_ << "\n";
    std::cout << "Will report throughput every 60 seconds...\n\n";

    while (!stop_monitoring_) {
      // Sleep for 60 seconds, but check for stop signal every second
      for (int i = 0; i < 60 && !stop_monitoring_; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      if (!stop_monitoring_) {
        ReportThroughput();
      }
    }
  }

  void ReportThroughput() {
    auto completion_times = ParseActorLogs();

    // Clear previous report if not the first one
    if (!first_report_) {
      // Move cursor up by the number of lines we printed last time
      for (int i = 0; i < lines_printed_; ++i) {
        std::cout << "\033[A";  // Move cursor up one line
      }
      // Move to beginning of line and clear from cursor to end of screen
      std::cout << "\r\033[J";
    }

    lines_printed_ = 0;  // Reset line counter

    if (completion_times.size() < 2) {
      std::cout << "[Throughput] Not enough data to calculate throughput. "
                   "Games completed so far: "
                << completion_times.size();
      std::cout.flush();
      lines_printed_ = 1;
      first_report_ = false;
      return;
    }

    // Calculate duration from first to last game
    auto duration = completion_times.back() - completion_times.front();
    auto total_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    if (total_seconds == 0) {
      std::cout << "[Throughput] All games finished in the same second. Cannot "
                   "calculate rate.";
      std::cout.flush();
      lines_printed_ = 1;
      first_report_ = false;
      return;
    }

    double total_hours = total_seconds / 3600.0;
    double games_per_hour = completion_times.size() / total_hours;

    // Format timestamps for display
    auto first_time = std::chrono::system_clock::from_time_t(
        std::chrono::system_clock::to_time_t(completion_times.front()));
    auto last_time = std::chrono::system_clock::from_time_t(
        std::chrono::system_clock::to_time_t(completion_times.back()));

    std::time_t first_tt = std::chrono::system_clock::to_time_t(first_time);
    std::time_t last_tt = std::chrono::system_clock::to_time_t(last_time);

    // Print the throughput report (tracking line count)
    std::cout << "--- Training Throughput Report ---\n";
    lines_printed_++;
    std::cout << "  Total Games Completed: " << completion_times.size() << "\n";
    lines_printed_++;
    std::cout << "  Time of First Game: "
              << std::put_time(std::localtime(&first_tt), "%Y-%m-%d %H:%M:%S")
              << "\n";
    lines_printed_++;
    std::cout << "  Time of Last Game:  "
              << std::put_time(std::localtime(&last_tt), "%Y-%m-%d %H:%M:%S")
              << "\n";
    lines_printed_++;
    std::cout << "  Duration Analyzed: " << FormatDuration(duration) << "\n";
    lines_printed_++;
    std::cout << "------------------------------------\n";
    lines_printed_++;
    std::cout << "  Current Throughput: " << std::fixed << std::setprecision(2)
              << games_per_hour << " games per hour\n";
    lines_printed_++;
    std::cout << "------------------------------------";
    lines_printed_++;

    std::cout.flush();  // Ensure output is displayed immediately
    first_report_ = false;
  }

  auto ParseActorLogs() -> std::vector<std::chrono::system_clock::time_point> {
    std::vector<std::chrono::system_clock::time_point> completion_times;

    // Find all actor log files
    std::vector<std::string> log_files;
    try {
      for (const auto& entry :
           std::filesystem::directory_iterator(log_directory_)) {
        if (entry.is_regular_file()) {
          std::string filename = entry.path().filename().string();
          if (filename.find("log-actor-") == 0 && filename.length() >= 4 &&
              filename.substr(filename.length() - 4) == ".txt") {
            log_files.push_back(entry.path().string());
          }
        }
      }
    } catch (const std::exception& e) {
      // Directory might not exist yet or be accessible
      return completion_times;
    }

    if (log_files.empty()) {
      return completion_times;
    }

    // Regex to match game completion lines: [YYYY-MM-DD HH:MM:SS.mmm] Game X:
    // Returns:
    std::regex game_line_regex(
        R"(\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] Game \d+: Returns:)");

    for (const auto& log_file : log_files) {
      try {
        std::ifstream file(log_file);
        std::string line;

        while (std::getline(file, line)) {
          std::smatch match;
          if (std::regex_search(line, match, game_line_regex)) {
            std::string timestamp_str = match[1].str();

            // Parse timestamp: YYYY-MM-DD HH:MM:SS.mmm
            std::tm tm = {};
            std::istringstream ss(timestamp_str);
            ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

            if (!ss.fail()) {
              // Extract milliseconds
              size_t dot_pos = timestamp_str.find_last_of('.');
              int milliseconds = 0;
              if (dot_pos != std::string::npos &&
                  dot_pos + 1 < timestamp_str.length()) {
                milliseconds = std::stoi(timestamp_str.substr(dot_pos + 1));
              }

              auto time_point =
                  std::chrono::system_clock::from_time_t(std::mktime(&tm));
              time_point += std::chrono::milliseconds(milliseconds);
              completion_times.push_back(time_point);
            }
          }
        }
      } catch (const std::exception& e) {
        // Skip files that can't be read
        continue;
      }
    }

    // Sort chronologically
    std::sort(completion_times.begin(), completion_times.end());
    return completion_times;
  }

  static auto FormatDuration(
      const std::chrono::system_clock::duration& duration) -> std::string {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    auto minutes =
        std::chrono::duration_cast<std::chrono::minutes>(duration - hours);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
        duration - hours - minutes);

    std::ostringstream oss;
    if (hours.count() > 0) {
      oss << hours.count() << "h ";
    }
    if (minutes.count() > 0) {
      oss << minutes.count() << "m ";
    }
    oss << seconds.count() << "s";

    return oss.str();
  }

  std::string log_directory_;
  std::atomic<bool> stop_monitoring_;
  std::thread monitor_thread_;
  int lines_printed_{};
  bool first_report_{};
};

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
  int train_batch_size = 256;
  // This should ideally match the number of actors to avoid contention.
  int inference_batch_size = 4;
  int inference_threads = 1;
  // This is crucial for performance, since a full NN inference is costly and
  // MCTS generates millions of states to be evaluated. A small cache size will
  // significantly slow down training.
  int inference_cache = 3200000;
  // replay_buffer_size/replay_buffer_reuse decides when we take a step to
  // train. Currently it takes about 15m for us to generate 10k states (200-300
  // game plays).
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
      : config_(std::move(config)),
        throughput_monitor_(config_.checkpoint_dir) {
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
    std::cout << "Inference batch size: " << config_.inference_batch_size
              << ", Inference threads: " << config_.inference_threads << '\n';
    std::cout << "Simulations per move: " << config_.max_simulations << '\n';
    std::cout << "Device: " << config_.device << '\n';
    std::cout << "Checkpoint dir: " << config_.checkpoint_dir << '\n';
    std::cout << '\n';

    auto start_time = std::chrono::high_resolution_clock::now();

    // Start throughput monitoring
    throughput_monitor_.Start();

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
      // Stop monitoring before reporting error
      throughput_monitor_.Stop();
      std::cerr << "Training failed: " << e.what() << '\n';
      throw;
    }

    // Stop throughput monitoring
    throughput_monitor_.Stop();

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
  ThroughputMonitor throughput_monitor_;
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
    } else if (arg.substr(0, 23) == "--inference-batch-size=") {
      config.inference_batch_size = std::stoi(arg.substr(23));
    } else if (arg.substr(0, 20) == "--inference-threads=") {
      config.inference_threads = std::stoi(arg.substr(20));
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