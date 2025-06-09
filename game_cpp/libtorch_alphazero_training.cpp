#include <atomic>
#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "azul.h"
#include "open_spiel/algorithms/alpha_zero_torch/alpha_zero.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/thread.h"

using json = nlohmann::json;

void force_azul_registration() {
  // Reference symbols from azul namespace to force linking
  (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}

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
  int replay_buffer_size = 1000000;
  int replay_buffer_reuse = 100;
  int checkpoint_freq = 10;
  int evaluation_window = 100;

  // MCTS parameters
  double uct_c = 1.4;
  double policy_alpha = 0.3;
  double policy_epsilon = 0.25;
  double temperature = 1.0;
  double temperature_drop = 30.0;
  double cutoff_probability = 0.0;
  double cutoff_value = 0.0;

  // Evaluation parameters
  int eval_levels = 3;

  // Output settings
  std::string checkpoint_dir = "models/libtorch_alphazero_azul";
  std::string device = "cpu";  // "cpu", "cuda", "mps"
  bool explicit_learning = false;
  bool resume_from_checkpoint =
      false;  // Whether to resume from existing checkpoint
};

void from_json(const json& j, LibTorchAZConfig& config) {
  config.game_name = j.value("game_name", "azul");
  config.max_steps = j.value("max_steps", 1000);
  config.actors = j.value("actors", 4);
  config.evaluators = j.value("evaluators", 1);
  config.max_simulations = j.value("max_simulations", 400);
  config.nn_model = j.value("nn_model", "resnet");
  config.nn_width = j.value("nn_width", 128);
  config.nn_depth = j.value("nn_depth", 6);
  config.learning_rate = j.value("learning_rate", 0.001);
  config.weight_decay = j.value("weight_decay", 1e-4);
  config.train_batch_size = j.value("train_batch_size", 256);
  config.inference_batch_size = j.value("inference_batch_size", 4);
  config.inference_threads = j.value("inference_threads", 1);
  config.inference_cache = j.value("inference_cache", 3200000);
  config.replay_buffer_size = j.value("replay_buffer_size", 1000000);
  config.replay_buffer_reuse = j.value("replay_buffer_reuse", 100);
  config.checkpoint_freq = j.value("checkpoint_freq", 10);
  config.evaluation_window = j.value("evaluation_window", 100);
  config.uct_c = j.value("uct_c", 1.4);
  config.policy_alpha = j.value("policy_alpha", 0.3);
  config.policy_epsilon = j.value("policy_epsilon", 0.25);
  config.temperature = j.value("temperature", 1.0);
  config.temperature_drop = j.value("temperature_drop", 30.0);
  config.cutoff_probability = j.value("cutoff_probability", 0.0);
  config.cutoff_value = j.value("cutoff_value", 0.0);
  config.eval_levels = j.value("eval_levels", 3);
  config.checkpoint_dir =
      j.value("checkpoint_dir", "models/libtorch_alphazero_azul");
  config.device = j.value("device", "cpu");
  // Infer explicit_learning based on multiple devices
  config.explicit_learning = config.device.find(',') != std::string::npos;
  config.resume_from_checkpoint = j.value("resume_from_checkpoint", false);
}

/**
 * Load configuration from JSON file
 */
auto LoadConfigFromJson(const std::string& config_file) -> LibTorchAZConfig {
  try {
    std::ifstream file(config_file);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open config file: " + config_file);
    }

    json j;
    file >> j;
    LibTorchAZConfig config;
    from_json(j, config);
    return config;

  } catch (const std::exception& e) {
    throw std::runtime_error("Error loading config file: " +
                             std::string(e.what()));
  }
}

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

/**
 * Command line argument parsing using cxxopts
 */
auto ParseArguments(int argc, char** argv) -> LibTorchAZConfig {
  LibTorchAZConfig config;
  std::string config_file;

  try {
    cxxopts::Options options("libtorch_alphazero_training",
                             "LibTorch AlphaZero Training for Azul");

    options.add_options()("c,config",
                          "Load configuration from JSON file (mutually "
                          "exclusive with other options)",
                          cxxopts::value<std::string>())(
        "s,steps", "Training steps",
        cxxopts::value<int>()->default_value("1000"))(
        "a,actors", "Actor threads", cxxopts::value<int>()->default_value("4"))(
        "e,evaluators", "Evaluator threads",
        cxxopts::value<int>()->default_value("1"))(
        "simulations", "MCTS simulations per move",
        cxxopts::value<int>()->default_value("400"))(
        "m,model", "NN model: mlp|conv2d|resnet",
        cxxopts::value<std::string>()->default_value("resnet"))(
        "w,width", "NN width", cxxopts::value<int>()->default_value("128"))(
        "d,depth", "NN depth", cxxopts::value<int>()->default_value("6"))(
        "lr", "Learning rate",
        cxxopts::value<double>()->default_value("0.001"))(
        "b,batch", "Batch size", cxxopts::value<int>()->default_value("256"))(
        "device", "Device: cpu|cuda|mps (comma-separated for multiple devices)",
        cxxopts::value<std::string>()->default_value("cpu"))(
        "dir", "Checkpoint directory",
        cxxopts::value<std::string>()->default_value(
            "models/libtorch_alphazero_azul"))(
        "resume", "Resume from existing checkpoint")("h,help", "Show help");

    auto result = options.parse(argc, argv);

    if (result.count("help") != 0U) {
      std::cout << options.help() << '\n';
      exit(0);
    }

    // Check if config file is specified
    bool has_config = result.count("config") != 0U;

    // Check if any other options are specified
    bool has_other_options =
        result.count("steps") != 0U || result.count("actors") != 0U ||
        result.count("evaluators") != 0U || result.count("simulations") != 0U ||
        result.count("model") != 0U || result.count("width") != 0U ||
        result.count("depth") != 0U || result.count("lr") != 0U ||
        result.count("batch") != 0U || result.count("device") != 0U ||
        result.count("dir") != 0U || result.count("resume") != 0U;

    // If both config and other options are specified, show error
    if (has_config && has_other_options) {
      std::cerr << "Error: --config option cannot be used with other "
                   "command-line options.\n"
                << "Either use a config file or specify individual options, "
                   "but not both.\n";
      exit(1);
    }

    if (has_config) {
      // Load from config file
      config_file = result["config"].as<std::string>();
      try {
        config = LoadConfigFromJson(config_file);
      } catch (const std::exception& e) {
        std::cerr << "Error loading config file: " << e.what() << '\n';
        exit(1);
      }
    } else {
      // Use command line arguments
      config.max_steps = result["steps"].as<int>();
      config.actors = result["actors"].as<int>();
      config.evaluators = result["evaluators"].as<int>();
      config.max_simulations = result["simulations"].as<int>();
      config.nn_model = result["model"].as<std::string>();
      config.nn_width = result["width"].as<int>();
      config.nn_depth = result["depth"].as<int>();
      config.learning_rate = result["lr"].as<double>();
      config.train_batch_size = result["batch"].as<int>();
      config.device = result["device"].as<std::string>();
      // Infer explicit_learning based on multiple devices
      config.explicit_learning = config.device.find(',') != std::string::npos;
      config.checkpoint_dir = result["dir"].as<std::string>();

      if (result.count("resume") != 0U) {
        config.resume_from_checkpoint = true;
      }
    }

  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "Error parsing options: " << e.what() << '\n';
    exit(1);
  }

  return config;
}

}  // namespace azul

auto main(int argc, char** argv) -> int {
  try {
    auto config = azul::ParseArguments(argc, argv);
    azul::LibTorchAZTrainer trainer(config);
    trainer.Train();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}