#pragma once

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

namespace azul {

// Forward declarations
class MinimaxAgent;
class AzulMCTSAgent;
class AlphaZeroMCTSAgentWrapper;

// Type aliases for compatibility
using GameState = open_spiel::State;
using Action = open_spiel::Action;

/**
 * High-resolution timer for precise performance measurements
 */
class Timer {
 public:
  Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); }

  [[nodiscard]] auto elapsed_ms() const -> double {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_);
    return duration.count() / 1000.0;
  }

  [[nodiscard]] auto elapsed_us() const -> long {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                 start_time_)
        .count();
  }

 private:
  std::chrono::high_resolution_clock::time_point start_time_;
};

/**
 * Statistics tracking for function calls
 */
struct FunctionStats {
  size_t call_count = 0;
  double total_time_ms = 0.0;
  double min_time_ms = std::numeric_limits<double>::max();
  double max_time_ms = 0.0;
  double avg_time_ms = 0.0;

  void update(double time_ms) {
    call_count++;
    total_time_ms += time_ms;
    min_time_ms = std::min(min_time_ms, time_ms);
    max_time_ms = std::max(max_time_ms, time_ms);
    avg_time_ms = total_time_ms / call_count;
  }

  [[nodiscard]] auto get_percentage(double total_time) const -> double {
    return total_time > 0.0 ? (total_time_ms / total_time) * 100.0 : 0.0;
  }
};

/**
 * Memory usage tracking
 */
struct MemoryStats {
  size_t peak_memory_kb = 0;
  size_t current_memory_kb = 0;
  size_t allocations = 0;
  size_t deallocations = 0;

  void track_allocation(size_t size_bytes) {
    allocations++;
    current_memory_kb += size_bytes / 1024;
    peak_memory_kb = std::max(peak_memory_kb, current_memory_kb);
  }

  void track_deallocation(size_t size_bytes) {
    deallocations++;
    current_memory_kb -= size_bytes / 1024;
  }
};

/**
 * Comprehensive profiler for Azul agents
 */
class AgentProfiler {
 public:
  static auto instance() -> AgentProfiler& {
    static AgentProfiler instance;
    return instance;
  }

  // Profiling control
  void start_profiling() {
    profiling_enabled_ = true;
    total_timer_.reset();
  }
  void stop_profiling() { profiling_enabled_ = false; }
  void reset_stats() {
    function_stats_.clear();
    memory_stats_ = MemoryStats{};
  }

  // Function timing
  void start_function(const std::string& function_name);
  void end_function(const std::string& function_name);

  // Memory tracking
  void track_memory_allocation(const std::string& context, size_t size_bytes);
  void track_memory_deallocation(const std::string& context, size_t size_bytes);

  // Agent-specific profiling
  void profile_minimax_search(
      const std::function<open_spiel::Action()>& search_func,
      const std::string& context = "minimax_search");
  void profile_mcts_search(
      const std::function<open_spiel::Action()>& search_func,
      const std::string& context = "mcts_search");

  // Reporting
  void print_profile_report(std::ostream& os = std::cout) const;
  void save_profile_report(const std::string& filename) const;
  void print_hotspots(std::ostream& os = std::cout, size_t top_n = 10) const;

  // Getters
  [[nodiscard]] auto is_profiling() const -> bool { return profiling_enabled_; }
  [[nodiscard]] auto get_function_stats() const
      -> const std::unordered_map<std::string, FunctionStats>& {
    return function_stats_;
  }
  [[nodiscard]] auto get_memory_stats() const -> const MemoryStats& {
    return memory_stats_;
  }
  [[nodiscard]] auto get_total_time_ms() const -> double {
    return total_timer_.elapsed_ms();
  }

 private:
  AgentProfiler() = default;

  bool profiling_enabled_ = false;
  Timer total_timer_;
  std::unordered_map<std::string, Timer> active_timers_;
  std::unordered_map<std::string, FunctionStats> function_stats_;
  MemoryStats memory_stats_;
};

/**
 * RAII helper for automatic function profiling
 */
class ScopedProfiler {
 public:
  explicit ScopedProfiler(std::string function_name)
      : function_name_(std::move(function_name)) {
    if (AgentProfiler::instance().is_profiling()) {
      AgentProfiler::instance().start_function(function_name_);
    }
  }

  ~ScopedProfiler() {
    if (AgentProfiler::instance().is_profiling()) {
      AgentProfiler::instance().end_function(function_name_);
    }
  }

 private:
  std::string function_name_;
};

/**
 * Profiling wrapper for minimax agent
 */
class ProfiledMinimaxAgent {
 public:
  explicit ProfiledMinimaxAgent(std::unique_ptr<MinimaxAgent> agent);

  [[nodiscard]] auto get_action(const open_spiel::State& state)
      -> open_spiel::Action;
  [[nodiscard]] static auto get_action_probabilities(
      const open_spiel::State& state) -> std::vector<double>;
  void reset();

  // Access to underlying agent
  [[nodiscard]] auto agent() const -> const MinimaxAgent& { return *agent_; }
  [[nodiscard]] auto agent() -> MinimaxAgent& { return *agent_; }

 private:
  std::unique_ptr<MinimaxAgent> agent_;
};

/**
 * Profiling wrapper for MCTS agent
 */
class ProfiledMCTSAgent {
 public:
  explicit ProfiledMCTSAgent(std::unique_ptr<AzulMCTSAgent> agent);

  [[nodiscard]] auto get_action(const open_spiel::State& state)
      -> open_spiel::Action;
  [[nodiscard]] auto get_action_probabilities(const open_spiel::State& state,
                                              double temperature = 1.0)
      -> std::vector<double>;
  void reset();

  // Access to underlying agent
  [[nodiscard]] auto agent() const -> const AzulMCTSAgent& { return *agent_; }
  [[nodiscard]] auto agent() -> AzulMCTSAgent& { return *agent_; }

 private:
  std::unique_ptr<AzulMCTSAgent> agent_;
};

/**
 * Profiling wrapper for AlphaZero MCTS agent
 */
class ProfiledAlphaZeroMCTSAgent {
 public:
  explicit ProfiledAlphaZeroMCTSAgent(
      std::unique_ptr<AlphaZeroMCTSAgentWrapper> agent);

  [[nodiscard]] auto get_action(const open_spiel::State& state)
      -> open_spiel::Action;
  void reset();

  // Access to underlying agent
  [[nodiscard]] auto agent() const -> const AlphaZeroMCTSAgentWrapper& {
    return *agent_;
  }
  [[nodiscard]] auto agent() -> AlphaZeroMCTSAgentWrapper& { return *agent_; }

 private:
  std::unique_ptr<AlphaZeroMCTSAgentWrapper> agent_;
};

// Convenience macros for profiling
#define PROFILE_FUNCTION() ScopedProfiler _prof(__FUNCTION__)
#define PROFILE_SCOPE(name) ScopedProfiler _prof(name)

// Factory functions
[[nodiscard]] auto create_profiled_minimax_agent(int player_id, int depth = 4)
    -> std::unique_ptr<ProfiledMinimaxAgent>;

[[nodiscard]] auto create_profiled_mcts_agent(int player_id,
                                              int num_simulations = 1000,
                                              double uct_c = 1.4, int seed = -1)
    -> std::unique_ptr<ProfiledMCTSAgent>;

[[nodiscard]] auto create_profiled_alphazero_agent(
    const std::string& checkpoint_path, int num_simulations = 400,
    double uct_c = 1.4, int seed = -1)
    -> std::unique_ptr<ProfiledAlphaZeroMCTSAgent>;

}  // namespace azul