#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace azul {

/**
 * Configuration for agent evaluation experiments.
 */
struct EvaluationConfig {
  // Basic evaluation settings
  int num_games = 100;
  int num_players = 2;
  double timeout_per_move = 15.0;  // seconds - increased for deeper minimax

  // Agent evaluation settings
  bool deterministic_evaluation =
      true;  // Use deterministic agent actions if possible
  bool swap_player_positions =
      true;  // Evaluate with agents in different starting positions

  // Randomization settings
  bool use_fixed_seeds = true;  // Use fixed seeds for reproducibility
  int random_seed = 42;

  // Logging and output
  bool verbose = false;
  bool save_detailed_logs = false;

  // Result aggregation
  double confidence_interval = 0.95;  // For statistical significance testing
};

/**
 * Result of a single game evaluation.
 */
struct GameResult {
  int game_id;
  int winner;                     // Player index of winner (-1 for draw)
  std::vector<int> final_scores;  // Final scores for all players
  int num_rounds;
  int total_moves;       // Total number of moves made in the game
  double game_duration;  // seconds

  // Performance statistics
  size_t test_agent_nodes = 0;
  size_t baseline_agent_nodes = 0;
  double test_agent_thinking_time = 0.0;
  double baseline_agent_thinking_time = 0.0;

  // Optional detailed information
  std::string error_log;
  bool timeout_occurred = false;

  GameResult(int id, int win, const std::vector<int>& scores, int rounds,
             int moves, double duration)
      : game_id(id),
        winner(win),
        final_scores(scores),
        num_rounds(rounds),
        total_moves(moves),
        game_duration(duration) {}
};

/**
 * Comprehensive results of an agent evaluation.
 */
struct EvaluationResult {
  // Evaluation metadata
  std::string timestamp;
  EvaluationConfig config;

  // Agent information
  std::string test_agent_name;
  std::string baseline_agent_name;

  // Game results
  int games_played;
  std::vector<GameResult> game_results;

  // Aggregate statistics
  int test_agent_wins;
  int baseline_agent_wins;
  int draws;

  // Performance metrics
  double test_agent_win_rate;
  double baseline_agent_win_rate;
  double test_agent_avg_score;
  double baseline_agent_avg_score;
  double average_score_difference;
  double average_game_duration;

  // Statistical analysis
  std::pair<double, double> confidence_interval;  // lower, upper bounds
  double p_value;
  bool is_statistically_significant;

  // Additional metrics
  int timeouts = 0;
  int errors = 0;

  // Constructor
  EvaluationResult(const std::string& test_name,
                   const std::string& baseline_name,
                   const EvaluationConfig& cfg)
      : test_agent_name(test_name),
        baseline_agent_name(baseline_name),
        config(cfg),
        games_played(0),
        test_agent_wins(0),
        baseline_agent_wins(0),
        draws(0),
        test_agent_win_rate(0.0),
        baseline_agent_win_rate(0.0),
        test_agent_avg_score(0.0),
        baseline_agent_avg_score(0.0),
        average_score_difference(0.0),
        average_game_duration(0.0),
        confidence_interval(0.0, 0.0),
        p_value(1.0),
        is_statistically_significant(false) {
    // Generate timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S",
                  std::localtime(&time_t));
    timestamp = std::string(buffer);
  }

  // Calculate derived metrics
  void finalize_results();

  // Generate summary string
  std::string summary() const;
};

/**
 * Statistics for a single agent in a tournament.
 */
struct AgentStats {
  std::string agent_name;
  int games_played = 0;
  int wins = 0;
  double win_rate = 0.0;
  double total_score = 0.0;
  double avg_score = 0.0;
};

/**
 * Results of a multi-agent tournament.
 */
struct TournamentResult {
  std::string timestamp;
  int num_agents = 0;
  std::vector<AgentStats> agent_stats;
  std::vector<EvaluationResult> matchup_results;
  EvaluationConfig config;

  // Legacy support
  std::vector<std::string> agents;
  std::vector<std::pair<std::string, std::string>> matchups;
  std::vector<EvaluationResult> evaluation_results;

  // Rankings: (agent_name, win_rate, avg_score_diff)
  std::vector<std::tuple<std::string, double, double>> rankings;

  TournamentResult() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S",
                  std::localtime(&time_t));
    timestamp = std::string(buffer);
  }

  void calculate_rankings();
  std::string summary() const;
};

}  // namespace azul