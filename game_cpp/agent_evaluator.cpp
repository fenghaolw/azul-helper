#include "agent_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace azul {

AgentEvaluator::AgentEvaluator(const EvaluationConfig &config)
    : config_(config) {}

auto AgentEvaluator::evaluate_agent(EvaluationAgent &test_agent,
                                    EvaluationAgent &baseline_agent)
    -> EvaluationResult {
  // Create result object
  EvaluationResult result(test_agent.get_name(), baseline_agent.get_name(),
                          config_);

  // Reset agent statistics
  test_agent.reset_stats();
  baseline_agent.reset_stats();

  // Plan games to play
  auto game_plans = plan_games();

  if (config_.verbose) {
    std::cout << "Starting evaluation: " << test_agent.get_name() << " vs "
              << baseline_agent.get_name() << '\n';
    std::cout << "Playing " << game_plans.size() << " games..." << '\n';
  }

  // Run all games
  for (const auto &plan : game_plans) {
    int game_id = std::get<0>(plan);
    int test_player = std::get<1>(plan);
    int baseline_player = std::get<2>(plan);
    int seed = std::get<3>(plan);

    try {
      GameResult game_result =
          run_single_game(test_agent, baseline_agent, game_id, test_player,
                          baseline_player, seed);

      result.game_results.push_back(game_result);

      // Update statistics
      if (game_result.winner == test_player) {
        result.test_agent_wins++;
      } else if (game_result.winner == baseline_player) {
        result.baseline_agent_wins++;
      } else {
        result.draws++;
      }

      if (game_result.timeout_occurred) {
        result.timeouts++;
      }
      if (!game_result.error_log.empty()) {
        result.errors++;
      }

    } catch (const std::exception &e) {
      if (config_.verbose) {
        std::cout << "Game " << game_id << " failed: " << e.what() << '\n';
      }
      result.errors++;
    }
  }

  result.games_played = static_cast<int>(result.game_results.size());

  // Calculate statistics
  result.finalize_results();

  // Calculate confidence interval and statistical significance
  result.confidence_interval = calculate_confidence_interval(
      result.test_agent_wins, result.games_played);

  auto [p_value, is_significant] = calculate_statistical_significance(
      result.test_agent_wins, result.games_played);
  result.p_value = p_value;
  result.is_statistically_significant = is_significant;

  if (config_.verbose) {
    std::cout << "\n=== Evaluation Summary ===" << '\n';
    std::cout << "Total games: " << result.games_played << '\n';
    std::cout << "Results: " << test_agent.get_name() << " " << std::fixed
              << std::setprecision(1) << (result.test_agent_win_rate * 100)
              << "% vs " << baseline_agent.get_name() << " "
              << (result.baseline_agent_win_rate * 100) << "%";
    if (result.draws > 0) {
      double draw_rate =
          static_cast<double>(result.draws) / result.games_played;
      std::cout << " (Draws: " << std::setprecision(1) << (draw_rate * 100)
                << "%)";
    }
    std::cout << '\n';
    if (result.timeouts > 0) {
      std::cout << "Timeouts: " << result.timeouts << '\n';
    }
    if (result.errors > 0) {
      std::cout << "Errors: " << result.errors << '\n';
    }
    std::cout << "Statistical significance: "
              << (result.is_statistically_significant ? "Yes" : "No")
              << " (p=" << std::setprecision(3) << result.p_value << ")"
              << '\n';
  }

  return result;
}

auto AgentEvaluator::quick_evaluation(EvaluationAgent &test_agent,
                                      EvaluationAgent &baseline_agent,
                                      int num_games) -> EvaluationResult {
  EvaluationConfig quick_config = config_;
  quick_config.num_games = num_games;
  quick_config.verbose = true;

  AgentEvaluator quick_evaluator(quick_config);
  return quick_evaluator.evaluate_agent(test_agent, baseline_agent);
}

auto AgentEvaluator::run_single_game(EvaluationAgent &test_agent,
                                     EvaluationAgent &baseline_agent,
                                     int game_id, int test_agent_player,
                                     int baseline_agent_player, int seed) const
    -> GameResult {
  auto start_time = std::chrono::high_resolution_clock::now();

  if (config_.verbose) {
    std::cout << "Starting game " << (game_id + 1) << ": "
              << test_agent.get_name() << " (P" << test_agent_player << ") vs "
              << baseline_agent.get_name() << " (P" << baseline_agent_player
              << ")";
    if (seed != -1) {
      std::cout << " [seed: " << seed << "]";
    }
    std::cout << '\n';
  }

  // Create OpenSpiel game
  open_spiel::GameParameters params;
  params["players"] = open_spiel::GameParameter(config_.num_players);
  if (seed != -1) {
    params["seed"] = open_spiel::GameParameter(seed);
  }
  auto game_instance = std::make_shared<AzulGame>(params);
  auto game_state = game_instance->NewInitialState();

  // Track performance statistics
  size_t test_nodes_before = test_agent.get_nodes_explored();
  size_t baseline_nodes_before = baseline_agent.get_nodes_explored();

  int total_moves = 0;
  int num_rounds = 1;
  int moves_since_round_start = 0;

  std::string error_log;
  bool timeout_occurred = false;

  try {
    while (!game_state->IsTerminal() &&
           total_moves < 200) { // Add move limit to prevent infinite loops
      int current_player = game_state->CurrentPlayer();

      // Check if game became terminal or is in chance node
      if (game_state->IsTerminal()) {
        break; // Game ended, exit loop
      }

      // Handle chance events (OpenSpiel uses -1 for chance player)
      if (current_player == -1) {
        // This is a chance node - apply the chance outcome
        auto chance_outcomes = game_state->ChanceOutcomes();
        if (!chance_outcomes.empty()) {
          // Take the first (usually only) chance outcome
          game_state->ApplyAction(chance_outcomes[0].first);
          continue;
        }
        // No chance outcomes available, this shouldn't happen
        throw std::runtime_error("Chance node with no outcomes available");
      }

      // Validate current player for regular (non-chance, non-terminal) states
      if (current_player < 0 || current_player >= config_.num_players) {
        throw std::runtime_error(
            "Invalid current player: " + std::to_string(current_player) +
            " (game terminal: " +
            std::to_string(static_cast<int>(game_state->IsTerminal())) + ")");
      }

      ActionType action = -1; // Default initialization for OpenSpiel action

      auto move_start = std::chrono::high_resolution_clock::now();

      try {
        // Pass the current player ID to the agent
        if (current_player == test_agent_player) {
          action = test_agent.get_action(*game_state, current_player);
        } else if (current_player == baseline_agent_player) {
          action = baseline_agent.get_action(*game_state, current_player);
        } else {
          throw std::runtime_error(
              "Unknown player mapping: current=" +
              std::to_string(current_player) +
              ", test=" + std::to_string(test_agent_player) +
              ", baseline=" + std::to_string(baseline_agent_player));
        }
      } catch (const std::exception &e) {
        // If agent fails, try to get a legal action as fallback
        auto legal_actions = game_state->LegalActions();
        if (!legal_actions.empty()) {
          action = legal_actions[0];
          if (config_.verbose) {
            std::cout << "Agent failed, using fallback action: " << e.what()
                      << '\n';
          }
        } else {
          throw std::runtime_error("No legal actions and agent failed: " +
                                   std::string(e.what()));
        }
      }

      auto move_end = std::chrono::high_resolution_clock::now();
      auto move_duration = std::chrono::duration_cast<std::chrono::seconds>(
          move_end - move_start);

      if (move_duration.count() > config_.timeout_per_move) {
        timeout_occurred = true;
        if (config_.verbose) {
          std::cout << "Timeout in game " << game_id << ", move " << total_moves
                    << '\n';
        }
        // Continue with the action anyway, but mark timeout
      }

      // Validate action before applying
      auto legal_actions = game_state->LegalActions();
      bool action_valid = false;
      for (const auto &legal_action : legal_actions) {
        if (action == legal_action) {
          action_valid = true;
          break;
        }
      }

      if (!action_valid) {
        if (!legal_actions.empty()) {
          action = legal_actions[0]; // Use first legal action as fallback
          if (config_.verbose) {
            std::cout << "Invalid action in game " << game_id
                      << ", using fallback" << '\n';
          }
        } else {
          throw std::runtime_error("No legal actions available in game " +
                                   std::to_string(game_id));
        }
      }

      // Apply action
      game_state->ApplyAction(action);

      total_moves++;
      moves_since_round_start++;

      // Detect round changes (basic heuristic - in a real implementation you'd
      // check game state)
      if (moves_since_round_start >=
          config_.num_players * 3) { // Rough estimate
        num_rounds++;
        moves_since_round_start = 0;
      }
    }
  } catch (const std::exception &e) {
    error_log = e.what();
    if (config_.verbose) {
      std::cout << "Game " << game_id << " error: " << error_log << '\n';
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto game_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  // Determine winner and final scores
  int winner = -1; // Default to draw
  std::vector<int> final_scores;

  if (game_state->IsTerminal()) {
    // Get actual game scores, not utility values
    const auto *azul_state =
        dynamic_cast<const open_spiel::azul::AzulState *>(game_state.get());
    final_scores.resize(config_.num_players);

    // Get actual Azul scores for each player
    int max_score = -1;
    for (int i = 0; i < config_.num_players; ++i) {
      final_scores[i] = azul_state->CalculateScore(i);
      if (final_scores[i] > max_score) {
        max_score = final_scores[i];
        winner = i;
      }
    }

    // Handle ties - use the same tiebreaker logic as AzulState::Returns()
    std::vector<int> tied_players;
    for (int i = 0; i < config_.num_players; ++i) {
      if (final_scores[i] == max_score) {
        tied_players.push_back(i);
      }
    }

    if (tied_players.size() > 1) {
      // Tiebreaker: most completed rows
      int max_completed_rows = -1;
      winner = -1; // Reset winner for tiebreaker

      for (int player : tied_players) {
        int completed_rows = 0;
        const auto &player_boards = azul_state->PlayerBoards();
        const auto &wall = player_boards[player].wall;

        for (int row = 0; row < open_spiel::azul::kWallSize; ++row) {
          bool row_complete = true;
          for (int col = 0; col < open_spiel::azul::kWallSize; ++col) {
            if (!wall[row][col]) {
              row_complete = false;
              break;
            }
          }
          if (row_complete) {
            completed_rows++;
          }
        }

        if (completed_rows > max_completed_rows) {
          max_completed_rows = completed_rows;
          winner = player;
        }
      }

      // If still tied, winner remains -1 (draw)
    }
  }

  // Track performance statistics
  size_t test_nodes_after = test_agent.get_nodes_explored();
  size_t baseline_nodes_after = baseline_agent.get_nodes_explored();

  GameResult result(game_id, winner, final_scores, num_rounds, total_moves,
                    game_duration.count() / 1000.0);
  result.timeout_occurred = timeout_occurred;
  result.error_log = error_log;
  result.test_agent_nodes = test_nodes_after - test_nodes_before;
  result.baseline_agent_nodes = baseline_nodes_after - baseline_nodes_before;

  if (config_.verbose) {
    std::cout << "Game " << (game_id + 1) << " completed: ";
    if (winner == test_agent_player) {
      std::cout << "Winner: " << test_agent.get_name();
    } else if (winner == baseline_agent_player) {
      std::cout << "Winner: " << baseline_agent.get_name();
    } else {
      std::cout << "Result: Draw";
    }
    std::cout << " | Duration: " << std::fixed << std::setprecision(2)
              << (game_duration.count() / 1000.0) << "s"
              << " | Moves: " << total_moves;
    if (!final_scores.empty()) {
      std::cout << " | Scores: [";
      for (size_t i = 0; i < final_scores.size(); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << final_scores[i];
      }
      std::cout << "]";
    }
    if (timeout_occurred) {
      std::cout << " | TIMEOUT";
    }
    if (!error_log.empty()) {
      std::cout << " | ERROR: " << error_log;
    }
    std::cout << '\n';
  }

  return result;
}

std::vector<std::tuple<int, int, int, int>> AgentEvaluator::plan_games() const {
  std::vector<std::tuple<int, int, int, int>> plans;

  // Initialize random number generator for seed generation
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> dist(1, 2147483647);

  if (config_.swap_player_positions) {
    // Split games between normal and swapped positions
    int normal_games = config_.num_games / 2;
    int swapped_games = config_.num_games - normal_games;

    // Normal position games (test agent as player 0)
    for (int i = 0; i < normal_games; ++i) {
      int seed =
          config_.use_fixed_seeds ? (config_.random_seed + i) : dist(gen);
      plans.emplace_back(i, 0, 1, seed);
    }

    // Swapped position games (test agent as player 1)
    for (int i = 0; i < swapped_games; ++i) {
      int seed = config_.use_fixed_seeds
                     ? (config_.random_seed + normal_games + i)
                     : dist(gen);
      plans.emplace_back(normal_games + i, 1, 0, seed);
    }
  } else {
    // All games with test agent as player 0
    for (int i = 0; i < config_.num_games; ++i) {
      int seed =
          config_.use_fixed_seeds ? (config_.random_seed + i) : dist(gen);
      plans.emplace_back(i, 0, 1, seed);
    }
  }

  return plans;
}

auto AgentEvaluator::calculate_statistical_significance(int test_wins,
                                                        int total_games)
    -> std::pair<double, bool> {
  if (total_games == 0) {
    return {1.0, false};
  }

  // Simple binomial test against 50% win rate
  double p = 0.5; // null hypothesis: equal performance
  double observed_rate = static_cast<double>(test_wins) / total_games;

  // Use normal approximation for large samples
  if (total_games >= 30) {
    double mean = total_games * p;
    double variance = total_games * p * (1 - p);
    double std_dev = std::sqrt(variance);

    double z_score = (test_wins - mean) / std_dev;

    // Two-tailed test
    double p_value = 2.0 * (1.0 - std::erf(std::abs(z_score) / std::sqrt(2.0)));

    bool is_significant = p_value < 0.05;
    return {p_value, is_significant};
  }
  // For small samples, use a simplified approach
  double p_value = std::abs(observed_rate - 0.5) > 0.3 ? 0.01 : 0.5;
  bool is_significant = p_value < 0.05;
  return {p_value, is_significant};
}

auto AgentEvaluator::calculate_confidence_interval(int wins,
                                                   int total_games) const
    -> std::pair<double, double> {
  if (total_games == 0) {
    return {0.0, 0.0};
  }

  double p = static_cast<double>(wins) / total_games;
  double alpha = 1.0 - config_.confidence_interval;
  double z = 1.96; // 95% confidence interval

  double margin = z * std::sqrt(p * (1 - p) / total_games);

  double lower = std::max(0.0, p - margin);
  double upper = std::min(1.0, p + margin);

  return {lower, upper};
}

// Tournament implementation
Tournament::Tournament(const EvaluationConfig &config)
    : config_(config), evaluator_(config) {}

auto Tournament::add_agent(std::unique_ptr<EvaluationAgent> agent) -> void {
  agents_.push_back(std::move(agent));
}

auto Tournament::run_tournament() -> TournamentResult {
  if (agents_.size() < 2) {
    throw std::runtime_error("Tournament requires at least 2 agents");
  }

  TournamentResult tournament_result;
  tournament_result.num_agents = static_cast<int>(agents_.size());

  // Initialize agent stats
  for (const auto &agent : agents_) {
    AgentStats stats;
    stats.agent_name = agent->get_name();
    tournament_result.agent_stats.push_back(stats);
  }

  if (config_.verbose) {
    std::cout << "Starting tournament with " << agents_.size() << " agents"
              << '\n';
  }

  // Run round-robin evaluation
  int total_matchups =
      static_cast<int>(agents_.size() * (agents_.size() - 1) / 2);
  int completed_matchups = 0;

  for (size_t i = 0; i < agents_.size(); ++i) {
    for (size_t j = i + 1; j < agents_.size(); ++j) {
      completed_matchups++;

      if (config_.verbose) {
        std::cout << "\n=== Matchup " << completed_matchups << "/"
                  << total_matchups << " ===\n";
        std::cout << "Evaluating " << agents_[i]->get_name() << " vs "
                  << agents_[j]->get_name() << '\n';
      }

      auto matchup_start = std::chrono::high_resolution_clock::now();
      EvaluationResult result =
          evaluator_.evaluate_agent(*agents_[i], *agents_[j]);
      auto matchup_end = std::chrono::high_resolution_clock::now();
      auto matchup_duration = std::chrono::duration_cast<std::chrono::seconds>(
          matchup_end - matchup_start);

      tournament_result.matchup_results.push_back(result);

      // Update agent stats
      tournament_result.agent_stats[i].games_played += result.games_played;
      tournament_result.agent_stats[j].games_played += result.games_played;
      tournament_result.agent_stats[i].wins += result.test_agent_wins;
      tournament_result.agent_stats[j].wins += result.baseline_agent_wins;
      tournament_result.agent_stats[i].total_score +=
          result.test_agent_avg_score * result.games_played;
      tournament_result.agent_stats[j].total_score +=
          result.baseline_agent_avg_score * result.games_played;

      if (config_.verbose) {
        std::cout << "Matchup completed in " << matchup_duration.count()
                  << "s - " << agents_[i]->get_name() << ": "
                  << result.test_agent_wins << " wins, "
                  << agents_[j]->get_name() << ": "
                  << result.baseline_agent_wins << " wins";
        if (result.draws > 0) {
          std::cout << ", " << result.draws << " draws";
        }
        std::cout << '\n';

        // Show current tournament standings
        std::cout << "Current standings:\n";
        std::vector<std::pair<std::string, double>> temp_standings;
        for (size_t k = 0; k < agents_.size(); ++k) {
          if (tournament_result.agent_stats[k].games_played > 0) {
            double win_rate =
                static_cast<double>(tournament_result.agent_stats[k].wins) /
                tournament_result.agent_stats[k].games_played;
            temp_standings.emplace_back(agents_[k]->get_name(), win_rate);
          }
        }
        std::sort(
            temp_standings.begin(), temp_standings.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

        for (size_t k = 0; k < temp_standings.size(); ++k) {
          std::cout << "  " << (k + 1) << ". " << temp_standings[k].first
                    << " - " << std::fixed << std::setprecision(1)
                    << (temp_standings[k].second * 100) << "% win rate\n";
        }
      }
    }
  }

  // Calculate final statistics
  for (auto &stats : tournament_result.agent_stats) {
    if (stats.games_played > 0) {
      stats.win_rate = static_cast<double>(stats.wins) / stats.games_played;
      stats.avg_score = stats.total_score / stats.games_played;
    }
  }

  tournament_result.calculate_rankings();

  // Sort agents by win rate (descending)
  std::sort(tournament_result.agent_stats.begin(),
            tournament_result.agent_stats.end(),
            [](const AgentStats &a, const AgentStats &b) {
              return a.win_rate > b.win_rate;
            });

  if (config_.verbose) {
    std::cout << "Tournament complete!" << '\n';
    std::cout << "Final rankings:" << '\n';
    for (size_t i = 0; i < tournament_result.agent_stats.size(); ++i) {
      const auto &stats = tournament_result.agent_stats[i];
      std::cout << (i + 1) << ". " << stats.agent_name
                << " - Win rate: " << (stats.win_rate * 100) << "%"
                << ", Avg score: " << stats.avg_score << '\n';
    }
  }

  return tournament_result;
}

// Factory functions
auto create_random_evaluation_agent(int seed, const std::string &name)
    -> std::unique_ptr<EvaluationAgent> {
  return std::make_unique<RandomAgentWrapper>(seed, name);
}

auto create_minimax_evaluation_agent(int depth, const std::string &name)
    -> std::unique_ptr<EvaluationAgent> {
  return std::make_unique<MinimaxAgentWrapper>(depth, name);
}

auto create_mcts_evaluation_agent(int num_simulations, double uct_c, int seed,
                                  const std::string &name)
    -> std::unique_ptr<EvaluationAgent> {
  return std::make_unique<MCTSAgentWrapper>(num_simulations, uct_c, seed, name);
}

auto create_alphazero_mcts_evaluation_agent(const std::string &checkpoint_path,
                                            int num_simulations, double uct_c,
                                            int seed, const std::string &name)
    -> std::unique_ptr<EvaluationAgent> {
  return std::make_unique<AlphaZeroMCTSAgentWrapper>(
      checkpoint_path, num_simulations, uct_c, seed, name);
}

} // namespace azul