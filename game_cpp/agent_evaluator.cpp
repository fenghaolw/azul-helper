#include "agent_evaluator.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace azul {

AgentEvaluator::AgentEvaluator(const EvaluationConfig& config) 
    : config_(config) {}

EvaluationResult AgentEvaluator::evaluate_agent(
    EvaluationAgent& test_agent,
    EvaluationAgent& baseline_agent
) {
    // Create result object
    EvaluationResult result(test_agent.get_name(), baseline_agent.get_name(), config_);
    
    // Reset agent statistics
    test_agent.reset_stats();
    baseline_agent.reset_stats();
    
    // Plan games to play
    auto game_plans = plan_games();
    
    if (config_.verbose) {
        std::cout << "Starting evaluation: " << test_agent.get_name() 
                  << " vs " << baseline_agent.get_name() << std::endl;
        std::cout << "Playing " << game_plans.size() << " games..." << std::endl;
    }
    
    // Run all games
    for (const auto& plan : game_plans) {
        int game_id = std::get<0>(plan);
        int test_player = std::get<1>(plan);
        int baseline_player = std::get<2>(plan);
        int seed = std::get<3>(plan);
        
        try {
            GameResult game_result = run_single_game(
                test_agent, baseline_agent, game_id, test_player, baseline_player, seed
            );
            
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
            
        } catch (const std::exception& e) {
            if (config_.verbose) {
                std::cout << "Game " << game_id << " failed: " << e.what() << std::endl;
            }
            result.errors++;
        }
        
        if (config_.verbose && (game_id + 1) % 10 == 0) {
            std::cout << "Completed " << (game_id + 1) << "/" << game_plans.size() << " games" << std::endl;
        }
    }
    
    result.games_played = static_cast<int>(result.game_results.size());
    
    // Calculate statistics
    result.finalize_results();
    
    // Calculate confidence interval and statistical significance
    result.confidence_interval = calculate_confidence_interval(
        result.test_agent_wins, result.games_played
    );
    
    auto [p_value, is_significant] = calculate_statistical_significance(
        result.test_agent_wins, result.games_played
    );
    result.p_value = p_value;
    result.is_statistically_significant = is_significant;
    
    if (config_.verbose) {
        std::cout << "Evaluation complete!" << std::endl;
        std::cout << "Results: " << test_agent.get_name() << " " 
                  << (result.test_agent_win_rate * 100) << "% vs " 
                  << baseline_agent.get_name() << " " 
                  << (result.baseline_agent_win_rate * 100) << "%" << std::endl;
    }
    
    return result;
}

EvaluationResult AgentEvaluator::quick_evaluation(
    EvaluationAgent& test_agent,
    EvaluationAgent& baseline_agent,
    int num_games
) {
    EvaluationConfig quick_config = config_;
    quick_config.num_games = num_games;
    quick_config.verbose = true;
    
    AgentEvaluator quick_evaluator(quick_config);
    return quick_evaluator.evaluate_agent(test_agent, baseline_agent);
}

GameResult AgentEvaluator::run_single_game(
    EvaluationAgent& test_agent,
    EvaluationAgent& baseline_agent,
    int game_id,
    int test_agent_player,
    int baseline_agent_player,
    int seed
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
#ifdef WITH_OPENSPIEL
    // Create OpenSpiel game
    open_spiel::GameParameters params;
    params["players"] = open_spiel::GameParameter(config_.num_players);
    if (seed != -1) {
        params["seed"] = open_spiel::GameParameter(seed);
    }
    auto game_instance = std::make_shared<AzulGame>(params);
    auto game_state = game_instance->NewInitialState();
#else
    // Create game using legacy GameState
    GameState game_state_obj = create_game(config_.num_players, seed);
    auto& game_state = game_state_obj; // For compatibility with the rest of the function
#endif
    
    // Track performance statistics
    size_t test_nodes_before = test_agent.get_nodes_explored();
    size_t baseline_nodes_before = baseline_agent.get_nodes_explored();
    
    int total_moves = 0;
    int num_rounds = 1;
    int moves_since_round_start = 0;
    
    std::string error_log;
    bool timeout_occurred = false;
    
    try {
#ifdef WITH_OPENSPIEL
        while (!game_state->IsTerminal() && total_moves < 200) { // Add move limit to prevent infinite loops
            int current_player = game_state->CurrentPlayer();
#else
        while (!game_state.is_game_over() && total_moves < 200) { // Add move limit to prevent infinite loops
            int current_player = game_state.current_player();
#endif
            
            // Validate current player
            if (current_player < 0 || current_player >= config_.num_players) {
                throw std::runtime_error("Invalid current player: " + std::to_string(current_player));
            }
            
#ifdef WITH_OPENSPIEL
            ActionType action = -1; // Default initialization for OpenSpiel action
#else
            Action action(0, TileColor::BLUE, 0); // Default initialization for legacy action
#endif
            
            auto move_start = std::chrono::high_resolution_clock::now();
            
            try {
                // Pass the current player ID to the agent
                if (current_player == test_agent_player) {
#ifdef WITH_OPENSPIEL
                    action = test_agent.get_action(*game_state, current_player);
#else
                    action = test_agent.get_action(game_state, current_player);
#endif
                } else if (current_player == baseline_agent_player) {
#ifdef WITH_OPENSPIEL
                    action = baseline_agent.get_action(*game_state, current_player);
#else
                    action = baseline_agent.get_action(game_state, current_player);
#endif
                } else {
                    throw std::runtime_error("Unknown player mapping: current=" + std::to_string(current_player) +
                                           ", test=" + std::to_string(test_agent_player) +
                                           ", baseline=" + std::to_string(baseline_agent_player));
                }
            } catch (const std::exception& e) {
                // If agent fails, try to get a legal action as fallback
#ifdef WITH_OPENSPIEL
                auto legal_actions = game_state->LegalActions();
                if (!legal_actions.empty()) {
                    action = legal_actions[0];
#else
                auto legal_actions = game_state.get_legal_actions(current_player);
                if (!legal_actions.empty()) {
                    action = legal_actions[0];
#endif
                    if (config_.verbose) {
                        std::cout << "Agent failed, using fallback action: " << e.what() << std::endl;
                    }
                } else {
                    throw std::runtime_error("No legal actions and agent failed: " + std::string(e.what()));
                }
            }
            
            auto move_end = std::chrono::high_resolution_clock::now();
            auto move_duration = std::chrono::duration_cast<std::chrono::seconds>(move_end - move_start);
            
            if (move_duration.count() > config_.timeout_per_move) {
                timeout_occurred = true;
                if (config_.verbose) {
                    std::cout << "Timeout in game " << game_id << ", move " << total_moves << std::endl;
                }
                // Continue with the action anyway, but mark timeout
            }
            
            // Validate action before applying
#ifdef WITH_OPENSPIEL
            auto legal_actions = game_state->LegalActions();
#else
            auto legal_actions = game_state.get_legal_actions(current_player);
#endif
            bool action_valid = false;
            for (const auto& legal_action : legal_actions) {
                if (action == legal_action) {
                    action_valid = true;
                    break;
                }
            }
            
            if (!action_valid) {
                if (!legal_actions.empty()) {
                    action = legal_actions[0]; // Use first legal action as fallback
                    if (config_.verbose) {
                        std::cout << "Invalid action in game " << game_id << ", using fallback" << std::endl;
                    }
                } else {
                    throw std::runtime_error("No legal actions available in game " + std::to_string(game_id));
                }
            }
            
            // Apply action
#ifdef WITH_OPENSPIEL
            game_state->ApplyAction(action);
#else
            bool action_applied = game_state.apply_action(action, false);
            if (!action_applied) {
                throw std::runtime_error("Failed to apply action in game " + std::to_string(game_id));
            }
#endif
            
            total_moves++;
            moves_since_round_start++;
            
            // Detect round changes (basic heuristic - in a real implementation you'd check game state)
            if (moves_since_round_start >= config_.num_players * 3) { // Rough estimate
                num_rounds++;
                moves_since_round_start = 0;
            }
        }
    } catch (const std::exception& e) {
        error_log = e.what();
        if (config_.verbose) {
            std::cout << "Game " << game_id << " error: " << error_log << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto game_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Determine winner and final scores
    int winner = -1; // Default to draw
    std::vector<int> final_scores;
    
#ifdef WITH_OPENSPIEL
    if (game_state->IsTerminal()) {
        auto returns = game_state->Returns();
        final_scores.resize(returns.size());
        
        // Convert returns to scores (assuming returns are in [-1, 1] range)
        double max_return = *std::max_element(returns.begin(), returns.end());
        for (size_t i = 0; i < returns.size(); ++i) {
            final_scores[i] = static_cast<int>(returns[i] * 100); // Scale to reasonable score range
            if (returns[i] == max_return) {
                winner = static_cast<int>(i);
            }
        }
    }
#else
    if (game_state.is_game_over()) {
        winner = game_state.get_winner();
        final_scores = game_state.get_scores();
    }
#endif
    
    // Track performance statistics
    size_t test_nodes_after = test_agent.get_nodes_explored();
    size_t baseline_nodes_after = baseline_agent.get_nodes_explored();
    
    GameResult result(game_id, winner, final_scores, num_rounds, total_moves, game_duration.count() / 1000.0);
    result.timeout_occurred = timeout_occurred;
    result.error_log = error_log;
    result.test_agent_nodes = test_nodes_after - test_nodes_before;
    result.baseline_agent_nodes = baseline_nodes_after - baseline_nodes_before;
    
    return result;
}

std::vector<std::tuple<int, int, int, int>> AgentEvaluator::plan_games() const {
    std::vector<std::tuple<int, int, int, int>> plans;
    
    if (config_.swap_player_positions) {
        // Split games between normal and swapped positions
        int normal_games = config_.num_games / 2;
        int swapped_games = config_.num_games - normal_games;
        
        // Normal position games (test agent as player 0)
        for (int i = 0; i < normal_games; ++i) {
            int seed = config_.use_fixed_seeds ? (config_.random_seed + i) : -1;
            plans.emplace_back(i, 0, 1, seed);
        }
        
        // Swapped position games (test agent as player 1)
        for (int i = 0; i < swapped_games; ++i) {
            int seed = config_.use_fixed_seeds ? (config_.random_seed + normal_games + i) : -1;
            plans.emplace_back(normal_games + i, 1, 0, seed);
        }
    } else {
        // All games with test agent as player 0
        for (int i = 0; i < config_.num_games; ++i) {
            int seed = config_.use_fixed_seeds ? (config_.random_seed + i) : -1;
            plans.emplace_back(i, 0, 1, seed);
        }
    }
    
    return plans;
}

std::pair<double, bool> AgentEvaluator::calculate_statistical_significance(
    int test_wins, int total_games
) const {
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
    } else {
        // For small samples, use a simplified approach
        double p_value = std::abs(observed_rate - 0.5) > 0.3 ? 0.01 : 0.5;
        bool is_significant = p_value < 0.05;
        return {p_value, is_significant};
    }
}

std::pair<double, double> AgentEvaluator::calculate_confidence_interval(
    int wins, int total_games
) const {
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
Tournament::Tournament(const EvaluationConfig& config) 
    : config_(config), evaluator_(config) {}

void Tournament::add_agent(std::unique_ptr<EvaluationAgent> agent) {
    agents_.push_back(std::move(agent));
}

TournamentResult Tournament::run_tournament() {
    if (agents_.size() < 2) {
        throw std::runtime_error("Tournament requires at least 2 agents");
    }
    
    TournamentResult tournament_result;
    tournament_result.num_agents = static_cast<int>(agents_.size());
    
    // Initialize agent stats
    for (const auto& agent : agents_) {
        AgentStats stats;
        stats.agent_name = agent->get_name();
        tournament_result.agent_stats.push_back(stats);
    }
    
    if (config_.verbose) {
        std::cout << "Starting tournament with " << agents_.size() << " agents" << std::endl;
    }
    
    // Run round-robin evaluation
    for (size_t i = 0; i < agents_.size(); ++i) {
        for (size_t j = i + 1; j < agents_.size(); ++j) {
            if (config_.verbose) {
                std::cout << "Evaluating " << agents_[i]->get_name() 
                          << " vs " << agents_[j]->get_name() << std::endl;
            }
            
            EvaluationResult result = evaluator_.evaluate_agent(*agents_[i], *agents_[j]);
            tournament_result.matchup_results.push_back(result);
            
            // Update agent stats
            tournament_result.agent_stats[i].games_played += result.games_played;
            tournament_result.agent_stats[j].games_played += result.games_played;
            tournament_result.agent_stats[i].wins += result.test_agent_wins;
            tournament_result.agent_stats[j].wins += result.baseline_agent_wins;
            tournament_result.agent_stats[i].total_score += result.test_agent_avg_score * result.games_played;
            tournament_result.agent_stats[j].total_score += result.baseline_agent_avg_score * result.games_played;
        }
    }
    
    // Calculate final statistics
    for (auto& stats : tournament_result.agent_stats) {
        if (stats.games_played > 0) {
            stats.win_rate = static_cast<double>(stats.wins) / stats.games_played;
            stats.avg_score = stats.total_score / stats.games_played;
        }
    }
    
    tournament_result.calculate_rankings();
    
    // Sort agents by win rate (descending)
    std::sort(tournament_result.agent_stats.begin(), tournament_result.agent_stats.end(),
              [](const AgentStats& a, const AgentStats& b) {
                  return a.win_rate > b.win_rate;
              });
    
    if (config_.verbose) {
        std::cout << "Tournament complete!" << std::endl;
        std::cout << "Final rankings:" << std::endl;
        for (size_t i = 0; i < tournament_result.agent_stats.size(); ++i) {
            const auto& stats = tournament_result.agent_stats[i];
            std::cout << (i + 1) << ". " << stats.agent_name 
                      << " - Win rate: " << (stats.win_rate * 100) << "%" 
                      << ", Avg score: " << stats.avg_score << std::endl;
        }
    }
    
    return tournament_result;
}

// Factory functions
std::unique_ptr<EvaluationAgent> create_random_evaluation_agent(
    int seed, const std::string& name
) {
    return std::make_unique<RandomAgentWrapper>(seed, name);
}

std::unique_ptr<EvaluationAgent> create_minimax_evaluation_agent(
    int depth, bool enable_alpha_beta, 
    int seed, const std::string& name
) {
    return std::make_unique<MinimaxAgentWrapper>(
        depth, enable_alpha_beta, seed, name
    );
}

#ifdef WITH_OPENSPIEL
std::unique_ptr<EvaluationAgent> create_mcts_evaluation_agent(
    int num_simulations, double uct_c,
    int seed, const std::string& name
) {
    return std::make_unique<MCTSAgentWrapper>(
        num_simulations, uct_c, seed, name
    );
}
#endif // WITH_OPENSPIEL

} // namespace azul 