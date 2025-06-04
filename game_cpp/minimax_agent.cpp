#include "minimax_agent.h"
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <random>
#include <iostream>
#include <chrono>

namespace azul {

MinimaxAgent::MinimaxAgent(int player_id, int depth, bool enable_alpha_beta, 
                          bool enable_memoization, int seed)
    : player_id_(player_id), depth_(depth), enable_alpha_beta_(enable_alpha_beta),
      enable_memoization_(enable_memoization), seed_(seed),
      nodes_explored_(0), cache_hits_(0) {
}

Action MinimaxAgent::get_action(const GameState& state) {
    if (state.is_game_over()) {
        throw std::runtime_error("Cannot get action from terminal state");
    }
    
    // Reset statistics for this search
    nodes_explored_ = 0;
    cache_hits_ = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    const auto time_limit = std::chrono::milliseconds(500); // 0.5 second time limit like Python
    
    auto legal_actions = state.get_legal_actions(player_id_);
    
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    if (legal_actions.size() == 1) {
        return legal_actions[0];
    }
    
    // Adaptive depth based on action count and time (like Python version)
    int effective_depth = depth_;
    if (legal_actions.size() > 60) {
        effective_depth = std::max(1, depth_ - 2); // Reduce depth for high branching factor
    } else if (legal_actions.size() > 30) {
        effective_depth = std::max(1, depth_ - 1);
    }
    
    Action best_action = legal_actions[0];
    double best_value = NEGATIVE_INFINITY;
    
    // Evaluate each legal action with time checks
    for (size_t i = 0; i < legal_actions.size(); ++i) {
        const auto& action = legal_actions[i];
        
        // Time-based early termination
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        if (elapsed > time_limit * 0.8) { // Use 80% of time limit
            break;
        }
        
        // Early termination if we've explored too many nodes
        if (nodes_explored_ > 50000) { // Reduced node limit
            break;
        }
        
        if (!state.is_action_legal(action)) {
            continue; // Skip invalid actions quickly
        }
        
        GameState next_state = state.copy(); // Only copy when we know action is valid
        if (!next_state.apply_action(action)) {
            continue; // Skip invalid actions
        }
        
        // Search from the resulting state with time limit
        double value = minimax_with_time_limit(next_state, effective_depth - 1, false, 
                                              NEGATIVE_INFINITY, POSITIVE_INFINITY, start_time, time_limit);
        
        if (value > best_value) {
            best_value = value;
            best_action = action;
        }
    }
    
    return best_action;
}

std::vector<double> MinimaxAgent::get_action_probabilities(const GameState& state) {
    if (state.is_game_over()) {
        return {};
    }
    
    auto legal_actions = state.get_legal_actions(player_id_);
    
    if (legal_actions.empty()) {
        return {};
    }
    
    // Minimax is deterministic, so best action gets probability 1.0
    std::vector<double> probabilities(legal_actions.size(), 0.0);
    
    Action best_action = get_action(state);
    
    // Find the index of the best action
    for (size_t i = 0; i < legal_actions.size(); ++i) {
        if (legal_actions[i] == best_action) {
            probabilities[i] = 1.0;
            break;
        }
    }
    
    return probabilities;
}

void MinimaxAgent::reset() {
    memo_cache_.clear();
    reset_stats();
}

void MinimaxAgent::reset_stats() {
    nodes_explored_ = 0;
    cache_hits_ = 0;
}

double MinimaxAgent::minimax(const GameState& state, int depth, bool maximizing_player,
                            double alpha, double beta) const {
    ++nodes_explored_;
    
    // Early termination if too many nodes explored
    if (nodes_explored_ > 100000) {
        return evaluate_state(state);
    }
    
    // Check memoization cache
    if (enable_memoization_) {
        size_t state_hash = compute_state_hash(state);
        auto cache_it = memo_cache_.find(state_hash);
        if (cache_it != memo_cache_.end() && cache_it->second.first >= depth) {
            ++cache_hits_;
            return cache_it->second.second;
        }
    }
    
    // Terminal node or depth limit reached
    if (state.is_game_over() || depth == 0) {
        double value = evaluate_state(state);
        
        // Cache the result
        if (enable_memoization_) {
            size_t state_hash = compute_state_hash(state);
            memo_cache_[state_hash] = {depth, value};
        }
        
        return value;
    }
    
    double best_value;
    auto legal_actions = state.get_legal_actions();
    
    if (maximizing_player) {
        best_value = NEGATIVE_INFINITY;
        
        for (const auto& action : legal_actions) {
            // Early termination check
            if (nodes_explored_ > 100000) {
                break;
            }
            
            // OPTIMIZATION: Apply action in-place and rollback instead of copying
            // This dramatically reduces allocation overhead
            if (!state.is_action_legal(action)) {
                continue; // Skip invalid actions quickly
            }
            
            GameState next_state = state.copy(); // Only copy when we know action is valid
            if (!next_state.apply_action(action)) {
                continue; // Skip invalid actions
            }
            
            double value = minimax(next_state, depth - 1, !maximizing_player, alpha, beta);
            best_value = std::max(best_value, value);
            
            if (enable_alpha_beta_) {
                alpha = std::max(alpha, best_value);
                if (beta <= alpha) {
                    break; // Beta cutoff
                }
            }
        }
    } else {
        best_value = POSITIVE_INFINITY;
        
        for (const auto& action : legal_actions) {
            // Early termination check
            if (nodes_explored_ > 100000) {
                break;
            }
            
            // OPTIMIZATION: Apply action in-place and rollback instead of copying
            // This dramatically reduces allocation overhead
            if (!state.is_action_legal(action)) {
                continue; // Skip invalid actions quickly
            }
            
            GameState next_state = state.copy(); // Only copy when we know action is valid
            if (!next_state.apply_action(action)) {
                continue; // Skip invalid actions
            }
            
            double value = minimax(next_state, depth - 1, !maximizing_player, alpha, beta);
            best_value = std::min(best_value, value);
            
            if (enable_alpha_beta_) {
                beta = std::min(beta, best_value);
                if (beta <= alpha) {
                    break; // Alpha cutoff
                }
            }
        }
    }
    
    // Cache the result
    if (enable_memoization_) {
        size_t state_hash = compute_state_hash(state);
        memo_cache_[state_hash] = {depth, best_value};
    }
    
    return best_value;
}

double MinimaxAgent::evaluate_state(const GameState& state) const {
    if (state.is_game_over()) {
        // Terminal state evaluation
        auto scores = state.get_scores();
        if (scores.size() <= static_cast<size_t>(player_id_)) {
            return 0.0; // Invalid state
        }
        
        // Return score difference from perspective of our player
        double our_score = static_cast<double>(scores[player_id_]);
        double best_opponent_score = 0.0;
        
        for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
            if (i != player_id_) {
                best_opponent_score = std::max(best_opponent_score, static_cast<double>(scores[i]));
            }
        }
        
        return our_score - best_opponent_score;
    }
    
    // SIMPLIFIED heuristic evaluation - optimized for speed like Python version
    const auto& players = state.players();
    if (players.size() <= static_cast<size_t>(player_id_)) {
        return 0.0; // Invalid state
    }
    
    double our_projected_score = calculate_round_end_score(players[player_id_]);
    
    // Calculate best opponent projected score
    double best_opponent_projected_score = 0.0;
    for (size_t i = 0; i < players.size(); ++i) {
        if (static_cast<int>(i) != player_id_) {
            double opponent_score = calculate_round_end_score(players[i]);
            best_opponent_projected_score = std::max(best_opponent_projected_score, opponent_score);
        }
    }
    
    return our_projected_score - best_opponent_projected_score;
}

double MinimaxAgent::calculate_round_end_score(const PlayerBoard& player_board) const {
    double current_score = static_cast<double>(player_board.score());
    double projected_additional_score = 0.0;
    
    const auto& pattern_lines = player_board.pattern_lines();
    
    // Quick check each pattern line for completion - SIMPLIFIED for performance
    for (int line_idx = 0; line_idx < 5; ++line_idx) {
        const auto& pattern_line = pattern_lines[line_idx];
        int line_capacity = line_idx + 1;
        
        if (pattern_line.tiles().size() == static_cast<size_t>(line_capacity) && 
            !pattern_line.tiles().empty()) {
            // This line is complete - estimate score without expensive simulation
            // Simplified approximation: base score + connection bonus estimate
            double base_score = 1.0;
            
            // Simple bonus estimation (much faster than full wall simulation)
            if (line_idx < 2) {
                base_score += 1.0; // Easier lines bonus
            }
            // Rough connection estimate without expensive wall traversal
            base_score += static_cast<double>(line_idx) * 0.5; // Larger lines may connect more
            
            projected_additional_score += base_score;
        } else if (!pattern_line.tiles().empty()) {
            // Partial line - simple progress bonus
            double progress = static_cast<double>(pattern_line.tiles().size()) / line_capacity;
            projected_additional_score += progress * 0.3; // Reduced weight for speed
        }
    }
    
    // Simplified floor line penalty calculation
    const auto& floor_line = player_board.floor_line();
    size_t floor_count = floor_line.size();
    if (floor_count > 0) {
        // Simple linear penalty approximation (much faster than lookup)
        double penalty = static_cast<double>(floor_count) * 1.5; // Average penalty per tile
        projected_additional_score -= penalty;
    }
    
    return current_score + projected_additional_score;
}

size_t MinimaxAgent::compute_state_hash(const GameState& state) const {
    // Use the state vector to compute a hash
    auto state_vector = state.get_state_vector();
    
    std::hash<float> hasher;
    size_t hash_value = 0;
    
    for (size_t i = 0; i < state_vector.size(); ++i) {
        // Combine hashes using a simple mixing function
        hash_value ^= hasher(state_vector[i]) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
    }
    
    // Include current player in hash
    hash_value ^= std::hash<int>{}(state.current_player()) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
    
    return hash_value;
}

bool MinimaxAgent::is_maximizing_player(const GameState& state) const {
    return state.current_player() == player_id_;
}

std::unique_ptr<MinimaxAgent> create_minimax_agent(int player_id, int depth, 
                                                  bool enable_alpha_beta,
                                                  bool enable_memoization,
                                                  int seed) {
    return std::make_unique<MinimaxAgent>(player_id, depth, enable_alpha_beta, 
                                         enable_memoization, seed);
}

double MinimaxAgent::minimax_with_time_limit(const GameState& state, int depth, bool maximizing_player,
                            double alpha, double beta, 
                            std::chrono::high_resolution_clock::time_point start_time,
                            std::chrono::milliseconds time_limit) const {
    ++nodes_explored_;
    
    // Time-based early termination
    auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
    if (elapsed > time_limit) {
        return evaluate_state(state); // Emergency evaluation if time is up
    }
    
    // Early termination if too many nodes explored
    if (nodes_explored_ > 50000) {
        return evaluate_state(state);
    }
    
    // Check memoization cache
    if (enable_memoization_) {
        size_t state_hash = compute_state_hash(state);
        auto cache_it = memo_cache_.find(state_hash);
        if (cache_it != memo_cache_.end() && cache_it->second.first >= depth) {
            ++cache_hits_;
            return cache_it->second.second;
        }
    }
    
    // Terminal node or depth limit reached
    if (state.is_game_over() || depth == 0) {
        double value = evaluate_state(state);
        
        // Cache the result
        if (enable_memoization_) {
            size_t state_hash = compute_state_hash(state);
            memo_cache_[state_hash] = {depth, value};
        }
        
        return value;
    }
    
    double best_value;
    auto legal_actions = state.get_legal_actions();
    
    if (maximizing_player) {
        best_value = NEGATIVE_INFINITY;
        
        for (const auto& action : legal_actions) {
            // Time check for each action
            auto elapsed_inner = std::chrono::high_resolution_clock::now() - start_time;
            if (elapsed_inner > time_limit * 0.9) {
                break;
            }
            
            // Early termination check
            if (nodes_explored_ > 50000) {
                break;
            }
            
            if (!state.is_action_legal(action)) {
                continue; // Skip invalid actions quickly
            }
            
            GameState next_state = state.copy(); // Only copy when we know action is valid
            if (!next_state.apply_action(action)) {
                continue; // Skip invalid actions
            }
            
            double value = minimax_with_time_limit(next_state, depth - 1, !maximizing_player, 
                                                  alpha, beta, start_time, time_limit);
            best_value = std::max(best_value, value);
            
            if (enable_alpha_beta_) {
                alpha = std::max(alpha, best_value);
                if (beta <= alpha) {
                    break; // Beta cutoff
                }
            }
        }
    } else {
        best_value = POSITIVE_INFINITY;
        
        for (const auto& action : legal_actions) {
            // Time check for each action
            auto elapsed_inner = std::chrono::high_resolution_clock::now() - start_time;
            if (elapsed_inner > time_limit * 0.9) {
                break;
            }
            
            // Early termination check
            if (nodes_explored_ > 50000) {
                break;
            }
            
            if (!state.is_action_legal(action)) {
                continue; // Skip invalid actions quickly
            }
            
            GameState next_state = state.copy(); // Only copy when we know action is valid
            if (!next_state.apply_action(action)) {
                continue; // Skip invalid actions
            }
            
            double value = minimax_with_time_limit(next_state, depth - 1, !maximizing_player, 
                                                  alpha, beta, start_time, time_limit);
            best_value = std::min(best_value, value);
            
            if (enable_alpha_beta_) {
                beta = std::min(beta, best_value);
                if (beta <= alpha) {
                    break; // Alpha cutoff
                }
            }
        }
    }
    
    // Cache the result
    if (enable_memoization_) {
        size_t state_hash = compute_state_hash(state);
        memo_cache_[state_hash] = {depth, best_value};
    }
    
    return best_value;
}

} // namespace azul 