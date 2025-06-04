#include "minimax_agent.h"
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <random>
#include <iostream>
#include <chrono>

namespace azul {

MinimaxAgent::MinimaxAgent(int player_id, int depth, bool enable_alpha_beta, int seed)
    : player_id_(player_id), depth_(depth), enable_alpha_beta_(enable_alpha_beta),
      nodes_explored_(0) {
    // Suppress unused parameter warning
    (void)seed;
}

Action MinimaxAgent::get_action(const GameState& state) {
    if (state.is_game_over()) {
        throw std::runtime_error("Cannot get action from terminal state");
    }
    
    // Reset statistics for this search
    nodes_explored_ = 0;
    
    auto legal_actions = state.get_legal_actions();
    
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    if (legal_actions.size() == 1) {
        return legal_actions[0];
    }
    
    // Apply action filtering to reduce branching factor
    auto filtered_actions = filter_obviously_bad_moves(legal_actions, state);
    
    // Limit search nodes to prevent timeout (adjust depth if too many actions)
    size_t estimated_nodes = 1;
    for (int d = 0; d < depth_; ++d) {
        estimated_nodes *= filtered_actions.size();
        if (estimated_nodes > 50000) { // Node limit
            // Reduce depth dynamically if estimated node count is too high
            depth_ = std::max(1, d);
            break;
        }
    }
    
    Action best_action = filtered_actions[0];
    double best_value = NEGATIVE_INFINITY;
    
    // Order actions using heuristics for first iteration
    auto ordered_actions = order_actions(filtered_actions, state, false);
    
    // Evaluate each legal action
    for (const auto& action : ordered_actions) {
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
        
        // Search from the resulting state
        double value = minimax(next_state, depth_ - 1, false);
        
        // Store score for move ordering in future iterations
        move_scores_[action_key(action)] = value;
        
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
    move_scores_.clear();
    reset_stats();
}

void MinimaxAgent::reset_stats() {
    nodes_explored_ = 0;
}

double MinimaxAgent::minimax(const GameState& state, int depth, bool maximizing_player,
                            double alpha, double beta) const {
    ++nodes_explored_;
    
    // Early termination if too many nodes explored
    if (nodes_explored_ > 100000) {
        return evaluate_state(state);
    }
    
    // Terminal node or depth limit reached
    if (state.is_game_over() || depth == 0) {
        double value = evaluate_state(state);
        
        return value;
    }
    
    double best_value;
    auto legal_actions = state.get_legal_actions();
    
    // Apply move ordering and filtering
    auto filtered_actions = filter_obviously_bad_moves(legal_actions, state);
    auto ordered_actions = order_actions(filtered_actions, state, depth < depth_ - 1);
    
    if (maximizing_player) {
        best_value = NEGATIVE_INFINITY;
        
        for (const auto& action : ordered_actions) {
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
        
        for (const auto& action : ordered_actions) {
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

bool MinimaxAgent::is_maximizing_player(const GameState& state) const {
    return state.current_player() == player_id_;
}

std::vector<Action> MinimaxAgent::order_actions(const std::vector<Action>& actions, const GameState& state, bool use_previous_scores) const {
    if (use_previous_scores && !move_scores_.empty()) {
        // Sort by previous iteration scores (highest first)
        auto ordered = actions;
        std::sort(ordered.begin(), ordered.end(), [&](const Action& a, const Action& b) {
            std::string key_a = action_key(a);
            std::string key_b = action_key(b);
            double score_a = move_scores_.count(key_a) ? move_scores_.at(key_a) : 0.0;
            double score_b = move_scores_.count(key_b) ? move_scores_.at(key_b) : 0.0;
            return score_a > score_b;
        });
        return ordered;
    } else {
        // Use heuristic ordering when no previous scores available
        auto ordered = actions;
        std::sort(ordered.begin(), ordered.end(), [&](const Action& a, const Action& b) {
            return evaluate_move_priority(a, state) > evaluate_move_priority(b, state);
        });
        return ordered;
    }
}

double MinimaxAgent::evaluate_move_priority(const Action& action, const GameState& state) const {
    double score = 0.0;
    
    // Prefer completing pattern lines (not floor line)
    if (action.destination() >= 0) {
        score += 10.0;
        
        // Prefer shorter lines (easier to complete)
        score += (5 - action.destination()) * 2.0;
        
        // Check if this move would complete a pattern line
        const auto& players = state.players();
        if (static_cast<size_t>(player_id_) < players.size()) {
            const auto& player = players[player_id_];
            const auto& pattern_lines = player.pattern_lines();
            
            if (action.destination() < static_cast<int>(pattern_lines.size())) {
                const auto& pattern_line = pattern_lines[action.destination()];
                int line_capacity = action.destination() + 1;
                
                // Bonus for completing the line
                if (static_cast<int>(pattern_line.tiles().size()) == line_capacity - 1) {
                    score += 15.0; // High priority for completing lines
                }
                
                // Bonus for continuing existing work on a line
                if (!pattern_line.tiles().empty() && 
                    pattern_line.color().has_value() && 
                    pattern_line.color().value() == action.color()) {
                    score += 5.0;
                }
                
                // Penalty for wasting tiles (placing on already full line)
                if (static_cast<int>(pattern_line.tiles().size()) >= line_capacity) {
                    score -= 20.0;
                }
            }
        }
        
        // Prefer taking from center (first player marker advantage)
        if (action.source() == -1) {
            score += 3.0;
        }
        
        // Estimate tiles taken (prefer taking more tiles)
        // This is approximate since we'd need to simulate the action
        if (action.source() >= 0) {
            // Factory typically has 4 tiles, estimate we get 1-4 of our color
            score += 2.0; // Base bonus for factory
        } else {
            // Center can have many tiles
            score += 1.0; // Smaller bonus since center varies more
        }
        
    } else {
        // Floor line moves are generally bad, but sometimes necessary
        score -= 5.0;
        
        // But if we're taking first player marker, it might be worth it
        if (action.source() == -1) {
            score += 2.0; // Mitigate penalty slightly
        }
    }
    
    return score;
}

std::string MinimaxAgent::action_key(const Action& action) const {
    return std::to_string(action.source()) + "_" + 
           std::to_string(static_cast<int>(action.color())) + "_" + 
           std::to_string(action.destination());
}

std::vector<Action> MinimaxAgent::filter_obviously_bad_moves(const std::vector<Action>& actions, const GameState& state) const {
    std::vector<Action> filtered;
    filtered.reserve(actions.size());
    
    // Separate actions by type for better filtering
    std::vector<Action> good_actions;
    std::vector<Action> floor_actions;
    std::vector<Action> mediocre_actions;
    
    for (const auto& action : actions) {
        if (is_obviously_bad_move(action, state)) {
            continue; // Skip truly bad moves
        }
        
        if (action.destination() == -1) {
            floor_actions.push_back(action);
        } else {
            double priority = evaluate_move_priority(action, state);
            if (priority >= 10.0) { // High priority moves
                good_actions.push_back(action);
            } else {
                mediocre_actions.push_back(action);
            }
        }
    }
    
    // Prefer good actions, but keep some alternatives
    filtered = good_actions;
    
    // Add some mediocre actions if we don't have many good ones
    if (filtered.size() < 5 && !mediocre_actions.empty()) {
        // Sort mediocre actions and take the best ones
        std::sort(mediocre_actions.begin(), mediocre_actions.end(), [&](const Action& a, const Action& b) {
            return evaluate_move_priority(a, state) > evaluate_move_priority(b, state);
        });
        
        size_t add_count = std::min(mediocre_actions.size(), size_t(8 - filtered.size()));
        filtered.insert(filtered.end(), mediocre_actions.begin(), mediocre_actions.begin() + add_count);
    }
    
    // Add floor actions only if we have very few alternatives or they're necessary
    if (filtered.size() < 3 && !floor_actions.empty()) {
        // Sort floor actions and take the best one or two
        std::sort(floor_actions.begin(), floor_actions.end(), [&](const Action& a, const Action& b) {
            return evaluate_move_priority(a, state) > evaluate_move_priority(b, state);
        });
        
        size_t add_count = std::min(floor_actions.size(), size_t(2));
        filtered.insert(filtered.end(), floor_actions.begin(), floor_actions.begin() + add_count);
    }
    
    // Ensure we always have at least some moves
    if (filtered.empty()) {
        // Emergency fallback - return top 5 moves by priority
        auto ordered = order_actions(actions, state, false);
        size_t keep_count = std::min(actions.size(), size_t(5));
        filtered.assign(ordered.begin(), ordered.begin() + keep_count);
    }
    
    // Limit total actions to prevent explosion
    if (filtered.size() > 15) {
        auto ordered = order_actions(filtered, state, false);
        filtered.assign(ordered.begin(), ordered.begin() + 15);
    }
    
    return filtered;
}

bool MinimaxAgent::is_obviously_bad_move(const Action& action, const GameState& state) const {
    // Floor line moves are usually bad (but sometimes necessary)
    if (action.destination() == -1) {
        // Only filter floor moves if there are good alternatives
        return false; // For now, don't filter floor moves
    }
    
    // Check if this would waste tiles by placing on an incompatible line
    const auto& players = state.players();
    if (static_cast<size_t>(player_id_) < players.size()) {
        const auto& player = players[player_id_];
        const auto& pattern_lines = player.pattern_lines();
        
        if (action.destination() < static_cast<int>(pattern_lines.size())) {
            const auto& pattern_line = pattern_lines[action.destination()];
            
            // Bad: placing on line with different color
            if (!pattern_line.tiles().empty() && 
                pattern_line.color().has_value() && 
                pattern_line.color().value() != action.color()) {
                return true;
            }
            
            // Bad: placing on already full line
            int line_capacity = action.destination() + 1;
            if (static_cast<int>(pattern_line.tiles().size()) >= line_capacity) {
                return true;
            }
            
            // Bad: can't place this color on wall
            if (!player.wall().can_place_tile(action.destination(), action.color())) {
                return true;
            }
        }
    }
    
    return false;
}

std::unique_ptr<MinimaxAgent> create_minimax_agent(int player_id, int depth, 
                                                  bool enable_alpha_beta,
                                                  int seed) {
    return std::make_unique<MinimaxAgent>(player_id, depth, enable_alpha_beta, seed);
}

} // namespace azul 