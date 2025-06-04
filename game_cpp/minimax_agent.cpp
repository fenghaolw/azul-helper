#include "minimax_agent.h"
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <random>

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
    
    auto legal_actions = state.get_legal_actions(player_id_);
    
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    if (legal_actions.size() == 1) {
        return legal_actions[0];
    }
    
    // Limit search nodes to prevent timeout (adjust depth if too many actions)
    size_t estimated_nodes = 1;
    for (int d = 0; d < depth_; ++d) {
        estimated_nodes *= legal_actions.size();
        if (estimated_nodes > 50000) { // Node limit
            // Reduce depth dynamically if estimated node count is too high
            depth_ = std::max(1, d);
            break;
        }
    }
    
    Action best_action = legal_actions[0];
    double best_value = NEGATIVE_INFINITY;
    
    // Evaluate each legal action
    for (const auto& action : legal_actions) {
        // Early termination if we've explored too many nodes
        if (nodes_explored_ > 100000) {
            break;
        }
        
        GameState next_state = state.copy();
        if (!next_state.apply_action(action)) {
            continue; // Skip invalid actions
        }
        
        // Search from the resulting state
        double value = minimax(next_state, depth_ - 1, false);
        
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
            
            GameState next_state = state.copy();
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
            
            GameState next_state = state.copy();
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
    
    // Advanced heuristic evaluation - simulate round ending immediately
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
    const auto& wall = player_board.wall();
    
    // Check each pattern line for completion potential
    for (int line_idx = 0; line_idx < 5; ++line_idx) {
        const auto& pattern_line = pattern_lines[line_idx];
        int line_capacity = line_idx + 1;
        
        if (pattern_line.tiles().size() == static_cast<size_t>(line_capacity) && 
            !pattern_line.tiles().empty()) {
            // This line is complete, simulate placing on wall
            auto wall_tile = pattern_line.get_wall_tile();
            if (wall_tile.has_value()) {
                TileColor color = wall_tile->color();
                
                // Check if we can place this tile
                if (wall.can_place_tile(line_idx, color)) {
                    // Simulate wall scoring
                    double wall_points = simulate_wall_scoring(wall, line_idx, color);
                    projected_additional_score += wall_points;
                    
                    // Bonus for completing pattern lines (strategic value)
                    projected_additional_score += 1.0;
                    
                    // Extra bonus for shorter lines (easier to complete)
                    if (line_idx < 2) {
                        projected_additional_score += 0.5;
                    }
                }
            }
        } else if (!pattern_line.tiles().empty()) {
            // Partial line - give small credit for progress
            double progress = static_cast<double>(pattern_line.tiles().size()) / line_capacity;
            projected_additional_score += progress * 0.5;
        }
    }
    
    // Subtract floor line penalties
    const auto& floor_line = player_board.floor_line();
    double floor_penalty = calculate_floor_penalty(floor_line.size());
    projected_additional_score -= floor_penalty;
    
    // Bonus for wall pattern completion potential
    projected_additional_score += evaluate_wall_completion_bonus(wall);
    
    return current_score + projected_additional_score;
}

double MinimaxAgent::simulate_wall_scoring(const Wall& wall, int row, TileColor color) const {
    // Find the column for this color on this row
    int col = get_wall_column_for_color(row, color);
    if (col == -1) {
        return 0.0; // Invalid placement
    }
    
    // Check if already filled
    if (wall.is_filled(row, col)) {
        return 0.0;
    }
    
    double score = 1.0; // Base score for the tile
    
    // Check horizontal connections
    int horizontal_length = 1;
    // Check left
    for (int c = col - 1; c >= 0; --c) {
        if (wall.is_filled(row, c)) {
            horizontal_length++;
        } else {
            break;
        }
    }
    // Check right
    for (int c = col + 1; c < 5; ++c) {
        if (wall.is_filled(row, c)) {
            horizontal_length++;
        } else {
            break;
        }
    }
    
    // Check vertical connections
    int vertical_length = 1;
    // Check up
    for (int r = row - 1; r >= 0; --r) {
        if (wall.is_filled(r, col)) {
            vertical_length++;
        } else {
            break;
        }
    }
    // Check down
    for (int r = row + 1; r < 5; ++r) {
        if (wall.is_filled(r, col)) {
            vertical_length++;
        } else {
            break;
        }
    }
    
    // Scoring logic: if connected to other tiles, use the larger connection
    if (horizontal_length > 1 || vertical_length > 1) {
        score = std::max(horizontal_length, vertical_length);
        // If connected both horizontally and vertically, add both
        if (horizontal_length > 1 && vertical_length > 1) {
            score = horizontal_length + vertical_length;
        }
    }
    
    return score;
}

double MinimaxAgent::calculate_floor_penalty(size_t floor_tiles) const {
    // Azul floor line penalty structure: [1, 1, 2, 2, 2, 3, 3]
    static const std::vector<int> penalties = {1, 1, 2, 2, 2, 3, 3};
    
    double total_penalty = 0.0;
    for (size_t i = 0; i < floor_tiles && i < penalties.size(); ++i) {
        total_penalty += penalties[i];
    }
    
    return total_penalty;
}

double MinimaxAgent::evaluate_wall_completion_bonus(const Wall& wall) const {
    double bonus = 0.0;
    
    // Bonus for completed rows
    for (int row = 0; row < 5; ++row) {
        if (wall.is_row_complete(row)) {
            bonus += 2.0; // Row completion bonus in Azul
        } else {
            // Partial bonus for progress towards row completion
            int filled_count = 0;
            for (int col = 0; col < 5; ++col) {
                if (wall.is_filled(row, col)) {
                    filled_count++;
                }
            }
            bonus += (filled_count / 5.0) * 0.5;
        }
    }
    
    // Bonus for completed columns
    for (int col = 0; col < 5; ++col) {
        if (wall.is_column_complete(col)) {
            bonus += 7.0; // Column completion bonus in Azul
        } else {
            // Partial bonus for progress towards column completion
            int filled_count = 0;
            for (int row = 0; row < 5; ++row) {
                if (wall.is_filled(row, col)) {
                    filled_count++;
                }
            }
            bonus += (filled_count / 5.0) * 1.0;
        }
    }
    
    // Bonus for completed colors
    for (int color_idx = 0; color_idx < 5; ++color_idx) {
        TileColor color = static_cast<TileColor>(color_idx);
        if (wall.is_color_complete(color)) {
            bonus += 10.0; // Color completion bonus in Azul
        }
    }
    
    return bonus;
}

int MinimaxAgent::get_wall_column_for_color(int row, TileColor color) const {
    // Azul wall pattern - each row has colors in specific positions
    // This is a simplified version - in real Azul the pattern is:
    // Row 0: Blue(0), Yellow(1), Red(2), Black(3), White(4)
    // Row 1: White(0), Blue(1), Yellow(2), Red(3), Black(4)
    // Row 2: Black(0), White(1), Blue(2), Yellow(3), Red(4)
    // Row 3: Red(0), Black(1), White(2), Blue(3), Yellow(4)
    // Row 4: Yellow(0), Red(1), Black(2), White(3), Blue(4)
    
    static const int wall_pattern[5][5] = {
        {0, 1, 2, 3, 4}, // BLUE, YELLOW, RED, BLACK, WHITE
        {4, 0, 1, 2, 3}, // WHITE, BLUE, YELLOW, RED, BLACK
        {3, 4, 0, 1, 2}, // BLACK, WHITE, BLUE, YELLOW, RED
        {2, 3, 4, 0, 1}, // RED, BLACK, WHITE, BLUE, YELLOW
        {1, 2, 3, 4, 0}  // YELLOW, RED, BLACK, WHITE, BLUE
    };
    
    int color_index = static_cast<int>(color);
    if (row >= 0 && row < 5 && color_index >= 0 && color_index < 5) {
        return wall_pattern[row][color_index];
    }
    
    return -1; // Invalid
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

} // namespace azul 