#pragma once

#include "action.h"
#include "game_state.h"
#include "player_board.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <limits>

namespace azul {

/**
 * Minimax agent with alpha-beta pruning for Azul.
 * Uses a simple heuristic evaluation function based on score differences.
 */
class MinimaxAgent {
public:
    MinimaxAgent(int player_id, int depth = 4, bool enable_alpha_beta = true, 
                 bool enable_memoization = true, int seed = -1);
    
    // Get the best action using minimax search
    [[nodiscard]] auto get_action(const GameState& state) -> Action;
    
    // Get action probabilities (minimax is deterministic, so best action gets probability 1.0)
    [[nodiscard]] auto get_action_probabilities(const GameState& state) -> std::vector<double>;
    
    // Reset the agent (clear memoization cache)
    void reset();
    
    // Configuration
    void set_depth(int depth) { depth_ = depth; }
    void set_alpha_beta(bool enable) { enable_alpha_beta_ = enable; }
    void set_memoization(bool enable) { enable_memoization_ = enable; }
    
    [[nodiscard]] auto player_id() const -> int { return player_id_; }
    [[nodiscard]] auto depth() const -> int { return depth_; }
    [[nodiscard]] auto alpha_beta_enabled() const -> bool { return enable_alpha_beta_; }
    [[nodiscard]] auto memoization_enabled() const -> bool { return enable_memoization_; }
    
    // Performance statistics
    [[nodiscard]] auto nodes_explored() const -> size_t { return nodes_explored_; }
    [[nodiscard]] auto cache_hits() const -> size_t { return cache_hits_; }
    void reset_stats();

private:
    int player_id_;
    int depth_;
    bool enable_alpha_beta_;
    bool enable_memoization_;
    int seed_;
    
    // Performance tracking
    mutable size_t nodes_explored_;
    mutable size_t cache_hits_;
    
    // Memoization cache: state hash -> (depth, value)
    mutable std::unordered_map<size_t, std::pair<int, double>> memo_cache_;
    
    static constexpr double POSITIVE_INFINITY = std::numeric_limits<double>::max();
    static constexpr double NEGATIVE_INFINITY = std::numeric_limits<double>::lowest();
    
    // Core minimax algorithm
    [[nodiscard]] auto minimax(const GameState& state, int depth, bool maximizing_player,
                              double alpha = NEGATIVE_INFINITY, 
                              double beta = POSITIVE_INFINITY) const -> double;
    
    // Evaluation function for non-terminal states
    [[nodiscard]] auto evaluate_state(const GameState& state) const -> double;
    
    // Advanced evaluation helper methods
    [[nodiscard]] auto calculate_round_end_score(const PlayerBoard& player_board) const -> double;
    [[nodiscard]] auto simulate_wall_scoring(const Wall& wall, int row, TileColor color) const -> double;
    [[nodiscard]] auto calculate_floor_penalty(size_t floor_tiles) const -> double;
    [[nodiscard]] auto evaluate_wall_completion_bonus(const Wall& wall) const -> double;
    [[nodiscard]] auto get_wall_column_for_color(int row, TileColor color) const -> int;
    
    // Helper methods
    [[nodiscard]] auto compute_state_hash(const GameState& state) const -> size_t;
    [[nodiscard]] auto is_maximizing_player(const GameState& state) const -> bool;
};

// Factory function for creating minimax agents
[[nodiscard]] auto create_minimax_agent(int player_id, int depth = 4, 
                                       bool enable_alpha_beta = true,
                                       bool enable_memoization = true,
                                       int seed = -1) -> std::unique_ptr<MinimaxAgent>;

} // namespace azul 