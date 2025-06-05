#pragma once

#include <array>
#include <memory>
#include <vector>

#include "azul.h"
#include "open_spiel/spiel.h"

namespace azul {

using ActionType = open_spiel::Action;
using GameStateType = open_spiel::State;

/**
 * Minimax agent for Azul using OpenSpiel's minimax algorithms.
 * Automatically selects the appropriate algorithm based on game type:
 * - AlphaBetaSearch for deterministic games
 * - ExpectiminimaxSearch for stochastic zero-sum games
 */
class MinimaxAgent {
 public:
  MinimaxAgent(int player_id, int depth = 4);

  // Get the best action using minimax search
  [[nodiscard]] auto get_action(const GameStateType& state) -> ActionType;

  // Get action probabilities (minimax is deterministic, so best action gets
  // probability 1.0)
  [[nodiscard]] auto get_action_probabilities(const GameStateType& state)
      -> std::vector<double>;

  // Configuration
  void set_depth(int depth) { depth_ = depth; }

  [[nodiscard]] auto player_id() const -> int { return player_id_; }
  [[nodiscard]] auto depth() const -> int { return depth_; }

  void reset_stats();
  [[nodiscard]] auto nodes_explored() const -> size_t {
    return nodes_explored_;
  }

 private:
  int player_id_;
  int depth_;
  mutable size_t nodes_explored_;

  // Evaluation function for non-terminal states
  [[nodiscard]] auto evaluate_state(const GameStateType& state) const -> double;

  // Helper methods for Azul-specific evaluation
  [[nodiscard]] static auto calculate_round_end_score(
      const open_spiel::azul::AzulState& state, int player) -> double;
  [[nodiscard]] static auto simulate_wall_scoring(
      const std::array<std::array<bool, open_spiel::azul::kWallSize>,
                       open_spiel::azul::kWallSize>& wall,
      int row, int col) -> double;
};

[[nodiscard]] inline auto create_minimax_agent(int player_id, int depth)
    -> std::unique_ptr<MinimaxAgent> {
  return std::make_unique<MinimaxAgent>(player_id, depth);
}
}  // namespace azul