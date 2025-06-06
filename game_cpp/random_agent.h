#pragma once

#include <memory>
#include <random>
#include <vector>

#include "open_spiel/spiel.h"

namespace azul {

using ActionType = open_spiel::Action;
using GameStateType = open_spiel::State;

/**
 * Random agent that selects actions uniformly at random from legal actions.
 * Serves as a baseline for comparing other agents.
 * Now supports OpenSpiel game states.
 */
class RandomAgent {
 public:
  explicit RandomAgent(int player_id, int seed = -1);

  // Get a random legal action for the current state
  [[nodiscard]] auto get_action(const GameStateType& state) -> ActionType;

  // Get uniform action probabilities over legal actions
  [[nodiscard]] static auto get_action_probabilities(const GameStateType& state)
      -> std::vector<double>;

  // Reset the agent (reseed random number generator)
  void reset();

  // Configuration
  void set_seed(int seed);

  [[nodiscard]] auto player_id() const -> int { return player_id_; }
  [[nodiscard]] auto seed() const -> int { return seed_; }

 private:
  int player_id_;
  int seed_;
  std::mt19937 rng_;

  void initialize_rng();
};

// Factory function for creating random agents
[[nodiscard]] auto create_random_agent(int player_id, int seed = -1)
    -> std::unique_ptr<RandomAgent>;

}  // namespace azul