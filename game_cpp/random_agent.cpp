#include "random_agent.h"

#include <stdexcept>

namespace azul {

RandomAgent::RandomAgent(int player_id, int seed)
    : player_id_(player_id), seed_(seed) {
  initialize_rng();
}

void RandomAgent::initialize_rng() {
  if (seed_ == -1) {
    // Use random device for seeding if no seed provided
    std::random_device rd;
    rng_.seed(rd());
  } else {
    rng_.seed(static_cast<std::mt19937::result_type>(seed_));
  }
}

auto RandomAgent::get_action(const GameStateType& state) -> ActionType {
  if (state.IsTerminal()) {
    throw std::runtime_error("Cannot get action from terminal state");
  }

  // Get legal actions from OpenSpiel state
  auto legal_actions = state.LegalActions();

  if (legal_actions.empty()) {
    throw std::runtime_error("No legal actions available");
  }

  // Select random action
  std::uniform_int_distribution<size_t> dist(0, legal_actions.size() - 1);
  size_t action_index = dist(rng_);

  return legal_actions[action_index];
}

auto RandomAgent::get_action_probabilities(const GameStateType& state)
    -> std::vector<double> {
  if (state.IsTerminal()) {
    return {};
  }

  // Get legal actions from OpenSpiel state
  auto legal_actions = state.LegalActions();

  if (legal_actions.empty()) {
    return {};
  }

  // Return uniform probabilities
  double uniform_prob = 1.0 / static_cast<double>(legal_actions.size());
  std::vector<double> probabilities(legal_actions.size(), uniform_prob);
  return probabilities;
}

void RandomAgent::reset() {
  initialize_rng();
}

void RandomAgent::set_seed(int seed) {
  seed_ = seed;
  initialize_rng();
}

auto create_random_agent(int player_id, int seed)
    -> std::unique_ptr<RandomAgent> {
  return std::make_unique<RandomAgent>(player_id, seed);
}

}  // namespace azul
