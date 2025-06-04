#include "random_agent.h"
#include <algorithm>
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

Action RandomAgent::get_action(const GameState& state) {
    if (state.is_game_over()) {
        throw std::runtime_error("Cannot get action from terminal state");
    }
    
    // Get legal actions for the current player
    auto legal_actions = state.get_legal_actions(player_id_);
    
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    // Select random action
    std::uniform_int_distribution<size_t> dist(0, legal_actions.size() - 1);
    size_t action_index = dist(rng_);
    
    return legal_actions[action_index];
}

std::vector<double> RandomAgent::get_action_probabilities(const GameState& state) {
    if (state.is_game_over()) {
        return {};
    }
    
    // Get legal actions for the current player
    auto legal_actions = state.get_legal_actions(player_id_);
    
    if (legal_actions.empty()) {
        return {};
    }
    
    // Return uniform probabilities
    double uniform_prob = 1.0 / static_cast<double>(legal_actions.size());
    return std::vector<double>(legal_actions.size(), uniform_prob);
}

void RandomAgent::reset() {
    initialize_rng();
}

void RandomAgent::set_seed(int seed) {
    seed_ = seed;
    initialize_rng();
}

std::unique_ptr<RandomAgent> create_random_agent(int player_id, int seed) {
    return std::make_unique<RandomAgent>(player_id, seed);
}

} // namespace azul 