#include "mcts_agent.h"
#include "azul.h"
#include <iostream>
#include <sstream>
#include <random>

namespace azul {

AzulMCTSAgent::AzulMCTSAgent(int player_id, int num_simulations, double uct_c, int seed)
    : player_id_(player_id), num_simulations_(num_simulations), uct_c_(uct_c), seed_(seed),
      nodes_explored_(0), total_thinking_time_(0.0), moves_played_(0) {
    
    // Create name
    std::ostringstream oss;
    oss << "MCTS_" << num_simulations << "_" << uct_c << "_P" << player_id;
    name_ = oss.str();
    
    // Initialize MCTS components
    initialize_mcts();
}

void AzulMCTSAgent::initialize_mcts() {
    // Load the Azul game
    game_ = open_spiel::LoadGame("azul");
    if (!game_) {
        throw std::runtime_error("Failed to load Azul game for MCTS agent");
    }
    
    // Create evaluator (random rollout evaluator)
    evaluator_ = std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
        1,      // num_rollouts
        seed_   // seed
    );
    
    // Create MCTS bot
    mcts_bot_ = std::make_unique<open_spiel::algorithms::MCTSBot>(
        *game_,
        evaluator_,
        uct_c_,              // UCT exploration constant
        num_simulations_,    // Max simulations per move
        1000,               // Max memory MB
        false,              // Don't solve for exact values
        seed_,              // Random seed
        false               // Verbose (set to false for cleaner output)
    );
}

open_spiel::Action AzulMCTSAgent::get_action(const open_spiel::State& state) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get action from MCTS bot
    open_spiel::Action action = mcts_bot_->Step(state);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Update performance statistics
    total_thinking_time_ += duration.count() / 1000.0; // Convert to seconds
    moves_played_++;
    
    // Estimate nodes explored (this is an approximation since OpenSpiel doesn't expose the exact count)
    nodes_explored_ += num_simulations_;
    
    return action;
}

std::vector<double> AzulMCTSAgent::get_action_probabilities(const open_spiel::State& state, double temperature) {
    auto legal_actions = state.LegalActions();
    std::vector<double> probabilities(legal_actions.size(), 0.0);
    
    // For simplicity, we'll use a uniform distribution over legal actions
    // In a more sophisticated implementation, we could use the MCTS visit counts
    if (!legal_actions.empty()) {
        double prob = 1.0 / legal_actions.size();
        std::fill(probabilities.begin(), probabilities.end(), prob);
    }
    
    return probabilities;
}

void AzulMCTSAgent::reset() {
    // Reset performance tracking
    reset_stats();
    
    // Reinitialize MCTS components
    initialize_mcts();
}

void AzulMCTSAgent::reset_stats() {
    nodes_explored_ = 0;
    total_thinking_time_ = 0.0;
    moves_played_ = 0;
}

std::unique_ptr<AzulMCTSAgent> create_mcts_agent(int player_id, int num_simulations, 
                                                double uct_c, int seed) {
    return std::make_unique<AzulMCTSAgent>(player_id, num_simulations, uct_c, seed);
}

} // namespace azul 