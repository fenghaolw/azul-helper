#ifdef WITH_OPENSPIEL

#include "mcts_agent.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel_utils.h"
#include <random>

namespace azul {

AzulMCTSAgent::AzulMCTSAgent(int player_id, int num_simulations, double uct_c, int seed)
    : player_id_(player_id), num_simulations_(num_simulations), uct_c_(uct_c), seed_(seed) {
    initialize_bot();
}

void AzulMCTSAgent::initialize_bot() {
    // Create MCTS evaluator
    auto evaluator = std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(1, seed_);
    
    // Create MCTS bot
    mcts_bot_ = std::make_unique<open_spiel::algorithms::MCTSBot>(
        player_id_,
        uct_c_,
        num_simulations_,
        /*max_memory_mb=*/1000,
        evaluator,
        /*random_state=*/seed_ != -1 ? std::make_unique<std::mt19937>(seed_) : nullptr,
        /*child_selection_policy=*/open_spiel::algorithms::ChildSelectionPolicy::kPuct,
        /*verbose=*/false,
        /*dont_return_chance_node=*/true
    );
}

open_spiel::Action AzulMCTSAgent::get_action(const open_spiel::State& state) {
    if (state.IsTerminal()) {
        return open_spiel::kInvalidAction;
    }
    
    return mcts_bot_->Step(state);
}

std::vector<double> AzulMCTSAgent::get_action_probabilities(const open_spiel::State& state, double temperature) {
    if (state.IsTerminal()) {
        return {};
    }
    
    // Run MCTS to get visit counts
    auto legal_actions = state.LegalActions();
    if (legal_actions.empty()) {
        return {};
    }
    
    // Get the search tree after running MCTS
    mcts_bot_->Step(state); // This runs the search
    
    // For now, return uniform probabilities
    // In a full implementation, you'd extract visit counts from the MCTS tree
    std::vector<double> probabilities(180, 0.0); // All possible actions
    double uniform_prob = 1.0 / legal_actions.size();
    
    for (auto action : legal_actions) {
        probabilities[action] = uniform_prob;
    }
    
    // Apply temperature
    if (temperature != 1.0 && temperature > 0.0) {
        for (auto& prob : probabilities) {
            if (prob > 0.0) {
                prob = std::pow(prob, 1.0 / temperature);
            }
        }
        
        // Renormalize
        double sum = 0.0;
        for (auto prob : probabilities) {
            sum += prob;
        }
        if (sum > 0.0) {
            for (auto& prob : probabilities) {
                prob /= sum;
            }
        }
    }
    
    return probabilities;
}

void AzulMCTSAgent::reset() {
    // Reinitialize the bot to clear the search tree
    initialize_bot();
}

std::unique_ptr<AzulMCTSAgent> create_mcts_agent(int player_id, int num_simulations, 
                                                 double uct_c, int seed) {
    return std::make_unique<AzulMCTSAgent>(player_id, num_simulations, uct_c, seed);
}

} // namespace azul

#endif // WITH_OPENSPIEL 