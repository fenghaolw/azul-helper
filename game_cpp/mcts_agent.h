#pragma once

#ifdef WITH_OPENSPIEL

#include "azul_openspiel.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include <memory>

namespace azul {

class AzulMCTSAgent {
public:
    AzulMCTSAgent(int player_id, int num_simulations = 1000, double uct_c = 1.4, int seed = -1);
    
    // Get the best action for the current state
    [[nodiscard]] auto get_action(const open_spiel::State& state) -> open_spiel::Action;
    
    // Get action probabilities (for training)
    [[nodiscard]] auto get_action_probabilities(const open_spiel::State& state, double temperature = 1.0) -> std::vector<double>;
    
    // Reset the agent (clear search tree)
    void reset();
    
    // Configuration
    void set_num_simulations(int num_simulations) { num_simulations_ = num_simulations; }
    void set_uct_c(double uct_c) { uct_c_ = uct_c; }
    
    [[nodiscard]] auto player_id() const -> int { return player_id_; }

private:
    int player_id_;
    int num_simulations_;
    double uct_c_;
    int seed_;
    std::unique_ptr<open_spiel::algorithms::MCTSBot> mcts_bot_;
    
    void initialize_bot();
};

// Factory function for creating MCTS agents
[[nodiscard]] auto create_mcts_agent(int player_id, int num_simulations = 1000, 
                                     double uct_c = 1.4, int seed = -1) -> std::unique_ptr<AzulMCTSAgent>;

} // namespace azul

#endif // WITH_OPENSPIEL 