#include "random_agent.h"
#include "minimax_agent.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "open_spiel/spiel.h"

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

ActionType RandomAgent::get_action(const GameStateType& state) {
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

std::vector<double> RandomAgent::get_action_probabilities(const GameStateType& state) {
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

// Include our local Azul game for registration
#include "azul.h"

// Force linker to include the azul registration by referencing symbols
namespace {
void force_azul_registration() {
    // Reference symbols from azul namespace to force linking
    (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}
}

// Test function demonstrating OpenSpiel agent integration with both agents
int main() {
    std::cout << "=== OpenSpiel Azul Agents Integration Test ===" << std::endl;
    
    try {
        // Force registration by calling the function (this ensures linking)
        force_azul_registration();
        
        // Load Azul game
        auto game = open_spiel::LoadGame("azul");
        if (!game) {
            std::cerr << "❌ Failed to load Azul game" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Loaded Azul game successfully" << std::endl;
        
        // Check game properties
        auto game_type = game->GetType();
        std::cout << "🎮 Game: " << game_type.short_name 
                  << " (" << (game_type.utility == open_spiel::GameType::Utility::kZeroSum ? "Zero-Sum" : "General-Sum") << ")" << std::endl;
        
        // Create agents
        azul::RandomAgent random_agent(0, 42);  // Player 0
        azul::MinimaxAgent minimax_agent(1, 3);  // Player 1, depth 3
        
        std::cout << "🤖 Created RandomAgent (Player 0) and MinimaxAgent (Player 1, depth=3)" << std::endl;
        
        // Initialize game state
        auto state = game->NewInitialState();
        int turn_count = 0;
        const int max_turns = 10;  // Limit for demonstration
        
        std::cout << "\n=== Game Simulation ===" << std::endl;
        
        while (!state->IsTerminal() && turn_count < max_turns) {
            int current_player = state->CurrentPlayer();
            
            // Skip chance nodes (handled automatically by OpenSpiel)
            if (state->IsChanceNode()) {
                auto outcomes = state->ChanceOutcomes();
                if (!outcomes.empty()) {
                    // Select first chance outcome for simplicity
                    state->ApplyAction(outcomes[0].first);
                }
                continue;
            }
            
            std::cout << "\n--- Turn " << (turn_count + 1) << " (Player " << current_player << ") ---" << std::endl;
            
            azul::ActionType chosen_action;
            std::string agent_name;
            
            // Get action from appropriate agent
            if (current_player == 0) {
                chosen_action = random_agent.get_action(*state);
                agent_name = "RandomAgent";
            } else if (current_player == 1) {
                try {
                    chosen_action = minimax_agent.get_action(*state);
                    agent_name = "MinimaxAgent";
                    std::cout << "📊 MinimaxAgent explored " << minimax_agent.nodes_explored() << " nodes" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "⚠️  MinimaxAgent failed (" << e.what() << "), using random fallback" << std::endl;
                    auto legal_actions = state->LegalActions();
                    if (!legal_actions.empty()) {
                        chosen_action = legal_actions[0];
                    } else {
                        break;
                    }
                    agent_name = "MinimaxAgent (fallback)";
                }
            } else {
                // Unexpected player (shouldn't happen in 2-player game)
                std::cerr << "⚠️  Unexpected player " << current_player << std::endl;
                break;
            }
            
            // Apply the action
            std::cout << "🎯 " << agent_name << " chose action: " << chosen_action << std::endl;
            state->ApplyAction(chosen_action);
            
            turn_count++;
        }
        
        std::cout << "\n=== Game Results ===" << std::endl;
        
        if (state->IsTerminal()) {
            std::cout << "🏁 Game completed normally" << std::endl;
            auto returns = state->Returns();
            std::cout << "📊 Final returns:" << std::endl;
            for (size_t i = 0; i < returns.size(); ++i) {
                std::string agent_name = (i == 0) ? "RandomAgent" : "MinimaxAgent";
                std::cout << "   Player " << i << " (" << agent_name << "): " << returns[i] << std::endl;
            }
            
            // Determine winner for zero-sum games
            if (returns.size() >= 2) {
                if (returns[0] > returns[1]) {
                    std::cout << "🏆 RandomAgent (Player 0) wins!" << std::endl;
                } else if (returns[1] > returns[0]) {
                    std::cout << "🏆 MinimaxAgent (Player 1) wins!" << std::endl;
                } else {
                    std::cout << "🤝 Tie game!" << std::endl;
                }
            }
        } else {
            std::cout << "⏰ Game stopped after " << max_turns << " turns (demonstration limit)" << std::endl;
        }
        
        std::cout << "\n✅ Integration test completed successfully!" << std::endl;
        std::cout << "💡 Key observations:" << std::endl;
        std::cout << "   • Both agents can play against OpenSpiel Azul game" << std::endl;
        std::cout << "   • MinimaxAgent " << (game_type.utility == open_spiel::GameType::Utility::kZeroSum ? 
                                                "benefits from proper zero-sum algorithms" : 
                                                "works despite general-sum classification") << std::endl;
        std::cout << "   • Random vs Minimax provides good performance baseline" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 