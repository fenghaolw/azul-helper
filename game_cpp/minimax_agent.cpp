#include "minimax_agent.h"
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <random>
#include <iostream>
#include <chrono>

#ifdef WITH_OPENSPIEL
#include "open_spiel/algorithms/minimax.h"
#endif

namespace azul {

MinimaxAgent::MinimaxAgent(int player_id, int depth, bool enable_alpha_beta, int seed)
    : player_id_(player_id), depth_(depth), enable_alpha_beta_(enable_alpha_beta),
      nodes_explored_(0) {
    // Suppress unused parameter warning
    (void)seed;
}

ActionType MinimaxAgent::get_action(const GameStateType& state) {
#ifdef WITH_OPENSPIEL
    if (state.IsTerminal()) {
        throw std::runtime_error("Cannot get action from terminal state");
    }
    
    // Reset statistics for this search
    nodes_explored_ = 0;
    
    auto legal_actions = state.LegalActions();
    
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    if (legal_actions.size() == 1) {
        return legal_actions[0];
    }
    
    // Create a custom evaluation function that captures our player_id
    auto evaluation_function = [this](const open_spiel::State& s) -> double {
        return this->evaluate_state(s);
    };
    
    auto game = state.GetGame();
    std::pair<double, open_spiel::Action> result;
    
            try {
            // OpenSpiel's minimax algorithms have specific requirements:
            // - AlphaBetaSearch: only works with kDeterministic games
            // - ExpectiminimaxSearch: works with stochastic games but may have its own limitations
            
            auto game_type = game->GetType();
            
            if (game_type.chance_mode == open_spiel::GameType::ChanceMode::kDeterministic) {
                // Use AlphaBetaSearch for deterministic games only
                result = open_spiel::algorithms::AlphaBetaSearch(
                    *game, &state, evaluation_function, depth_, player_id_
                );
            } else if (game_type.utility == open_spiel::GameType::Utility::kZeroSum) {
                // Try ExpectiminimaxSearch for stochastic zero-sum games
                result = open_spiel::algorithms::ExpectiminimaxSearch(
                    *game, &state, evaluation_function, depth_, player_id_
                );
            } else {
                // For general-sum stochastic games, OpenSpiel's algorithms may not work
                // Fall back to our own simple minimax-like evaluation
                throw std::runtime_error("Unsupported game type for OpenSpiel minimax algorithms");
            }
            
            // Update node exploration count (approximation)
            nodes_explored_ = static_cast<int>(std::pow(legal_actions.size(), std::min(depth_, 3)));
            
            return result.second;
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: OpenSpiel minimax search failed (" << e.what() 
                      << "), falling back to simple greedy selection" << std::endl;
            
            // Fallback: implement our own simple minimax-like search
            ActionType best_action = legal_actions[0];
            double best_value = std::numeric_limits<double>::lowest();
            
            // Evaluate each action one level deep
            for (ActionType action : legal_actions) {
                auto next_state = state.Child(action);
                double value = evaluation_function(*next_state);
                
                if (value > best_value) {
                    best_value = value;
                    best_action = action;
                }
            }
            
            return best_action;
        }
#else
    // Fallback implementation for non-OpenSpiel builds
    if (state.is_game_over()) {
        throw std::runtime_error("Cannot get action from terminal state");
    }
    
    auto legal_actions = state.get_legal_actions();
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    // Simple fallback: return first legal action
    return legal_actions[0];
#endif
}

std::vector<double> MinimaxAgent::get_action_probabilities(const GameStateType& state) {
#ifdef WITH_OPENSPIEL
    if (state.IsTerminal()) {
        return {};
    }
    
    auto legal_actions = state.LegalActions();
    
    if (legal_actions.empty()) {
        return {};
    }
    
    // Minimax is deterministic, so best action gets probability 1.0
    std::vector<double> probabilities(legal_actions.size(), 0.0);
    
    ActionType best_action = get_action(state);
    
    // Find the index of the best action
    for (size_t i = 0; i < legal_actions.size(); ++i) {
        if (legal_actions[i] == best_action) {
            probabilities[i] = 1.0;
            break;
        }
    }
    
    return probabilities;
#else
    if (state.is_game_over()) {
        return {};
    }
    
    auto legal_actions = state.get_legal_actions(player_id_);
    
    if (legal_actions.empty()) {
        return {};
    }
    
    // Simple fallback: uniform probabilities
    double uniform_prob = 1.0 / static_cast<double>(legal_actions.size());
    return std::vector<double>(legal_actions.size(), uniform_prob);
#endif
}

void MinimaxAgent::reset() {
    move_scores_.clear();
    reset_stats();
}

void MinimaxAgent::reset_stats() {
    nodes_explored_ = 0;
}

// Minimax search is now handled by OpenSpiel, so this method is simplified
double MinimaxAgent::minimax(const GameStateType& state, int depth, bool maximizing_player,
                            double alpha, double beta) const {
#ifdef WITH_OPENSPIEL
    // This method is now primarily for compatibility
    // The actual minimax is handled by OpenSpiel's algorithms
    return evaluate_state(state);
#else
    // Fallback for non-OpenSpiel builds would go here
    (void)depth; (void)maximizing_player; (void)alpha; (void)beta;
    return evaluate_state(state);
#endif
}

double MinimaxAgent::evaluate_state(const GameStateType& state) const {
#ifdef WITH_OPENSPIEL
    if (state.IsTerminal()) {
        // Terminal state evaluation - use OpenSpiel's Returns for zero-sum games
        auto returns = state.Returns();
        if (player_id_ >= 0 && player_id_ < static_cast<int>(returns.size())) {
            // In zero-sum games, our return is what matters (opponent's return is -ours)
            return returns[player_id_];
        }
        return 0.0;
    }
    
    // For non-terminal states, implement a basic Azul heuristic evaluation
    // This is a simplified heuristic - in practice, you'd want more sophisticated evaluation
    
    try {
        // Get state string and parse for basic information
        std::string state_str = state.ToString();
        
        // Basic heuristic: favor states where we're likely to score more points
        // This is a placeholder - real Azul evaluation would consider:
        // - Pattern line completion potential
        // - Wall placement strategy  
        // - Floor line penalties
        // - Tile availability in factories/center
        
        // For now, use a simple random-like evaluation with slight bias toward our player
        // This ensures the search explores different paths
        std::hash<std::string> hasher;
        double hash_value = static_cast<double>(hasher(state_str) % 1000) / 1000.0;
        
        // Add slight bias for our player position to break ties
        double player_bias = (player_id_ + 1) * 0.001;
        
        return hash_value + player_bias;
        
    } catch (const std::exception&) {
        // If state parsing fails, return neutral evaluation
        return 0.0;
    }
#else
    // Fallback for non-OpenSpiel builds would implement custom evaluation
    return 0.0;
#endif
}

bool MinimaxAgent::is_maximizing_player(const GameStateType& state) const {
#ifdef WITH_OPENSPIEL
    return state.CurrentPlayer() == player_id_;
#else
    return state.current_player() == player_id_;
#endif
}

// Simplified helper methods since OpenSpiel handles the search
std::vector<ActionType> MinimaxAgent::order_actions(const std::vector<ActionType>& actions, const GameStateType& state, bool use_previous_scores) const {
    // OpenSpiel's search algorithms handle action ordering internally
    // Return actions as-is for now
    (void)state; (void)use_previous_scores;
    return actions;
}

double MinimaxAgent::evaluate_move_priority(const ActionType& action, const GameStateType& state) const {
    // Simplified since OpenSpiel handles move ordering
    (void)action; (void)state;
    return 0.0;
}

std::string MinimaxAgent::action_key(const ActionType& action) const {
    return std::to_string(action);
}

std::vector<ActionType> MinimaxAgent::filter_obviously_bad_moves(const std::vector<ActionType>& actions, const GameStateType& state) const {
    // OpenSpiel's algorithms are efficient enough that we don't need aggressive filtering
    // Return all actions for now
    (void)state;
    return actions;
}

bool MinimaxAgent::is_obviously_bad_move(const ActionType& action, const GameStateType& state) const {
    // Simplified - let OpenSpiel's search evaluate all moves
    (void)action; (void)state;
    return false;
}

// Time-limited search is not needed since OpenSpiel handles performance
double MinimaxAgent::minimax_with_time_limit(const GameStateType& state, int depth, bool maximizing_player,
                          double alpha, double beta, 
                          std::chrono::high_resolution_clock::time_point start_time,
                          std::chrono::milliseconds time_limit) const {
    (void)start_time; (void)time_limit;
    return minimax(state, depth, maximizing_player, alpha, beta);
}

std::unique_ptr<MinimaxAgent> create_minimax_agent(int player_id, int depth, 
                                                  bool enable_alpha_beta,
                                                  int seed) {
    return std::make_unique<MinimaxAgent>(player_id, depth, enable_alpha_beta, seed);
}

} // namespace azul 