#include "random_agent.h"
#include "minimax_agent.h"
#include "game_state.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace azul;

void test_random_agent() {
    std::cout << "=== Testing Random Agent ===" << std::endl;
    
    // Create game and agent
    GameState game = create_game(2, 42); // Fixed seed for reproducibility
    RandomAgent agent(0, 42); // Fixed seed for reproducibility
    
    std::cout << "Agent player ID: " << agent.player_id() << std::endl;
    std::cout << "Agent seed: " << agent.seed() << std::endl;
    
    // Test action selection for a few moves
    int moves = 0;
    while (!game.is_game_over() && moves < 5) {
        if (game.current_player() == agent.player_id()) {
            auto legal_actions = game.get_legal_actions();
            std::cout << "Turn " << moves + 1 << ", Legal actions: " << legal_actions.size() << std::endl;
            
            // Get action
            auto start_time = std::chrono::high_resolution_clock::now();
            Action action = agent.get_action(game);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            std::cout << "Selected action: " << action.to_string() 
                      << " (time: " << duration.count() << " μs)" << std::endl;
            
            // Get probabilities
            auto probs = agent.get_action_probabilities(game);
            std::cout << "Action probabilities (uniform): ";
            for (size_t i = 0; i < std::min(probs.size(), size_t(3)); ++i) {
                std::cout << std::fixed << std::setprecision(3) << probs[i] << " ";
            }
            if (probs.size() > 3) std::cout << "...";
            std::cout << std::endl;
            
            // Apply action
            if (!game.apply_action(action)) {
                std::cout << "Error: Failed to apply action!" << std::endl;
                break;
            }
            
            ++moves;
        } else {
            // Skip opponent's turn by selecting a random legal action
            auto legal_actions = game.get_legal_actions();
            if (!legal_actions.empty()) {
                game.apply_action(legal_actions[0]);
            }
        }
    }
    
    std::cout << "Random agent test completed.\n" << std::endl;
}

void test_minimax_agent() {
    std::cout << "=== Testing Minimax Agent ===" << std::endl;
    
    // Create game and agent
    GameState game = create_game(2, 42); // Fixed seed for reproducibility
    MinimaxAgent agent(0, 3, true, true, 42); // Depth 3, alpha-beta, memoization
    
    std::cout << "Agent player ID: " << agent.player_id() << std::endl;
    std::cout << "Search depth: " << agent.depth() << std::endl;
    std::cout << "Alpha-beta pruning: " << (agent.alpha_beta_enabled() ? "enabled" : "disabled") << std::endl;
    std::cout << "Memoization: " << (agent.memoization_enabled() ? "enabled" : "disabled") << std::endl;
    
    // Test action selection for a few moves
    int moves = 0;
    while (!game.is_game_over() && moves < 3) { // Fewer moves for minimax due to computational cost
        if (game.current_player() == agent.player_id()) {
            auto legal_actions = game.get_legal_actions();
            std::cout << "Turn " << moves + 1 << ", Legal actions: " << legal_actions.size() << std::endl;
            
            // Reset stats before search
            agent.reset_stats();
            
            // Get action
            auto start_time = std::chrono::high_resolution_clock::now();
            Action action = agent.get_action(game);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "Selected action: " << action.to_string() 
                      << " (time: " << duration.count() << " ms)" << std::endl;
            std::cout << "Nodes explored: " << agent.nodes_explored() << std::endl;
            std::cout << "Cache hits: " << agent.cache_hits() << std::endl;
            
            // Get probabilities (should be deterministic)
            auto probs = agent.get_action_probabilities(game);
            std::cout << "Action probabilities (deterministic): ";
            for (size_t i = 0; i < std::min(probs.size(), size_t(3)); ++i) {
                std::cout << std::fixed << std::setprecision(3) << probs[i] << " ";
            }
            if (probs.size() > 3) std::cout << "...";
            std::cout << std::endl;
            
            // Apply action
            if (!game.apply_action(action)) {
                std::cout << "Error: Failed to apply action!" << std::endl;
                break;
            }
            
            ++moves;
        } else {
            // Skip opponent's turn by selecting a random legal action
            auto legal_actions = game.get_legal_actions();
            if (!legal_actions.empty()) {
                game.apply_action(legal_actions[0]);
            }
        }
    }
    
    std::cout << "Minimax agent test completed.\n" << std::endl;
}

void test_agent_comparison() {
    std::cout << "=== Agent Comparison ===" << std::endl;
    
    // Create agents
    RandomAgent random_agent(0, 42);
    MinimaxAgent minimax_agent(1, 4, true, true, 42); // Shallow depth for speed
    
    // Create game
    GameState game = create_game(2, 42);
    
    std::cout << "Random vs Minimax (depth 4) game:" << std::endl;
    
    int turn = 0;
    while (!game.is_game_over() && turn < 10) {
        int current_player = game.current_player();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Action action(0, TileColor::BLUE, 0); // Default initialization
        
        if (current_player == 0) {
            // Random agent's turn
            action = random_agent.get_action(game);
            std::cout << "Turn " << turn + 1 << " - Random: " << action.to_string();
        } else {
            // Minimax agent's turn
            action = minimax_agent.get_action(game);
            std::cout << "Turn " << turn + 1 << " - Minimax: " << action.to_string();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << " (" << duration.count() << " μs)" << std::endl;
        
        if (!game.apply_action(action)) {
            std::cout << "Error: Failed to apply action!" << std::endl;
            break;
        }
        
        ++turn;
    }
    
    auto scores = game.get_scores();
    std::cout << "Final scores - Random: " << scores[0] << ", Minimax: " << scores[1] << std::endl;
    
    if (game.is_game_over()) {
        int winner = game.get_winner();
        if (winner == 0) {
            std::cout << "Random agent wins!" << std::endl;
        } else if (winner == 1) {
            std::cout << "Minimax agent wins!" << std::endl;
        } else {
            std::cout << "Game is a tie!" << std::endl;
        }
    }
    
    std::cout << "Agent comparison completed.\n" << std::endl;
}

void test_complete_game() {
    std::cout << "=== Complete Game: Random vs Minimax ===" << std::endl;
    
    // Create agents
    RandomAgent random_agent(0, 42);
    MinimaxAgent minimax_agent(1, 3, true, true, 42); // Depth 3 for reasonable speed
    
    // Create game
    GameState game = create_game(2, 42);
    
    std::cout << "Players:" << std::endl;
    std::cout << "  Player 0: Random Agent" << std::endl;
    std::cout << "  Player 1: Minimax Agent (depth 3)" << std::endl;
    std::cout << std::endl;
    
    int turn = 0;
    int round = 1;
    
    while (!game.is_game_over()) {
        int current_player = game.current_player();
        
        Action action(0, TileColor::BLUE, 0); // Initialize with dummy values
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (current_player == 0) {
            // Random agent's turn
            action = random_agent.get_action(game);
        } else {
            // Minimax agent's turn
            action = minimax_agent.get_action(game);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Round " << round << ", Turn " << turn + 1 << " - " 
                  << (current_player == 0 ? "Random" : "Minimax") << ": " 
                  << action.to_string() << " (" << duration.count() << " ms)" << std::endl;
        
        if (!game.apply_action(action)) {
            std::cout << "Error: Failed to apply action!" << std::endl;
            break;
        }
        
        ++turn;
        
        // Check if round ended (simple heuristic: no legal actions for a while)
        auto legal_actions = game.get_legal_actions();
        if (legal_actions.empty() || (turn % 20 == 0)) {
            auto scores = game.get_scores();
            std::cout << "  Current scores - Random: " << scores[0] 
                      << ", Minimax: " << scores[1] << std::endl;
            if (turn % 20 == 0) {
                ++round;
            }
        }
    }
    
    // Final results
    auto scores = game.get_scores();
    std::cout << std::endl << "=== FINAL RESULTS ===" << std::endl;
    std::cout << "Random Agent Score: " << scores[0] << std::endl;
    std::cout << "Minimax Agent Score: " << scores[1] << std::endl;
    std::cout << "Total turns played: " << turn << std::endl;
    
    if (game.is_game_over()) {
        int winner = game.get_winner();
        if (winner == 0) {
            std::cout << "WINNER: Random Agent!" << std::endl;
        } else if (winner == 1) {
            std::cout << "WINNER: Minimax Agent!" << std::endl;
        } else {
            std::cout << "RESULT: Tie!" << std::endl;
        }
    } else {
        std::cout << "Game did not complete normally." << std::endl;
    }
    
    std::cout << "Complete game test completed.\n" << std::endl;
}

int main() {
    try {
        test_random_agent();
        test_minimax_agent();
        test_agent_comparison();
        test_complete_game();
        
        std::cout << "All tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 