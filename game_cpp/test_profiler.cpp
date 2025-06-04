#include "agent_profiler.h"
#include "minimax_agent.h"
#include "random_agent.h"
#include "game_state.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace azul;

void run_minimax_performance_test() {
    std::cout << "\n=== MINIMAX AGENT PERFORMANCE TEST ===\n";
    
    // Create profiled minimax agent
    auto profiled_agent = create_profiled_minimax_agent(0, 3, true, 42);
    auto random_agent = std::make_unique<RandomAgent>(1, 42);
    
    // Start profiling
    auto& profiler = AgentProfiler::instance();
    profiler.start_profiling();
    
    // Run multiple games to collect performance data
    int num_games = 5;
    int minimax_wins = 0;
    int random_wins = 0;
    
    for (int game = 0; game < num_games; ++game) {
        std::cout << "Game " << (game + 1) << "/" << num_games << " - ";
        
        GameState state(2, 42 + game);
        
        while (!state.is_game_over()) {
            if (state.current_player() == 0) {
                // Minimax agent turn
                auto action = profiled_agent->get_action(state);
                if (!state.apply_action(action)) {
                    std::cout << "Invalid action from minimax agent!\n";
                    break;
                }
            } else {
                // Random agent turn
                auto action = random_agent->get_action(state);
                if (!state.apply_action(action)) {
                    std::cout << "Invalid action from random agent!\n";
                    break;
                }
            }
        }
        
        auto scores = state.get_scores();
        if (scores[0] > scores[1]) {
            minimax_wins++;
            std::cout << "Minimax wins (" << scores[0] << " vs " << scores[1] << ")\n";
        } else if (scores[1] > scores[0]) {
            random_wins++;
            std::cout << "Random wins (" << scores[1] << " vs " << scores[0] << ")\n";
        } else {
            std::cout << "Tie (" << scores[0] << " vs " << scores[1] << ")\n";
        }
    }
    
    profiler.stop_profiling();
    
    std::cout << "\nFinal Results: Minimax " << minimax_wins << " - " << random_wins << " Random\n";
    
    // Print performance analysis
    profiler.print_profile_report();
    profiler.print_hotspots(std::cout, 5);
    
    // Save detailed report
    profiler.save_profile_report("minimax_performance_report.txt");
    std::cout << "\nDetailed report saved to: minimax_performance_report.txt\n";
}

void demonstrate_minimax_hotspots() {
    std::cout << "\n=== MINIMAX HOTSPOT ANALYSIS BY DEPTH ===\n";
    
    GameState state(2, 42);
    
    // Test different depths
    for (int depth = 1; depth <= 4; ++depth) {
        std::cout << "\nTesting depth " << depth << ":\n";
        
        auto profiled_agent = create_profiled_minimax_agent(0, depth, true, 42);
        auto& profiler = AgentProfiler::instance();
        
        profiler.reset_stats();
        profiler.start_profiling();
        
        // Time multiple action selections
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 3; ++i) {
            if (!state.is_game_over()) {
                auto action = profiled_agent->get_action(state);
                state.apply_action(action);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        profiler.stop_profiling();
        
        std::cout << "Total time for 3 actions: " << duration.count() << " ms\n";
        std::cout << "Agent stats: nodes=" << profiled_agent->agent().nodes_explored() << "\n";
        
        profiler.print_hotspots(std::cout, 3);
    }
}

void analyze_minimax_algorithms() {
    std::cout << "\n=== MINIMAX ALGORITHM COMPARISON ===\n";
    
    GameState state(2, 42);
    
    struct Config {
        std::string name;
        bool alpha_beta;
    };
    
    std::vector<Config> configs = {
        {"Basic Minimax", false},
        {"Alpha-Beta", true}
    };
    
    for (const auto& config : configs) {
        std::cout << "\nTesting: " << config.name << "\n";
        
        auto profiled_agent = create_profiled_minimax_agent(0, 3, config.alpha_beta, 42);
        auto& profiler = AgentProfiler::instance();
        
        profiler.reset_stats();
        profiler.start_profiling();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Run 5 action selections
        GameState test_state = state.copy();
        for (int i = 0; i < 5 && !test_state.is_game_over(); ++i) {
            auto action = profiled_agent->get_action(test_state);
            test_state.apply_action(action);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        profiler.stop_profiling();
        
        std::cout << "Time: " << duration.count() << " ms, "
                  << "Nodes: " << profiled_agent->agent().nodes_explored() << "\n";
        
        // Show top 2 hotspots
        profiler.print_hotspots(std::cout, 2);
    }
}

#ifdef WITH_OPENSPIEL
void run_mcts_performance_test() {
    std::cout << "\n=== MCTS AGENT PERFORMANCE TEST ===\n";
    
    // This would require OpenSpiel integration
    // For now, just show the structure
    std::cout << "MCTS profiling requires OpenSpiel integration.\n";
    std::cout << "Enable WITH_OPENSPIEL and provide OpenSpiel state conversion.\n";
}
#endif

int main() {
    std::cout << "Azul Agent Performance Profiler\n";
    std::cout << "===============================\n";
    
    try {
        // Run comprehensive minimax analysis
        run_minimax_performance_test();
        
        // Analyze depth performance
        demonstrate_minimax_hotspots();
        
        // Compare algorithm variants
        analyze_minimax_algorithms();
        
        #ifdef WITH_OPENSPIEL
        run_mcts_performance_test();
        #endif
        
        std::cout << "\n=== PROFILING COMPLETE ===\n";
        std::cout << "Check minimax_performance_report.txt for detailed analysis.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error during profiling: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 