#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>
#include <fstream>

#include "agent_evaluator.h"
#include "agent_profiler.h"
#include "evaluation_config.h"
#include "minimax_agent.h"
#include "mcts_agent.h"
#include "random_agent.h"

namespace {
void force_azul_registration() {
    // Reference symbols from azul namespace to force linking
    (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}
}

int main() {
    std::cout << "=== AZUL AGENT EVALUATION, TOURNAMENT & PROFILER DEMO ===" << std::endl;
    std::cout << "Testing minimax vs MCTS agents with comprehensive analysis" << std::endl;
    std::cout << std::endl;
    
    try {
        // Force registration
        force_azul_registration();
        
        // Verify Azul game loads
        auto game = open_spiel::LoadGame("azul");
        if (!game) {
            std::cerr << "❌ Failed to load Azul game" << std::endl;
            return 1;
        }
        std::cout << "✅ Azul game loaded successfully" << std::endl;
        
        // =================================================================
        // PART 1: INDIVIDUAL AGENT PROFILING
        // =================================================================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PART 1: AGENT PROFILING ANALYSIS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Enable profiling
        auto& profiler = azul::AgentProfiler::instance();
        profiler.start_profiling();
        
        // Create profiled agents
        auto profiled_minimax = azul::create_profiled_minimax_agent(0, 3, true, 42);
        auto profiled_mcts = azul::create_profiled_mcts_agent(1, 500, 1.4, 42);
        
        // Run a few test games to collect profiling data
        std::cout << "Running profiling games..." << std::endl;
        auto test_state = game->NewInitialState();
        
        for (int i = 0; i < 3; ++i) {
            auto state = test_state->Clone();
            int moves = 0;
            
            while (!state->IsTerminal() && moves < 20) {
                auto current_player = state->CurrentPlayer();
                open_spiel::Action action;
                
                if (current_player == 0) {
                    action = profiled_minimax->get_action(*state);
                } else {
                    action = profiled_mcts->get_action(*state);
                }
                
                state->ApplyAction(action);
                moves++;
            }
        }
        
        // Print profiling results
        std::cout << "\nPROFILING RESULTS:" << std::endl;
        profiler.print_profile_report();
        profiler.print_hotspots(std::cout, 5);
        
        // Save profiling report
        profiler.save_profile_report("profiling_report.txt");
        std::cout << "Profiling report saved to: profiling_report.txt" << std::endl;
        
        profiler.stop_profiling();
        
        // =================================================================
        // PART 2: HEAD-TO-HEAD EVALUATION
        // =================================================================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PART 2: HEAD-TO-HEAD AGENT EVALUATION" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Configure evaluation
        azul::EvaluationConfig eval_config;
        eval_config.num_games = 50;
        eval_config.verbose = true;
        eval_config.swap_player_positions = true;
        eval_config.timeout_per_move = 10.0;
        
        azul::AgentEvaluator evaluator(eval_config);
        
        // Create agents for evaluation
        auto minimax_agent = azul::create_minimax_evaluation_agent(4, true, 42, "Minimax_D4");
        auto mcts_agent = azul::create_mcts_evaluation_agent(1000, 1.4, 42, "MCTS_1000");
        auto random_agent = azul::create_random_evaluation_agent(42, "Random");
        
        // Test 1: Minimax vs MCTS
        std::cout << "\n--- Evaluation 1: Minimax vs MCTS ---" << std::endl;
        auto result1 = evaluator.evaluate_agent(*minimax_agent, *mcts_agent);
        std::cout << result1.summary() << std::endl;
        
        // Test 2: Minimax vs Random (should win easily)
        std::cout << "\n--- Evaluation 2: Minimax vs Random ---" << std::endl;
        auto result2 = evaluator.evaluate_agent(*minimax_agent, *random_agent);
        std::cout << result2.summary() << std::endl;
        
        // Test 3: MCTS vs Random (should win easily)
        std::cout << "\n--- Evaluation 3: MCTS vs Random ---" << std::endl;
        auto result3 = evaluator.evaluate_agent(*mcts_agent, *random_agent);
        std::cout << result3.summary() << std::endl;
        
        // =================================================================
        // PART 3: TOURNAMENT EVALUATION
        // =================================================================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PART 3: ROUND-ROBIN TOURNAMENT" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Configure tournament
        azul::EvaluationConfig tournament_config;
        tournament_config.num_games = 30; // Fewer games per matchup for faster tournament
        tournament_config.verbose = false; // Less verbose for tournament
        tournament_config.swap_player_positions = true;
        
        azul::Tournament tournament(tournament_config);
        
        // Add various agents to tournament
        tournament.add_agent(azul::create_minimax_evaluation_agent(3, true, 42, "Minimax_D3"));
        tournament.add_agent(azul::create_minimax_evaluation_agent(4, true, 42, "Minimax_D4"));
        tournament.add_agent(azul::create_mcts_evaluation_agent(500, 1.4, 42, "MCTS_500"));
        tournament.add_agent(azul::create_mcts_evaluation_agent(1000, 1.4, 42, "MCTS_1000"));
        tournament.add_agent(azul::create_mcts_evaluation_agent(1000, 2.0, 42, "MCTS_1000_UCT2"));
        tournament.add_agent(azul::create_random_evaluation_agent(42, "Random"));
        
        std::cout << "Starting tournament with " << tournament.get_num_agents() << " agents..." << std::endl;
        
        // Run tournament
        auto tournament_result = tournament.run_tournament();
        
        // Display results
        std::cout << "\nTOURNAMENT RESULTS:" << std::endl;
        std::cout << tournament_result.summary() << std::endl;
        
        // =================================================================
        // PART 4: PARAMETER SENSITIVITY ANALYSIS
        // =================================================================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PART 4: PARAMETER SENSITIVITY ANALYSIS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Quick evaluation config for parameter testing
        azul::EvaluationConfig quick_config;
        quick_config.num_games = 20;
        quick_config.verbose = false;
        
        azul::AgentEvaluator quick_evaluator(quick_config);
        
        // Test different MCTS simulation counts
        std::cout << "\nMCTS Simulation Count Analysis:" << std::endl;
        std::vector<int> sim_counts = {100, 500, 1000, 2000};
        auto baseline_random = azul::create_random_evaluation_agent(42, "Random_Baseline");
        
        for (int sims : sim_counts) {
            auto mcts_test = azul::create_mcts_evaluation_agent(sims, 1.4, 42, 
                                                              "MCTS_" + std::to_string(sims));
            auto result = quick_evaluator.quick_evaluation(*mcts_test, *baseline_random, 20);
            
            std::cout << "  MCTS(" << sims << " sims): " 
                      << std::fixed << std::setprecision(1) 
                      << (result.test_agent_win_rate * 100) << "% win rate" << std::endl;
        }
        
        // Test different minimax depths
        std::cout << "\nMinimax Depth Analysis:" << std::endl;
        std::vector<int> depths = {2, 3, 4, 5};
        
        for (int depth : depths) {
            auto minimax_test = azul::create_minimax_evaluation_agent(depth, true, 42, 
                                                                    "Minimax_D" + std::to_string(depth));
            auto result = quick_evaluator.quick_evaluation(*minimax_test, *baseline_random, 20);
            
            std::cout << "  Minimax(depth " << depth << "): " 
                      << std::fixed << std::setprecision(1) 
                      << (result.test_agent_win_rate * 100) << "% win rate" << std::endl;
        }
        
        // =================================================================
        // PART 5: SAVE DETAILED RESULTS
        // =================================================================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PART 5: SAVING DETAILED RESULTS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Save evaluation results to file
        std::ofstream results_file("evaluation_results.txt");
        if (results_file.is_open()) {
            results_file << "=== AZUL AGENT EVALUATION RESULTS ===" << std::endl;
            results_file << std::endl;
            
            results_file << "HEAD-TO-HEAD EVALUATIONS:" << std::endl;
            results_file << result1.summary() << std::endl;
            results_file << result2.summary() << std::endl;
            results_file << result3.summary() << std::endl;
            
            results_file << "TOURNAMENT RESULTS:" << std::endl;
            results_file << tournament_result.summary() << std::endl;
            
            results_file.close();
            std::cout << "Detailed results saved to: evaluation_results.txt" << std::endl;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "✅ EVALUATION DEMO COMPLETED SUCCESSFULLY!" << std::endl;
        std::cout << "Key outputs:" << std::endl;
        std::cout << "  - profiling_report.txt: Performance profiling data" << std::endl;
        std::cout << "  - evaluation_results.txt: Detailed evaluation results" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during evaluation: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 