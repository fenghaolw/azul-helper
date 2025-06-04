#include "agent_evaluator.h"
#include <iostream>
#include <memory>

using namespace azul;

void test_agent_evaluation() {
    std::cout << "=== Agent Evaluation Test ===" << std::endl;
    
    // Create evaluation configuration - OPTIMIZED for speed
    EvaluationConfig config;
    config.num_games = 5; // Minimal for statistical validity
    config.verbose = false; // Reduce output
    config.swap_player_positions = true;
    config.use_fixed_seeds = true;
    config.random_seed = 42;
    config.timeout_per_move = 3.0; // Short timeout
    
    // Create agents
    auto random_agent = create_random_evaluation_agent(42, "Random_42");
    auto minimax_agent = create_minimax_evaluation_agent(3, true, 42, "Minimax_D3");
    
    // Create evaluator
    AgentEvaluator evaluator(config);
    
    std::cout << "Testing Random vs Minimax (depth 3, 5 games)..." << std::endl;
    
    auto result1 = evaluator.evaluate_agent(*random_agent, *minimax_agent);
    std::cout << "✓ Random: " << result1.test_agent_win_rate * 100 << "%, Minimax: " 
              << result1.baseline_agent_win_rate * 100 << "%" << std::endl;
}

void test_quick_evaluation() {
    std::cout << "\n=== Quick Evaluation Test ===" << std::endl;
    
    // Create agents
    auto random_agent = create_random_evaluation_agent(123, "Random_Fast");
    auto minimax_agent = create_minimax_evaluation_agent(3, true, 123, "Minimax_Fast");
    
    // Create evaluator with default config
    AgentEvaluator evaluator;
    
    std::cout << "Quick evaluation: 2 games..." << std::endl;
    auto result = evaluator.quick_evaluation(*random_agent, *minimax_agent, 2);
    std::cout << "✓ Completed: " << result.games_played << " games, " 
              << result.timeouts << " timeouts" << std::endl;
}

void test_tournament() {
    std::cout << "\n=== Tournament Test ===" << std::endl;
    
    // Create tournament configuration
    EvaluationConfig config;
    config.num_games = 5; // Fewer games per matchup for speed
    config.verbose = true;
    config.use_fixed_seeds = true;
    config.random_seed = 100;
    
    // Create tournament
    Tournament tournament(config);
    
    // Add agents
    tournament.add_agent(create_random_evaluation_agent(100, "Random_A"));
    tournament.add_agent(create_random_evaluation_agent(200, "Random_B"));
    tournament.add_agent(create_minimax_evaluation_agent(2, true, 100, "Minimax_D2"));
    tournament.add_agent(create_minimax_evaluation_agent(3, true, 100, "Minimax_D3"));
    
    std::cout << "Running tournament with " << tournament.get_num_agents() << " agents..." << std::endl;
    
    auto result = tournament.run_tournament();
    std::cout << "\n" << result.summary() << std::endl;
}

void test_minimax_variants() {
    std::cout << "\n=== Minimax Variants Test ===" << std::endl;
    
    EvaluationConfig config;
    config.num_games = 5; // Minimal
    config.verbose = false;
    config.use_fixed_seeds = true;
    config.random_seed = 500;
    config.timeout_per_move = 3.0;
    
    // Create different minimax configurations
    auto minimax_basic = create_minimax_evaluation_agent(3, false, 500, "Minimax_Basic");
    auto minimax_ab = create_minimax_evaluation_agent(3, true, 500, "Minimax_AlphaBeta");
    auto minimax_d2 = create_minimax_evaluation_agent(2, true, 500, "Minimax_D2");
    auto minimax_d4 = create_minimax_evaluation_agent(4, true, 500, "Minimax_D4");
    
    AgentEvaluator evaluator(config);
    
    std::cout << "\nBasic vs Alpha-Beta:" << std::endl;
    auto result1 = evaluator.evaluate_agent(*minimax_basic, *minimax_ab);
    std::cout << "Winner: " << (result1.test_agent_win_rate > 0.5 ? result1.test_agent_name : result1.baseline_agent_name)
              << " (" << std::max(result1.test_agent_win_rate, result1.baseline_agent_win_rate) * 100 << "%)" << std::endl;
    
    std::cout << "\nDepth 2 vs Depth 4:" << std::endl;
    auto result2 = evaluator.evaluate_agent(*minimax_d2, *minimax_d4);
    std::cout << "Winner: " << (result2.test_agent_win_rate > 0.5 ? result2.test_agent_name : result2.baseline_agent_name)
              << " (" << std::max(result2.test_agent_win_rate, result2.baseline_agent_win_rate) * 100 << "%)" << std::endl;
}

void test_performance_analysis() {
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    
    EvaluationConfig config;
    config.num_games = 1; // Just 1 game
    config.verbose = false;
    config.save_detailed_logs = false;
    config.timeout_per_move = 3.0;
    
    auto random_agent = create_random_evaluation_agent(999, "Random_Perf");
    auto minimax_agent = create_minimax_evaluation_agent(3, true, 999, "Minimax_Perf");
    
    AgentEvaluator evaluator(config);
    
    std::cout << "Performance test (1 game)..." << std::endl;
    auto result = evaluator.evaluate_agent(*random_agent, *minimax_agent);
    
    std::cout << "✓ Performance Summary:" << std::endl;
    std::cout << "  Games: " << result.games_played << std::endl;
    std::cout << "  Duration: " << result.average_game_duration << "s" << std::endl;
    std::cout << "  Timeouts: " << result.timeouts << std::endl;
    std::cout << "  Errors: " << result.errors << std::endl;
}

int main() {
    try {
        test_agent_evaluation();
        test_quick_evaluation();
        test_tournament();
        test_minimax_variants();
        test_performance_analysis();
        
        std::cout << "\n=== All Evaluation Tests Completed Successfully! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Evaluation test failed with exception: " << e.what() << std::endl;
        return 1;
    }
} 