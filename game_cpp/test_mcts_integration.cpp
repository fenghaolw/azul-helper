#include <iostream>
#include <memory>

#ifdef WITH_OPENSPIEL
#include "open_spiel/spiel.h"
#include "open_spiel/algorithms/mcts.h"
#endif

int main() {
    std::cout << "=== Direct OpenSpiel Azul MCTS Demo ===" << std::endl;
    std::cout << "Pure OpenSpiel integration - no bridge complexity" << std::endl;
    std::cout << std::endl;
    
#ifdef WITH_OPENSPIEL
    try {
        // Load Azul game directly from OpenSpiel
        auto game = open_spiel::LoadGame("azul");
        if (!game) {
            std::cerr << "❌ Failed to load Azul game from OpenSpiel" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Successfully loaded OpenSpiel Azul game!" << std::endl;
        std::cout << "   Game: " << game->GetType().short_name << std::endl;
        std::cout << "   Max players: " << game->NumPlayers() << std::endl;
        std::cout << std::endl;
        
        // Create initial state
        auto state = game->NewInitialState();
        std::cout << "✅ Initial state created successfully" << std::endl;
        
        // Create MCTS evaluator and bot directly
        auto evaluator = std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(1, 42);
        open_spiel::algorithms::MCTSBot mcts_bot(
            *game,
            evaluator,
            1.4,   // UCT exploration constant
            400,   // Max simulations per move
            1000,  // Max memory MB
            false, // Don't solve for exact values
            42,    // Random seed
            true   // Verbose (shows tree search details)
        );
        
        std::cout << "✅ MCTS bot created with 100 simulations per move" << std::endl;
        std::cout << std::endl;
        
        // Run game simulation
        int turn = 1;
        while (!state->IsTerminal() && turn <= 5) {
            auto current_player = state->CurrentPlayer();
            auto legal_actions = state->LegalActions();
            
            std::cout << "Turn " << turn << ": " << legal_actions.size() << " legal actions" << std::endl;
            
            // Get MCTS action - this runs the tree search
            auto action = mcts_bot.Step(*state);
            std::cout << "MCTS selected action: " << action << std::endl;
            
            // Apply action directly to OpenSpiel state
            state->ApplyAction(action);
            
            turn++;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
#else
    std::cout << "❌ OpenSpiel not available - cannot run demo" << std::endl;
    std::cout << "   Please build OpenSpiel with BUILD_SHARED_LIB=ON" << std::endl;
    return 1;
#endif
    
    return 0;
} 