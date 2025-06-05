#include <iostream>
#include <memory>
#include "open_spiel/spiel.h"
#include "open_spiel/algorithms/mcts.h"
// Include our local Azul game (which auto-registers itself)
#include "azul.h"

// Force linker to include the azul registration by referencing symbols
namespace {
void force_azul_registration() {
    // Reference symbols from azul namespace to force linking
    (void)open_spiel::azul::TileColorToString(open_spiel::azul::TileColor::kBlue);
}
}

int main() {
    std::cout << "=== Local Azul MCTS Demo ===" << std::endl;
    std::cout << "Using local forked Azul game with OpenSpiel MCTS" << std::endl;
    std::cout << std::endl;
    
    try {
        // Force registration by calling the function (this ensures linking)
        force_azul_registration();
        
        // Load our local Azul game (auto-registered by REGISTER_SPIEL_GAME in azul.cc)
        auto game = open_spiel::LoadGame("azul");
        if (!game) {
            std::cerr << "❌ Failed to load local Azul game" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Successfully loaded local Azul game!" << std::endl;
        std::cout << "   Game: " << game->GetType().short_name << std::endl;
        std::cout << "   Max players: " << game->NumPlayers() << std::endl;
        std::cout << "   Utility: " << (game->GetType().utility == open_spiel::GameType::Utility::kZeroSum ? "Zero-Sum" : "General-Sum") << std::endl;
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
        
        std::cout << "✅ MCTS bot created with 400 simulations per move" << std::endl;
        std::cout << std::endl;
        
        // Run game simulation
        int turn = 1;
        while (!state->IsTerminal() && turn <= 5) {
            auto current_player = state->CurrentPlayer();
            auto legal_actions = state->LegalActions();
            
            std::cout << "Turn " << turn << " (Player " << current_player << "): " << legal_actions.size() << " legal actions" << std::endl;
            
            // Get MCTS action - this runs the tree search
            auto action = mcts_bot.Step(*state);
            std::cout << "MCTS selected action: " << action << std::endl;
            
            // Apply action directly to OpenSpiel state
            state->ApplyAction(action);
            
            turn++;
        }
        
        std::cout << std::endl;
        std::cout << "✅ Demo completed successfully!" << std::endl;
        std::cout << "💡 Your local Azul game is working with OpenSpiel MCTS" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 