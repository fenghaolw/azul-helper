#include "game_state.h"
#include <iostream>

using namespace azul;

int main() {
    GameState game(2, 42);
    
    std::cout << "=== Debug Round Progress ===" << std::endl;
    std::cout << "Initial state:" << std::endl;
    std::cout << "  Round: " << game.round_number() << std::endl;
    std::cout << "  Current player: " << game.current_player() << std::endl;
    std::cout << "  Round over: " << (game.factory_area().is_round_over() ? "Yes" : "No") << std::endl;
    std::cout << "  Factories: " << game.factory_area().factories().size() << std::endl;
    
    // Count tiles in factories
    int total_factory_tiles = 0;
    for (size_t i = 0; i < game.factory_area().factories().size(); ++i) {
        int factory_tiles = static_cast<int>(game.factory_area().factories()[i].tiles().size());
        std::cout << "  Factory " << i << ": " << factory_tiles << " tiles" << std::endl;
        total_factory_tiles += factory_tiles;
    }
    
    int center_tiles = static_cast<int>(game.factory_area().center().tiles().size());
    std::cout << "  Center: " << center_tiles << " tiles" << std::endl;
    std::cout << "  Total tiles in play: " << (total_factory_tiles + center_tiles) << std::endl;
    
    std::cout << "\n=== Playing Actions ===" << std::endl;
    
    int action_count = 0;
    while (!game.factory_area().is_round_over() && action_count < 30) {
        auto actions = game.get_legal_actions();
        if (actions.empty()) {
            std::cout << "No legal actions at step " << action_count << std::endl;
            break;
        }
        
        std::cout << "Action " << action_count << ": ";
        std::cout << "Player " << game.current_player() << ", ";
        std::cout << actions.size() << " legal actions" << std::endl;
        
        // Show first few actions
        for (size_t i = 0; i < std::min(actions.size(), size_t(3)); ++i) {
            std::cout << "  Option: Source=" << actions[i].source() 
                      << ", Color=" << static_cast<int>(actions[i].color())
                      << ", Dest=" << actions[i].destination() << std::endl;
        }
        
        bool success = game.apply_action(actions[0]);
        if (!success) {
            std::cout << "Action failed!" << std::endl;
            break;
        }
        
        action_count++;
        
        // Count remaining tiles
        total_factory_tiles = 0;
        for (const auto& factory : game.factory_area().factories()) {
            total_factory_tiles += static_cast<int>(factory.tiles().size());
        }
        center_tiles = static_cast<int>(game.factory_area().center().tiles().size());
        
        std::cout << "  After action: Factories=" << total_factory_tiles 
                  << ", Center=" << center_tiles 
                  << ", Round over=" << (game.factory_area().is_round_over() ? "Yes" : "No") << std::endl;
        
        if (action_count % 5 == 0) {
            std::cout << "  [Progress check at action " << action_count << "]" << std::endl;
        }
    }
    
    std::cout << "\n=== Final State ===" << std::endl;
    std::cout << "Actions taken: " << action_count << std::endl;
    std::cout << "Round over: " << (game.factory_area().is_round_over() ? "Yes" : "No") << std::endl;
    std::cout << "Game over: " << (game.is_game_over() ? "Yes" : "No") << std::endl;
    
    return 0;
} 