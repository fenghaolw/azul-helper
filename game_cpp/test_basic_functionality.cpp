#include "game_state.h"
#include "tile.h"
#include "player_board.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace azul;

class TestBasicFunctionality {
public:
    static void test_tile_creation() {
        std::cout << "Testing tile creation..." << std::endl;
        
        // Test standard tiles
        auto tiles = Tile::create_standard_tiles();
        assert(tiles.size() == 100);
        
        // Count each color
        std::vector<int> color_counts(5, 0); // BLUE=0, YELLOW=1, RED=2, BLACK=3, WHITE=4
        for (const auto& tile : tiles) {
            int color_idx = static_cast<int>(tile.color());
            if (color_idx >= 0 && color_idx < 5) {
                color_counts[color_idx]++;
            }
        }
        
        // Should have 20 of each color
        for (int i = 0; i < 5; ++i) {
            assert(color_counts[i] == 20);
        }
        
        // Test first player marker
        auto first_player = Tile::create_first_player_marker();
        assert(first_player.is_first_player_marker());
        
        std::cout << "✓ Tile creation tests passed" << std::endl;
    }
    
    static void test_player_board() {
        std::cout << "Testing player board..." << std::endl;
        
        PlayerBoard board;
        
        // Test pattern line capacity
        for (int i = 0; i < 5; ++i) {
            assert(board.pattern_lines()[i].capacity() == i + 1);
        }
        
        // Test placing tiles
        std::vector<Tile> blue_tiles = {Tile(TileColor::BLUE), Tile(TileColor::BLUE)};
        board.place_tiles_on_pattern_line(1, blue_tiles);
        
        assert(board.pattern_lines()[1].tiles().size() == 2);
        assert(board.pattern_lines()[1].color().has_value());
        assert(board.pattern_lines()[1].color().value() == TileColor::BLUE);
        
        // Test wall pattern exists
        const auto& wall = board.wall();
        // Wall should be able to place any color on empty positions
        assert(wall.can_place_tile(0, TileColor::BLUE));
        assert(wall.can_place_tile(1, TileColor::WHITE));
        
        std::cout << "✓ Player board tests passed" << std::endl;
    }
    
    static void test_game_creation() {
        std::cout << "Testing game creation..." << std::endl;
        
        // Test 2-player game
        GameState game(2, 42);
        assert(game.num_players() == 2);
        assert(game.players().size() == 2);
        assert(game.factory_area().factories().size() == 5); // 2*2 + 1
        
        // Test 4-player game  
        GameState game4(4, 42);
        assert(game4.num_players() == 4);
        assert(game4.players().size() == 4);
        assert(game4.factory_area().factories().size() == 9); // 2*4 + 1
        
        std::cout << "✓ Game creation tests passed" << std::endl;
    }
    
    static void test_legal_actions() {
        std::cout << "Testing legal actions..." << std::endl;
        
        GameState game(2, 42);
        
        // Should have legal actions at start
        auto actions = game.get_legal_actions();
        assert(actions.size() > 0);
        
        // All actions should be valid
        for (size_t i = 0; i < std::min(actions.size(), size_t(5)); ++i) {
            assert(game.is_action_legal(actions[i]));
        }
        
        std::cout << "✓ Legal actions tests passed" << std::endl;
    }
    
    static void test_action_application() {
        std::cout << "Testing action application..." << std::endl;
        
        GameState game(2, 42);
        int initial_player = game.current_player();
        
        // Get and apply a legal action
        auto actions = game.get_legal_actions();
        assert(actions.size() > 0);
        
        bool success = game.apply_action(actions[0]);
        assert(success);
        
        // Player should change (unless round ended)
        if (!game.factory_area().is_round_over()) {
            assert(game.current_player() != initial_player);
        }
        
        std::cout << "✓ Action application tests passed" << std::endl;
    }
    
    static void test_state_vector() {
        std::cout << "Testing state vector..." << std::endl;
        
        GameState game(2, 42);
        auto state = game.get_state_vector();
        
        // Should be a vector of numbers
        assert(state.size() > 0);
        
        // All values should be valid floats in [0,1] range (allowing small epsilon for floating point)
        for (size_t i = 0; i < state.size(); ++i) {
            float value = state[i];
            assert(!std::isnan(value) && !std::isinf(value)); // Should be valid numbers
            assert(value >= -0.001f && value <= 1.001f); // Allow small epsilon for floating point errors
        }
        
        // Test state vector after some game progression
        for (int i = 0; i < 3; ++i) {
            auto actions = game.get_legal_actions();
            if (!actions.empty()) {
                game.apply_action(actions[0]);
            }
        }
        
        auto state2 = game.get_state_vector();
        assert(state2.size() > 0);
        
        // Should still be normalized
        for (size_t i = 0; i < state2.size(); ++i) {
            float value = state2[i];
            assert(!std::isnan(value) && !std::isinf(value));
            assert(value >= -0.001f && value <= 1.001f);
        }
        
        std::cout << "✓ State vector tests passed" << std::endl;
    }
    
    static void test_game_copy() {
        std::cout << "Testing game copying..." << std::endl;
        
        GameState game(2, 42);
        
        // Apply some actions
        for (int i = 0; i < 3; ++i) {
            auto actions = game.get_legal_actions();
            if (!actions.empty()) {
                game.apply_action(actions[0]);
            }
        }
        
        // Copy the game
        auto game_copy = game.copy();
        
        // Should have same state
        assert(game.current_player() == game_copy.current_player());
        assert(game.round_number() == game_copy.round_number());
        
        auto scores1 = game.get_scores();
        auto scores2 = game_copy.get_scores();
        assert(scores1.size() == scores2.size());
        for (size_t i = 0; i < scores1.size(); ++i) {
            assert(scores1[i] == scores2[i]);
        }
        
        std::cout << "✓ Game copying tests passed" << std::endl;
    }
    
    static void run_all_tests() {
        std::cout << "Running C++ Azul game tests...\n" << std::endl;
        
        try {
            test_tile_creation();
            test_player_board();
            test_game_creation();
            test_legal_actions();
            test_action_application();
            test_state_vector();
            test_game_copy();
            
            std::cout << "\n🎉 All basic functionality tests passed!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\n❌ Test failed: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    TestBasicFunctionality::run_all_tests();
    return 0;
} 