#include "game_state.h"
#include "tile.h"
#include "action.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace azul;

class TestEdgeCases {
public:
    static void test_invalid_player_counts() {
        std::cout << "Testing invalid player counts..." << std::endl;
        
        // Test too few players
        try {
            GameState game(1, 42); // Should handle gracefully or throw
            // If it doesn't throw, at least verify it's bounded
            assert(game.num_players() >= 2);
        } catch (...) {
            // Expected to throw for invalid player count
        }
        
        // Test too many players
        try {
            GameState game(10, 42); // Should handle gracefully or throw
            assert(game.num_players() <= 4);
        } catch (...) {
            // Expected to throw for invalid player count
        }
        
        std::cout << "✓ Invalid player counts tests passed" << std::endl;
    }
    
    static void test_empty_factories_and_center() {
        std::cout << "Testing empty factories and center..." << std::endl;
        
        GameState game(2, 42);
        int initial_round = game.round_number();
        int action_count = 0;
        int action_limit = 50; // Safety limit
        
        // Play until round progresses (indicating factories were emptied) or we hit limit
        while (game.round_number() == initial_round && !game.is_game_over() && action_count < action_limit) {
            auto actions = game.get_legal_actions();
            if (actions.empty()) break;
            game.apply_action(actions[0]);
            action_count++;
        }
        
        std::cout << "   Round emptied in " << action_count << " actions" << std::endl;
        
        // After round transition, factories should be refilled for new round
        if (!game.is_game_over() && game.round_number() > initial_round) {
            auto actions = game.get_legal_actions();
            // Should have legal actions for the new round
            std::cout << "   New round has " << actions.size() << " legal actions" << std::endl;
        }
        
        std::cout << "✓ Empty factories and center tests passed" << std::endl;
    }
    
    static void test_invalid_actions() {
        std::cout << "Testing invalid actions..." << std::endl;
        
        GameState game(2, 42);
        
        // Test invalid source
        Action invalid_source(-999, TileColor::BLUE, 0);
        assert(!game.is_action_legal(invalid_source));
        assert(!game.apply_action(invalid_source));
        
        // Test invalid destination
        Action invalid_dest(0, TileColor::BLUE, 999);
        assert(!game.is_action_legal(invalid_dest));
        assert(!game.apply_action(invalid_dest));
        
        // Test valid action
        auto valid_actions = game.get_legal_actions();
        if (!valid_actions.empty()) {
            assert(game.is_action_legal(valid_actions[0]));
            assert(game.apply_action(valid_actions[0]));
        }
        
        std::cout << "✓ Invalid actions tests passed" << std::endl;
    }
    
    static void test_overflow_tiles_handling() {
        std::cout << "Testing overflow tiles handling..." << std::endl;
        
        PlayerBoard board;
        
        // Try to place too many tiles on pattern line
        std::vector<Tile> too_many_tiles;
        for (int i = 0; i < 10; ++i) {
            too_many_tiles.push_back(Tile(TileColor::BLUE));
        }
        
        // In Azul, you can't split tile groups - either all fit or none fit
        // Pattern line 0 has capacity 1, so all 10 tiles should be rejected
        auto overflow = board.place_tiles_on_pattern_line(0, too_many_tiles);
        assert(overflow.size() == 10); // All 10 tiles should be rejected
        assert(board.pattern_lines()[0].tiles().size() == 0); // None should fit
        
        // Test with tiles that do fit
        std::vector<Tile> fitting_tiles = {Tile(TileColor::BLUE)};
        auto no_overflow = board.place_tiles_on_pattern_line(0, fitting_tiles);
        assert(no_overflow.size() == 0); // No overflow
        assert(board.pattern_lines()[0].tiles().size() == 1); // Should fit
        
        std::cout << "✓ Overflow tiles handling tests passed" << std::endl;
    }
    
    static void test_wrong_color_on_pattern_line() {
        std::cout << "Testing wrong color on pattern line..." << std::endl;
        
        PlayerBoard board;
        
        // Place blue tiles on line 1
        std::vector<Tile> blue_tiles = {Tile(TileColor::BLUE)};
        board.place_tiles_on_pattern_line(1, blue_tiles);
        
        // Try to place red tiles on same line (should fail)
        std::vector<Tile> red_tiles = {Tile(TileColor::RED)};
        assert(!board.can_place_tiles_on_pattern_line(1, red_tiles));
        
        // All red tiles should be returned as overflow
        auto overflow = board.place_tiles_on_pattern_line(1, red_tiles);
        assert(overflow.size() == red_tiles.size());
        
        std::cout << "✓ Wrong color on pattern line tests passed" << std::endl;
    }
    
    static void test_wall_position_already_filled() {
        std::cout << "Testing wall position already filled..." << std::endl;
        
        PlayerBoard board;
        
        // Place a blue tile to complete pattern line 0
        board.place_tiles_on_pattern_line(0, {Tile(TileColor::BLUE)});
        board.end_round_scoring(); // This should place blue at (0,0)
        
        // Now blue position (0,0) should be filled
        assert(board.wall().is_filled(0, 0));
        
        // Trying to place blue on line 0 again should not be allowed
        std::vector<Tile> more_blue = {Tile(TileColor::BLUE)};
        assert(!board.can_place_tiles_on_pattern_line(0, more_blue));
        
        std::cout << "✓ Wall position already filled tests passed" << std::endl;
    }
    
    static void test_floor_line_overflow() {
        std::cout << "Testing floor line overflow..." << std::endl;
        
        PlayerBoard board;
        
        // Fill floor line beyond capacity
        std::vector<Tile> many_tiles;
        for (int i = 0; i < 20; ++i) {
            many_tiles.push_back(Tile(TileColor::BLUE));
        }
        
        auto overflow = board.place_tiles_on_floor_line(many_tiles);
        // Floor line has limited capacity, excess should overflow
        assert(board.floor_line().size() <= 7); // Max floor line capacity
        
        std::cout << "✓ Floor line overflow tests passed" << std::endl;
    }
    
    static void test_empty_tile_vectors() {
        std::cout << "Testing empty tile vectors..." << std::endl;
        
        PlayerBoard board;
        
        // Try to place empty vector of tiles
        std::vector<Tile> empty_tiles;
        auto overflow1 = board.place_tiles_on_pattern_line(0, empty_tiles);
        auto overflow2 = board.place_tiles_on_floor_line(empty_tiles);
        
        assert(overflow1.empty());
        assert(overflow2.empty());
        assert(board.pattern_lines()[0].tiles().empty());
        
        std::cout << "✓ Empty tile vectors tests passed" << std::endl;
    }
    
    static void test_multiple_first_player_markers() {
        std::cout << "Testing multiple first player markers..." << std::endl;
        
        PlayerBoard board;
        
        // Try to place multiple first player markers
        std::vector<Tile> multiple_fpm = {
            Tile::create_first_player_marker(),
            Tile::create_first_player_marker()
        };
        
        board.place_tiles_on_floor_line(multiple_fpm);
        // Should handle multiple markers gracefully
        assert(board.has_first_player_marker());
        
        std::cout << "✓ Multiple first player markers tests passed" << std::endl;
    }
    
    static void test_pattern_line_bounds() {
        std::cout << "Testing pattern line bounds..." << std::endl;
        
        PlayerBoard board;
        
        // Test negative line index
        std::vector<Tile> tiles = {Tile(TileColor::BLUE)};
        assert(!board.can_place_tiles_on_pattern_line(-1, tiles));
        
        // Test too high line index
        assert(!board.can_place_tiles_on_pattern_line(10, tiles));
        
        // Test valid range
        for (int i = 0; i < 5; ++i) {
            assert(board.can_place_tiles_on_pattern_line(i, tiles));
        }
        
        std::cout << "✓ Pattern line bounds tests passed" << std::endl;
    }
    
    static void test_wall_bounds() {
        std::cout << "Testing wall bounds..." << std::endl;
        
        PlayerBoard board;
        const auto& wall = board.wall();
        
        // Test negative indices
        assert(!wall.is_filled(-1, 0));
        assert(!wall.is_filled(0, -1));
        
        // Test too high indices
        assert(!wall.is_filled(10, 0));
        assert(!wall.is_filled(0, 10));
        
        // Test valid range
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 5; ++col) {
                // Should not crash and return valid boolean
                bool filled = wall.is_filled(row, col);
                (void)filled; // Suppress unused variable warning
            }
        }
        
        std::cout << "✓ Wall bounds tests passed" << std::endl;
    }
    
    static void test_copy_edge_cases() {
        std::cout << "Testing copy edge cases..." << std::endl;
        
        // Test copying game with complex state
        GameState game(3, 42);
        
        // Play several moves to create complex state
        for (int i = 0; i < 15 && !game.is_game_over(); ++i) {
            auto actions = game.get_legal_actions();
            if (actions.empty()) break;
            game.apply_action(actions[0]);
        }
        
        // Copy the complex state
        auto game_copy = game.copy();
        
        // Verify copy integrity
        assert(game.current_player() == game_copy.current_player());
        assert(game.round_number() == game_copy.round_number());
        assert(game.is_game_over() == game_copy.is_game_over());
        
        // Verify independence (modifying one doesn't affect the other)
        auto original_legal = game.get_legal_actions();
        auto copy_legal = game_copy.get_legal_actions();
        
        if (!original_legal.empty() && !game.is_game_over()) {
            game.apply_action(original_legal[0]);
            // Copy should still have same state as before
            auto copy_legal_after = game_copy.get_legal_actions();
            assert(copy_legal.size() == copy_legal_after.size());
        }
        
        std::cout << "✓ Copy edge cases tests passed" << std::endl;
    }
    
    static void test_extreme_penalty_scenarios() {
        std::cout << "Testing extreme penalty scenarios..." << std::endl;
        
        PlayerBoard board;
        
        // Fill floor line completely
        std::vector<Tile> max_penalties;
        for (int i = 0; i < 7; ++i) {
            max_penalties.push_back(Tile(TileColor::BLUE));
        }
        board.place_tiles_on_floor_line(max_penalties);
        
        // Add first player marker for extra penalty
        board.place_tiles_on_floor_line({Tile::create_first_player_marker()});
        
        // Score should handle extreme penalties
        auto [points, discard] = board.end_round_scoring();
        assert(board.score() >= 0); // Score should not go negative
        assert(points <= 0); // Should have penalty points
        
        std::cout << "✓ Extreme penalty scenarios tests passed" << std::endl;
    }
    
    static void run_all_tests() {
        std::cout << "Running C++ Edge Cases tests...\n" << std::endl;
        
        try {
            test_invalid_player_counts();
            test_empty_factories_and_center();
            test_invalid_actions();
            test_overflow_tiles_handling();
            test_wrong_color_on_pattern_line();
            test_wall_position_already_filled();
            test_floor_line_overflow();
            test_empty_tile_vectors();
            test_multiple_first_player_markers();
            test_pattern_line_bounds();
            test_wall_bounds();
            test_copy_edge_cases();
            test_extreme_penalty_scenarios();
            
            std::cout << "\n🎉 All edge cases tests passed!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\n❌ Test failed: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    TestEdgeCases::run_all_tests();
    return 0;
} 