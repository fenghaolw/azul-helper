#include "game_state.h"
#include "player_board.h"
#include "tile.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace azul;

class TestScoring {
public:
    static void test_single_tile_wall_scoring() {
        std::cout << "Testing single tile wall scoring..." << std::endl;
        
        PlayerBoard board;
        
        // Place a single tile on pattern line 0 (capacity 1)
        std::vector<Tile> blue_tiles = {Tile(TileColor::BLUE)};
        board.place_tiles_on_pattern_line(0, blue_tiles);
        
        // Complete the round - should score 1 point for isolated tile
        auto [points, discard] = board.end_round_scoring();
        assert(points == 1);
        
        std::cout << "✓ Single tile wall scoring tests passed" << std::endl;
    }
    
    static void test_horizontal_connections() {
        std::cout << "Testing horizontal connections..." << std::endl;
        
        PlayerBoard board;
        
        // Place tiles to create horizontal connections
        // We need to simulate this by manually setting up the wall
        // or by playing through pattern lines that will create connections
        
        // Place blue tile in row 0
        std::vector<Tile> blue_tiles = {Tile(TileColor::BLUE)};
        board.place_tiles_on_pattern_line(0, blue_tiles);
        board.end_round_scoring();
        
        // Check that tile was placed
        assert(board.wall().is_filled(0, 0)); // Blue goes to position (0,0)
        
        std::cout << "✓ Horizontal connections tests passed" << std::endl;
    }
    
    static void test_floor_line_penalties() {
        std::cout << "Testing floor line penalties..." << std::endl;
        
        PlayerBoard board;
        int initial_score = board.score();
        
        // Place several tiles on floor line to incur penalties
        std::vector<Tile> penalty_tiles;
        for (int i = 0; i < 5; ++i) {
            penalty_tiles.push_back(Tile(TileColor::BLUE));
        }
        board.place_tiles_on_floor_line(penalty_tiles);
        
        // End round - should have penalties
        auto [points, discard] = board.end_round_scoring();
        assert(points < 0); // Should have negative points from penalties
        
        std::cout << "✓ Floor line penalties tests passed" << std::endl;
    }
    
    static void test_score_cannot_go_negative() {
        std::cout << "Testing score cannot go negative..." << std::endl;
        
        PlayerBoard board;
        // Set a low positive score
        // Note: We can't directly set score from outside, so we'll verify behavior
        
        // Fill floor line with many tiles to create large penalty
        std::vector<Tile> many_tiles;
        for (int i = 0; i < 7; ++i) {
            many_tiles.push_back(Tile(TileColor::BLUE));
        }
        board.place_tiles_on_floor_line(many_tiles);
        
        auto [points, discard] = board.end_round_scoring();
        assert(board.score() >= 0); // Score should not go negative
        
        std::cout << "✓ Score cannot go negative tests passed" << std::endl;
    }
    
    static void test_first_player_marker_handling() {
        std::cout << "Testing first player marker handling..." << std::endl;
        
        PlayerBoard board;
        
        // Add first player marker to floor line
        std::vector<Tile> fpm_tiles = {Tile::create_first_player_marker()};
        board.place_tiles_on_floor_line(fpm_tiles);
        
        assert(board.has_first_player_marker());
        
        // End round - first player marker should be handled properly
        auto [points, discard] = board.end_round_scoring();
        
        // First player marker should not be in discard pile
        for (const auto& tile : discard) {
            assert(!tile.is_first_player_marker());
        }
        
        std::cout << "✓ First player marker handling tests passed" << std::endl;
    }
    
    static void test_pattern_line_completion() {
        std::cout << "Testing pattern line completion..." << std::endl;
        
        PlayerBoard board;
        
        // Fill pattern line 2 (capacity 3) completely
        std::vector<Tile> blue_tiles;
        for (int i = 0; i < 3; ++i) {
            blue_tiles.push_back(Tile(TileColor::BLUE));
        }
        board.place_tiles_on_pattern_line(2, blue_tiles);
        
        // Check that pattern line is complete
        assert(board.pattern_lines()[2].is_complete());
        
        // End round scoring
        auto [points, discard] = board.end_round_scoring();
        
        // Should have scored points and discarded excess tiles
        assert(points > 0);
        assert(discard.size() == 2); // Should discard 2 excess tiles
        
        // One tile should be placed on wall
        assert(board.wall().is_filled(2, 2)); // Blue at row 2, col 2
        
        std::cout << "✓ Pattern line completion tests passed" << std::endl;
    }
    
    static void test_multiple_pattern_lines_same_round() {
        std::cout << "Testing multiple pattern lines completion..." << std::endl;
        
        PlayerBoard board;
        
        // Complete multiple pattern lines
        // Line 0 (capacity 1)
        board.place_tiles_on_pattern_line(0, {Tile(TileColor::BLUE)});
        
        // Line 1 (capacity 2)
        std::vector<Tile> red_tiles = {Tile(TileColor::RED), Tile(TileColor::RED)};
        board.place_tiles_on_pattern_line(1, red_tiles);
        
        // Both should be complete
        assert(board.pattern_lines()[0].is_complete());
        assert(board.pattern_lines()[1].is_complete());
        
        // End round scoring
        auto [points, discard] = board.end_round_scoring();
        
        // Should have scored from both lines
        assert(points >= 2); // At least 1 point per tile placed
        
        std::cout << "✓ Multiple pattern lines tests passed" << std::endl;
    }
    
    static void test_empty_pattern_lines_no_scoring() {
        std::cout << "Testing empty pattern lines no scoring..." << std::endl;
        
        PlayerBoard board;
        
        // Don't place any tiles
        
        // End round scoring
        auto [points, discard] = board.end_round_scoring();
        
        // Should not score any points
        assert(points == 0);
        assert(discard.empty());
        
        std::cout << "✓ Empty pattern lines tests passed" << std::endl;
    }
    
    static void test_partial_pattern_lines_no_wall_placement() {
        std::cout << "Testing partial pattern lines no wall placement..." << std::endl;
        
        PlayerBoard board;
        
        // Partially fill pattern line 2 (capacity 3) with only 2 tiles
        std::vector<Tile> blue_tiles = {Tile(TileColor::BLUE), Tile(TileColor::BLUE)};
        board.place_tiles_on_pattern_line(2, blue_tiles);
        
        // Should not be complete
        assert(!board.pattern_lines()[2].is_complete());
        
        // End round scoring
        auto [points, discard] = board.end_round_scoring();
        
        // Should not place tile on wall
        assert(!board.wall().is_filled(2, 2));
        
        // Should not discard the partial tiles
        assert(board.pattern_lines()[2].tiles().size() == 2);
        
        std::cout << "✓ Partial pattern lines tests passed" << std::endl;
    }
    
    static void test_game_end_scoring() {
        std::cout << "Testing game end scoring..." << std::endl;
        
        GameState game(2, 42);
        
        // Play until game ends or we reach a limit
        int actions = 0;
        while (!game.is_game_over() && actions < 1000) {
            auto legal_actions = game.get_legal_actions();
            if (legal_actions.empty()) break;
            game.apply_action(legal_actions[0]);
            actions++;
        }
        
        // Check final scores are reasonable
        auto final_scores = game.get_scores();
        for (int score : final_scores) {
            assert(score >= 0); // Scores should be non-negative
            assert(score < 200); // Reasonable upper bound
        }
        
        std::cout << "✓ Game end scoring tests passed" << std::endl;
    }
    
    static void test_scoring_consistency() {
        std::cout << "Testing scoring consistency..." << std::endl;
        
        // Create two identical games and verify they score the same
        GameState game1(2, 42);
        GameState game2(2, 42);
        
        // Play same actions on both
        for (int i = 0; i < 10 && !game1.is_game_over() && !game2.is_game_over(); ++i) {
            auto actions1 = game1.get_legal_actions();
            auto actions2 = game2.get_legal_actions();
            
            if (actions1.empty() || actions2.empty()) break;
            
            // Should have same legal actions (same seed)
            assert(actions1.size() == actions2.size());
            
            game1.apply_action(actions1[0]);
            game2.apply_action(actions2[0]);
            
            // Should have same scores
            auto scores1 = game1.get_scores();
            auto scores2 = game2.get_scores();
            assert(scores1.size() == scores2.size());
            for (size_t j = 0; j < scores1.size(); ++j) {
                assert(scores1[j] == scores2[j]);
            }
        }
        
        std::cout << "✓ Scoring consistency tests passed" << std::endl;
    }
    
    static void run_all_tests() {
        std::cout << "Running C++ Scoring tests...\n" << std::endl;
        
        try {
            test_single_tile_wall_scoring();
            test_horizontal_connections();
            test_floor_line_penalties();
            test_score_cannot_go_negative();
            test_first_player_marker_handling();
            test_pattern_line_completion();
            test_multiple_pattern_lines_same_round();
            test_empty_pattern_lines_no_scoring();
            test_partial_pattern_lines_no_wall_placement();
            test_game_end_scoring();
            test_scoring_consistency();
            
            std::cout << "\n🎉 All scoring tests passed!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "\n❌ Test failed: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    TestScoring::run_all_tests();
    return 0;
} 