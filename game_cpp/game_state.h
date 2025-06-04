#pragma once

#include "action.h"
#include "factory.h"
#include "player_board.h"
#include "tile.h"
#include <vector>
#include <random>

namespace azul {

class GameState {
public:
    explicit GameState(int num_players = 2, int seed = -1);
    
    // Game flow methods
    [[nodiscard]] auto get_legal_actions(int player_id = -1) const -> std::vector<Action>;
    [[nodiscard]] auto is_action_legal(const Action& action, int player_id = -1) const -> bool;
    [[nodiscard]] auto apply_action(const Action& action, bool skip_validation = false) -> bool;
    
    // Game state queries
    [[nodiscard]] auto is_game_over() const -> bool { return game_over_; }
    [[nodiscard]] auto get_winner() const -> int { return winner_; }
    [[nodiscard]] auto get_scores() const -> std::vector<int>;
    [[nodiscard]] auto get_state_vector() const -> std::vector<float>;
    
    // OpenSpiel compatibility
    [[nodiscard]] auto current_player() const -> int { return current_player_; }
    [[nodiscard]] auto num_players() const -> int { return num_players_; }
    
    // Copy/clone operations
    [[nodiscard]] auto copy() const -> GameState;
    
    // Getters for game components
    [[nodiscard]] auto players() const -> const std::vector<PlayerBoard>& { return players_; }
    [[nodiscard]] auto factory_area() const -> const FactoryArea& { return factory_area_; }
    [[nodiscard]] auto bag() const -> const std::vector<Tile>& { return bag_; }
    [[nodiscard]] auto discard_pile() const -> const std::vector<Tile>& { return discard_pile_; }
    [[nodiscard]] auto round_number() const -> int { return round_number_; }

private:
    int num_players_;
    int current_player_;
    bool game_over_;
    int winner_;
    int round_number_;
    
    std::vector<PlayerBoard> players_;
    FactoryArea factory_area_;
    std::vector<Tile> bag_;
    std::vector<Tile> discard_pile_;
    
    std::mt19937 rng_;
    
    // Helper methods
    void initialize_tiles();
    void start_new_round();
    [[nodiscard]] auto get_tiles_from_source(int source, TileColor color) const -> std::vector<Tile>;
    void next_player();
    void end_round();
    void end_game();
};

[[nodiscard]] auto create_game(int num_players = 2, int seed = -1) -> GameState;

} // namespace azul 