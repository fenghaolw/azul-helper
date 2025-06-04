#include "game_state.h"
#include <algorithm>
#include <stdexcept>
#include <optional>

namespace azul {

GameState::GameState(int num_players, int seed) 
    : num_players_(num_players), current_player_(0), game_over_(false), 
      winner_(-1), round_number_(1), factory_area_(num_players) {
    
    if (num_players < 2 || num_players > 4) {
        throw std::invalid_argument("Number of players must be between 2 and 4");
    }
    
    // Initialize random number generator
    if (seed != -1) {
        rng_.seed(static_cast<unsigned int>(seed));
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
    
    // Initialize players
    players_.resize(num_players);
    
    // Initialize game components
    initialize_tiles();
    start_new_round();
}

void GameState::initialize_tiles() {
    bag_ = Tile::create_standard_tiles();
    std::shuffle(bag_.begin(), bag_.end(), rng_);
    discard_pile_.clear();
}

void GameState::start_new_round() {
    // If bag doesn't have enough tiles, refill from discard pile
    int tiles_needed = factory_area_.num_factories() * 4;
    if (static_cast<int>(bag_.size()) < tiles_needed) {
        bag_.insert(bag_.end(), discard_pile_.begin(), discard_pile_.end());
        discard_pile_.clear();
        std::shuffle(bag_.begin(), bag_.end(), rng_);
    }
    
    // Setup factory area
    factory_area_.setup_round(bag_);
}

auto GameState::get_tiles_from_source(int source, TileColor color) const -> std::vector<Tile> {
    if (source == -1) {
        // Taking from center
        const auto& center_tiles = factory_area_.center().tiles();
        if (center_tiles.empty()) { 
            return {};
        }
        
        std::vector<Tile> result;
        for (const auto& tile : center_tiles) {
            if (tile.color() == color) {
                result.push_back(tile);
            }
        }
        return result;
    }          
    // Taking from factory
    const auto& factories = factory_area_.factories();
    if (source < 0 || source >= static_cast<int>(factories.size())) {
        return {};
    }
    
    const auto& factory_tiles = factories[source].tiles();
    if (factory_tiles.empty()) {
        return {};
    }
    
    std::vector<Tile> result;
    for (const auto& tile : factory_tiles) {
        if (tile.color() == color) {
            result.push_back(tile);
        }
    }
    return result;
   
}

auto GameState::get_legal_actions(int player_id) const -> std::vector<Action> {
    if (player_id == -1) {
        player_id = current_player_;
    }
    
    if (game_over_ || player_id < 0 || player_id >= num_players_) {
        return {};
    }
    
    std::vector<Action> actions;
    actions.reserve(100); // Pre-allocate to avoid reallocations
    
    // Get available moves directly without extra method calls
    const auto& factories = factory_area_.factories();
    const auto& center = factory_area_.center();
    const auto& player = players_[player_id];
    
    // Process center tiles first
    if (!center.tiles().empty()) {
        // Get unique colors directly without creating intermediate vectors
        bool has_color[5] = {false}; // BLUE=0, YELLOW=1, RED=2, BLACK=3, WHITE=4
        
        for (const auto& tile : center.tiles()) {
            if (tile.color() != TileColor::FIRST_PLAYER) {
                int color_idx = static_cast<int>(tile.color());
                if (color_idx >= 0 && color_idx < 5) {
                    has_color[color_idx] = true;
                }
            }
        }
        
        // For each unique color in center
        for (int color_idx = 0; color_idx < 5; ++color_idx) {
            if (has_color[color_idx]) {
                TileColor color = static_cast<TileColor>(color_idx);
                
                // Count tiles of this color for capacity checking
                int tile_count = 0;
                for (const auto& tile : center.tiles()) {
                    if (tile.color() == color) {
                        tile_count++;
                    }
                }
                
                // Floor line is always valid
                actions.emplace_back(-1, color, -1);
                
                // Check pattern lines directly
                for (int line_idx = 0; line_idx < 5; ++line_idx) {
                    const auto& pattern_line = player.pattern_lines()[line_idx];
                    
                    // Inline all validation to avoid method call overhead
                    if (color == TileColor::FIRST_PLAYER) continue;
                    if (tile_count > pattern_line.capacity()) continue;
                    
                    bool can_place = false;
                    if (pattern_line.tiles().empty()) {
                        // Empty line - check wall only
                        can_place = player.wall().can_place_tile(line_idx, color);
                    } else if (pattern_line.color().has_value() && 
                              pattern_line.color().value() == color &&
                              static_cast<int>(pattern_line.tiles().size()) + tile_count <= pattern_line.capacity()) {
                        // Line has same color and fits - check wall
                        can_place = player.wall().can_place_tile(line_idx, color);
                    }
                    
                    if (can_place) {
                        actions.emplace_back(-1, color, line_idx);
                    }
                }
            }
        }
    }
    
    // Process factory tiles
    for (int factory_idx = 0; factory_idx < static_cast<int>(factories.size()); ++factory_idx) {
        const auto& factory = factories[factory_idx];
        if (factory.tiles().empty()) continue;
        
        // Get unique colors directly
        bool has_color[5] = {false};
        
        for (const auto& tile : factory.tiles()) {
            if (tile.color() != TileColor::FIRST_PLAYER) {
                int color_idx = static_cast<int>(tile.color());
                if (color_idx >= 0 && color_idx < 5) {
                    has_color[color_idx] = true;
                }
            }
        }
        
        // For each unique color in factory
        for (int color_idx = 0; color_idx < 5; ++color_idx) {
            if (has_color[color_idx]) {
                TileColor color = static_cast<TileColor>(color_idx);
                
                // Count tiles of this color
                int tile_count = 0;
                for (const auto& tile : factory.tiles()) {
                    if (tile.color() == color) {
                        tile_count++;
                    }
                }
                
                // Floor line is always valid
                actions.emplace_back(factory_idx, color, -1);
                
                // Check pattern lines directly
                for (int line_idx = 0; line_idx < 5; ++line_idx) {
                    const auto& pattern_line = player.pattern_lines()[line_idx];
                    
                    // Inline all validation
                    if (color == TileColor::FIRST_PLAYER) continue;
                    if (tile_count > pattern_line.capacity()) continue;
                    
                    bool can_place = false;
                    if (pattern_line.tiles().empty()) {
                        can_place = player.wall().can_place_tile(line_idx, color);
                    } else if (pattern_line.color().has_value() && 
                              pattern_line.color().value() == color &&
                              static_cast<int>(pattern_line.tiles().size()) + tile_count <= pattern_line.capacity()) {
                        can_place = player.wall().can_place_tile(line_idx, color);
                    }
                    
                    if (can_place) {
                        actions.emplace_back(factory_idx, color, line_idx);
                    }
                }
            }
        }
    }
    
    // Check for round over condition
    if (actions.empty()) {
        // If no actions are available but round isn't over, something might be wrong
        // But we shouldn't end the game from get_legal_actions - that's apply_action's job
        return {};
    }
    
    return actions;
}

auto GameState::is_action_legal(const Action& action, int player_id) const -> bool {
    if (player_id == -1) {
        player_id = current_player_;
    }
    
    if (game_over_ || player_id < 0 || player_id >= num_players_) {
        return false;
    }
    
    auto legal_actions = get_legal_actions(player_id);
    return std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end();
}

auto GameState::apply_action(const Action& action, bool skip_validation) -> bool {
    if (!skip_validation && !is_action_legal(action)) {
        return false;
    }
    
    std::vector<Tile> taken_tiles;
    
    // Take tiles from source
    if (action.source() == -1) {
        taken_tiles = factory_area_.take_from_center(action.color());
    } else {
        taken_tiles = factory_area_.take_from_factory(action.source(), action.color());
    }
    
    if (taken_tiles.empty()) {
        return false;
    }
    
    auto& player = players_[current_player_];
    std::vector<Tile> overflow;
    
    // Place tiles on destination
    if (action.destination() == -1) {
        // Floor line
        overflow = player.place_tiles_on_floor_line(taken_tiles);
    } else {
        // Pattern line
        overflow = player.place_tiles_on_pattern_line(action.destination(), taken_tiles);
        // Any overflow goes to floor line
        if (!overflow.empty()) {
            auto floor_overflow = player.place_tiles_on_floor_line(overflow);
            // Floor line should accept all tiles, but just in case:
            if (!floor_overflow.empty()) {
                discard_pile_.insert(discard_pile_.end(), floor_overflow.begin(), floor_overflow.end());
            }
        }
    }
    
    // Check if round is over
    if (factory_area_.is_round_over()) {
        end_round();
    } else {
        next_player();
    }
    
    return true;
}

void GameState::next_player() {
    current_player_ = (current_player_ + 1) % num_players_;
}

void GameState::end_round() {
    // Score all players and collect discard tiles
    int next_first_player = -1;
    
    for (int i = 0; i < num_players_; ++i) {
        auto [points, discard_tiles] = players_[i].end_round_scoring();
        discard_pile_.insert(discard_pile_.end(), discard_tiles.begin(), discard_tiles.end());
        
        // Check for first player marker
        if (players_[i].remove_first_player_marker()) {
            next_first_player = i;
        }
    }
    
    // Check for game end condition (any player has a complete row)
    bool game_should_end = false;
    for (const auto& player : players_) {
        if (!player.wall().get_completed_rows().empty()) {
            game_should_end = true;
            break;
        }
    }
    
    if (game_should_end) {
        end_game();
        return;
    }
    
    // Set up next round
    round_number_++;
    current_player_ = (next_first_player != -1) ? next_first_player : 0;
    start_new_round();
}

void GameState::end_game() {
    game_over_ = true;
    
    // Calculate final scores
    std::vector<int> final_scores(num_players_);
    for (int i = 0; i < num_players_; ++i) {
        final_scores[i] = players_[i].score() + players_[i].final_scoring();
    }
    
    // Find winner (highest score)
    int max_score = *std::max_element(final_scores.begin(), final_scores.end());
    winner_ = static_cast<int>(std::distance(final_scores.begin(), 
                                           std::find(final_scores.begin(), final_scores.end(), max_score)));
}

auto GameState::get_scores() const -> std::vector<int> {
    std::vector<int> scores;
    scores.reserve(num_players_);
    for (const auto& player : players_) {
        scores.push_back(player.score());
    }
    return scores;
}

auto GameState::get_state_vector() const -> std::vector<float> {
    // Normalized state vector for ML/AI applications
    std::vector<float> state;
    state.reserve(200); // More realistic estimate
    
    // Game metadata (normalized)
    state.push_back(static_cast<float>(current_player_) / static_cast<float>(num_players_ - 1)); // [0,1]
    state.push_back(static_cast<float>(round_number_) / 20.0F); // Assume max ~20 rounds, normalize to [0,1]
    state.push_back(game_over_ ? 1.0F : 0.0F); // [0,1]
    
    // Player states
    for (const auto& player : players_) {
        // Normalize score (assume max reasonable score ~150)
        state.push_back(static_cast<float>(player.score()) / 150.0F);
        
        // Wall state (25 positions) - binary filled/not filled
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 5; ++col) {
                state.push_back(player.wall().is_filled(row, col) ? 1.0F : 0.0F);
            }
        }
        
        // Pattern lines state
        for (int i = 0; i < 5; ++i) {
            const auto& line = player.pattern_lines()[i];
            // Normalize tile count by capacity
            state.push_back(static_cast<float>(line.tiles().size()) / static_cast<float>(line.capacity()));
            
            // One-hot encode color (5 values for 5 colors)
            for (int color_idx = 0; color_idx < 5; ++color_idx) {
                bool has_this_color = line.color().has_value() && 
                                     static_cast<int>(line.color().value()) == color_idx;
                state.push_back(has_this_color ? 1.0F : 0.0F);
            }
        }
        
        // Floor line (normalize by max capacity ~7)
        state.push_back(std::min(static_cast<float>(player.floor_line().size()) / 7.0F, 1.0F));
        
        // First player marker
        state.push_back(player.has_first_player_marker() ? 1.0F : 0.0F);
    }
    
    // Factory area state
    for (const auto& factory : factory_area_.factories()) {
        // Color counts in each factory (normalized by 4 tiles max)
        std::array<int, 5> color_counts = {0}; // BLUE=0, YELLOW=1, RED=2, BLACK=3, WHITE=4
        for (const auto& tile : factory.tiles()) {
            int color_idx = static_cast<int>(tile.color());
            if (color_idx >= 0 && color_idx < 5) {
                color_counts[color_idx]++;
            }
        }
        for (int count : color_counts) {
            state.push_back(static_cast<float>(count) / 4.0F); // Max 4 tiles per factory
        }
    }
    
    // Center area state
    std::array<int, 5> center_color_counts = {0};
    for (const auto& tile : factory_area_.center().tiles()) {
        int color_idx = static_cast<int>(tile.color());
        if (color_idx >= 0 && color_idx < 5) {
            center_color_counts[color_idx]++;
        }
    }
    for (int count : center_color_counts) {
        state.push_back(std::min(static_cast<float>(count) / 20.0F, 1.0F)); // Normalize by reasonable max
    }
    
    // First player marker in center
    state.push_back(factory_area_.center().has_first_player_marker() ? 1.0F : 0.0F);
    
    return state;
}

auto GameState::copy() const -> GameState {
    GameState new_state(num_players_); // Use proper player count
    new_state.current_player_ = current_player_;
    new_state.game_over_ = game_over_;
    new_state.winner_ = winner_;
    new_state.round_number_ = round_number_;
    
    new_state.players_.clear();
    new_state.players_.reserve(players_.size());
    for (const auto& player : players_) {
        new_state.players_.push_back(player.copy());
    }
    
    new_state.factory_area_ = factory_area_.copy();
    new_state.bag_ = bag_;
    new_state.discard_pile_ = discard_pile_;
    new_state.rng_ = rng_;
    
    return new_state;
}

auto create_game(int num_players, int seed) -> GameState {
    return GameState(num_players, seed);
}

} // namespace azul 