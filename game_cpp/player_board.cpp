#include "player_board.h"
#include <algorithm>

namespace azul {

// Wall pattern constants - using int arrays instead of TileColor for performance
const std::array<std::array<int, 5>, 5> Wall::WALL_PATTERN = {{
    {{0, 1, 2, 3, 4}}, // BLUE=0, YELLOW=1, RED=2, BLACK=3, WHITE=4
    {{4, 0, 1, 2, 3}},
    {{3, 4, 0, 1, 2}},
    {{2, 3, 4, 0, 1}},
    {{1, 2, 3, 4, 0}}
}};

// COLOR_COLUMNS[row][color_index] = column where that color goes in that row
const std::array<std::array<int, 5>, 5> Wall::COLOR_COLUMNS = {{
    {{0, 1, 2, 3, 4}}, // Row 0: BLUE->0, YELLOW->1, RED->2, BLACK->3, WHITE->4
    {{1, 2, 3, 4, 0}}, // Row 1: BLUE->1, YELLOW->2, RED->3, BLACK->4, WHITE->0
    {{2, 3, 4, 0, 1}}, // Row 2: BLUE->2, YELLOW->3, RED->4, BLACK->0, WHITE->1
    {{3, 4, 0, 1, 2}}, // Row 3: BLUE->3, YELLOW->4, RED->0, BLACK->1, WHITE->2
    {{4, 0, 1, 2, 3}}  // Row 4: BLUE->4, YELLOW->0, RED->1, BLACK->2, WHITE->3
}};

const std::array<int, 7> PlayerBoard::FLOOR_PENALTIES = {{-1, -1, -2, -2, -2, -3, -3}};

// PatternLine implementation
PatternLine::PatternLine() : capacity_(0) {
}

PatternLine::PatternLine(int capacity) : capacity_(capacity) {
    tiles_.reserve(capacity);
}

PatternLine::PatternLine(const PatternLine& other) = default;

auto PatternLine::operator=(const PatternLine& other) -> PatternLine& {
    if (this != &other) {
        capacity_ = other.capacity_;
        tiles_ = other.tiles_;
        color_ = other.color_;
    }
    return *this;
}

auto PatternLine::can_add_tiles(const std::vector<Tile>& tiles) const -> bool {
    if (tiles.empty()) { 
        return false;
    }
    
    TileColor tile_color = tiles[0].color();
    
    // Can't add first player marker to pattern lines
    if (tile_color == TileColor::FIRST_PLAYER) {
        return false;
    }
    
    // Fast path: check capacity first
    if (static_cast<int>(tiles.size()) > capacity_) {
        return false;
    }
    
    // If line is empty, any valid color can be added
    if (tiles_.empty()) {
        return true;
    }
    
    // If line has tiles, new tiles must be same color and fit
    return color_.has_value() && color_.value() == tile_color && 
           static_cast<int>(tiles_.size() + tiles.size()) <= capacity_;
}

auto PatternLine::add_tiles(const std::vector<Tile>& tiles) -> std::vector<Tile> {
    if (!can_add_tiles(tiles)) {
        return tiles;
    }
    
    if (tiles_.empty()) {
        color_ = tiles[0].color();
    }
    
    std::vector<Tile> overflow;
    for (const auto& tile : tiles) {
        if (static_cast<int>(tiles_.size()) < capacity_) {
            tiles_.push_back(tile);
        } else {
            overflow.push_back(tile);
        }
    }
    
    return overflow;
}

auto PatternLine::is_complete() const -> bool {
    return static_cast<int>(tiles_.size()) == capacity_;
}

auto PatternLine::clear() -> std::pair<std::optional<Tile>, std::vector<Tile>> {
    if (!is_complete()) {
        return {std::nullopt, {}};
    }
    
    std::optional<Tile> wall_tile;
    std::vector<Tile> discard_tiles;
    
    if (!tiles_.empty()) {
        wall_tile = tiles_[0];
        if (tiles_.size() > 1) {
            discard_tiles.assign(tiles_.begin() + 1, tiles_.end());
        }
    }
    
    tiles_.clear();
    color_.reset();
    
    return {wall_tile, discard_tiles};
}

auto PatternLine::copy() const -> PatternLine {
    PatternLine new_line(capacity_);
    new_line.tiles_ = tiles_;
    new_line.color_ = color_;
    return new_line;
}

auto PatternLine::has_color(TileColor color) const -> bool {
    return color_.has_value() && color_.value() == color;
}

auto PatternLine::get_wall_tile() const -> std::optional<Tile> {
    if (is_complete() && !tiles_.empty()) {
        return tiles_[0];
    }
    return std::nullopt;
}

// Wall implementation
Wall::Wall() {
    for (auto& row : filled_) {
        row.fill(false);
    }
}

auto Wall::can_place_tile(int row, TileColor color) const -> bool {
    if (row < 0 || row >= 5) {
        return false;
    }
    
    int color_index = color_to_index(color);
    if (color_index < 0) {
        return false;
    }
    
    int col = COLOR_COLUMNS[row][color_index];
    return !filled_[row][col];
}

auto Wall::place_tile(int row, TileColor color) -> int {
    if (row < 0 || row >= 5) {
        return 0;
    }
    
    int color_index = color_to_index(color);
    if (color_index < 0) {
        return 0;
    }
    
    int col = COLOR_COLUMNS[row][color_index];
    filled_[row][col] = true;
    return calculate_points(row, col);
}

auto Wall::is_filled(int row, int col) const -> bool {
    if (row < 0 || row >= 5 || col < 0 || col >= 5) {
        return false;
    }
    return filled_[row][col];
}

auto Wall::calculate_points(int row, int col) const -> int {
    int points = 1;
    
    // Check horizontal connections
    int horizontal = 1;
    // Check left
    for (int c = col - 1; c >= 0; --c) {
        if (filled_[row][c]) {
            horizontal++;
        } else {
            break;
        }
    }
    // Check right
    for (int c = col + 1; c < 5; ++c) {
        if (filled_[row][c]) {
            horizontal++;
        } else {
            break;
        }
    }
    
    // Check vertical connections
    int vertical = 1;
    // Check up
    for (int r = row - 1; r >= 0; --r) {
        if (filled_[r][col]) {
            vertical++;
        } else {
            break;
        }
    }
    // Check down
    for (int r = row + 1; r < 5; ++r) {
        if (filled_[r][col]) {
            vertical++;
        } else {
            break;
        }
    }
    
    // If connected in both directions, add both
    if (horizontal > 1 && vertical > 1) {
        points = horizontal + vertical;
    } else if (horizontal > 1) {
        points = horizontal;
    } else if (vertical > 1) {
        points = vertical;
    }
    
    return points;
}

auto Wall::is_row_complete(int row) const -> bool {
    if (row < 0 || row >= 5) {
        return false;
    }
    for (int col = 0; col < 5; ++col) {
        if (!filled_[row][col]) {
            return false;
        }
    }
    return true;
}

auto Wall::is_column_complete(int col) const -> bool {
    if (col < 0 || col >= 5) {
        return false;
    }
    for (int row = 0; row < 5; ++row) {
        if (!filled_[row][col]) {
            return false;
        }
    }
    return true;
}

auto Wall::is_color_complete(TileColor color) const -> bool {
    int color_index = color_to_index(color);
    if (color_index < 0) {
        return false;
    }
    for (int row = 0; row < 5; ++row) {
        int col = COLOR_COLUMNS[row][color_index];
        if (!filled_[row][col]) {
            return false;
        }
    }
    return true;
}

auto Wall::get_completed_rows() const -> std::vector<int> {
    std::vector<int> completed;
    for (int row = 0; row < 5; ++row) {
        if (is_row_complete(row)) {
            completed.push_back(row);
        }
    }
    return completed;
}

auto Wall::get_completed_columns() const -> std::vector<int> {
    std::vector<int> completed;
    for (int col = 0; col < 5; ++col) {
        if (is_column_complete(col)) {
            completed.push_back(col);
        }
    }
    return completed;
}

auto Wall::get_completed_colors() const -> std::vector<TileColor> {
    std::vector<TileColor> completed;
    for (const auto color : {TileColor::BLUE, TileColor::YELLOW, TileColor::RED, 
                            TileColor::BLACK, TileColor::WHITE}) {
        if (is_color_complete(color)) {
            completed.push_back(color);
        }
    }
    return completed;
}

auto Wall::copy() const -> Wall {
    Wall new_wall;
    new_wall.filled_ = filled_;
    return new_wall;
}

auto Wall::color_to_index(TileColor color) -> int {
    switch (color) {
        case TileColor::BLUE: return 0;
        case TileColor::YELLOW: return 1;
        case TileColor::RED: return 2;
        case TileColor::BLACK: return 3;
        case TileColor::WHITE: return 4;
        default: return -1; // Invalid color
    }
}

// PlayerBoard implementation
PlayerBoard::PlayerBoard() : score_(0) {
    // Initialize pattern lines with capacities 1-5
    for (int i = 0; i < 5; ++i) {
        pattern_lines_[i] = PatternLine(i + 1);
    }
}

auto PlayerBoard::can_place_tiles_on_pattern_line(int line_index, const std::vector<Tile>& tiles) const -> bool {
    // Early bounds check
    if (line_index < 0 || line_index >= 5 || tiles.empty()) {
        return false;
    }
    
    TileColor color = tiles[0].color();
    const auto& pattern_line = pattern_lines_[line_index];
    
    // Inline pattern line validation (avoid method call overhead like Python does)
    
    // Can't add first player marker to pattern lines
    if (color == TileColor::FIRST_PLAYER) {
        return false;
    }
    
    // Fast capacity check first (most likely constraint)
    if (static_cast<int>(tiles.size()) > pattern_line.capacity()) {
        return false;
    }
    
    // If line is empty, any valid color can be added (still need to check wall)
    if (pattern_line.tiles().empty()) {
        // Check wall constraint only after pattern line validation passes
        return wall_.can_place_tile(line_index, color);
    }
    
    // If line has tiles, new tiles must be same color and fit
    if (!pattern_line.color().has_value() || 
        pattern_line.color().value() != color ||
        static_cast<int>(pattern_line.tiles().size() + tiles.size()) > pattern_line.capacity()) {
        return false;
    }
    
    // Only check wall constraint after pattern line validation passes (like Python)
    return wall_.can_place_tile(line_index, color);
}

auto PlayerBoard::place_tiles_on_pattern_line(int line_index, const std::vector<Tile>& tiles) -> std::vector<Tile> {
    if (line_index < 0 || line_index >= 5) {
        return tiles;
    }
    
    std::vector<Tile> overflow = pattern_lines_[line_index].add_tiles(tiles);
    return overflow;
}

auto PlayerBoard::place_tiles_on_floor_line(const std::vector<Tile>& tiles) -> std::vector<Tile> {
    std::vector<Tile> discarded;
    for (const auto& tile : tiles) {
        if (floor_line_.size() < 7) {  // Floor line can only hold 7 tiles
            floor_line_.push_back(tile);
        } else {
            discarded.push_back(tile);  // Excess tiles are discarded (returned to box) per Azul rules
        }
    }
    return discarded;
}

auto PlayerBoard::end_round_scoring() -> std::pair<int, std::vector<Tile>> {
    int points_scored = 0;
    std::vector<Tile> discard_tiles;
    
    // Score pattern lines
    for (int i = 0; i < 5; ++i) {
        auto [wall_tile, line_discard] = pattern_lines_[i].clear();
        if (wall_tile.has_value()) {
            points_scored += wall_.place_tile(i, wall_tile.value().color());
            
            // Add line discard tiles to discard pile
            discard_tiles.insert(discard_tiles.end(), line_discard.begin(), line_discard.end());
        }
    }
    
    // Score floor line penalties
    int penalty = 0;
    size_t penalty_tiles = std::min(floor_line_.size(), FLOOR_PENALTIES.size());
    for (size_t i = 0; i < penalty_tiles; ++i) {
        penalty += FLOOR_PENALTIES[i];
    }
    
    // Add floor tiles to discard (except first player marker)
    for (const auto& tile : floor_line_) {
        if (!tile.is_first_player_marker()) {
            discard_tiles.push_back(tile);
        }
    }
    
    floor_line_.clear();
    
    score_ += points_scored + penalty;
    score_ = std::max(0, score_); // Score can't go below 0
    
    return {points_scored + penalty, discard_tiles};
}

auto PlayerBoard::final_scoring() const -> int {
    int bonus_points = 0;
    
    // Completed rows: 2 points each
    bonus_points += static_cast<int>(wall_.get_completed_rows().size()) * 2;
    
    // Completed columns: 7 points each
    bonus_points += static_cast<int>(wall_.get_completed_columns().size()) * 7;
    
    // Completed colors: 10 points each
    bonus_points += static_cast<int>(wall_.get_completed_colors().size()) * 10;
    
    return bonus_points;
}

auto PlayerBoard::has_first_player_marker() const -> bool {
    return std::any_of(floor_line_.begin(), floor_line_.end(),
                      [](const Tile& tile) { return tile.is_first_player_marker(); });
}

auto PlayerBoard::remove_first_player_marker() -> bool {
    auto it = std::find_if(floor_line_.begin(), floor_line_.end(),
                          [](const Tile& tile) { return tile.is_first_player_marker(); });
    
    if (it != floor_line_.end()) {
        floor_line_.erase(it);
        return true;
    }
    return false;
}

auto PlayerBoard::copy() const -> PlayerBoard {
    PlayerBoard new_board;
    for (int i = 0; i < 5; ++i) {
        new_board.pattern_lines_[i] = pattern_lines_[i].copy();
    }
    new_board.wall_ = wall_.copy();
    new_board.floor_line_ = floor_line_;
    new_board.score_ = score_;
    return new_board;
}

} // namespace azul 