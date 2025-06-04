#pragma once

#include "tile.h"
#include <vector>
#include <array>
#include <optional>
#include <utility>

namespace azul {

class PatternLine {
public:
    PatternLine(); // Default constructor
    explicit PatternLine(int capacity);
    PatternLine(const PatternLine& other);
    auto operator=(const PatternLine& other) -> PatternLine&;
    
    [[nodiscard]] auto capacity() const -> int { return capacity_; }
    [[nodiscard]] auto tiles() const -> const std::vector<Tile>& { return tiles_; }
    [[nodiscard]] auto color() const -> std::optional<TileColor> { return color_; }
    [[nodiscard]] auto has_color(TileColor color) const -> bool;
    
    [[nodiscard]] auto can_add_tiles(const std::vector<Tile>& tiles) const -> bool;
    [[nodiscard]] auto add_tiles(const std::vector<Tile>& tiles) -> std::vector<Tile>;
    [[nodiscard]] auto is_complete() const -> bool;
    [[nodiscard]] auto clear() -> std::pair<std::optional<Tile>, std::vector<Tile>>;
    [[nodiscard]] auto get_wall_tile() const -> std::optional<Tile>;
    
    [[nodiscard]] auto copy() const -> PatternLine;

private:
    int capacity_;
    std::vector<Tile> tiles_;
    std::optional<TileColor> color_;
};

class Wall {
public:
    Wall();
    
    [[nodiscard]] auto can_place_tile(int row, TileColor color) const -> bool;
    [[nodiscard]] auto place_tile(int row, TileColor color) -> int;
    [[nodiscard]] auto is_filled(int row, int col) const -> bool;
    
    [[nodiscard]] auto is_row_complete(int row) const -> bool;
    [[nodiscard]] auto is_column_complete(int col) const -> bool;
    [[nodiscard]] auto is_color_complete(TileColor color) const -> bool;
    
    [[nodiscard]] auto get_completed_rows() const -> std::vector<int>;
    [[nodiscard]] auto get_completed_columns() const -> std::vector<int>;
    [[nodiscard]] auto get_completed_colors() const -> std::vector<TileColor>;
    
    [[nodiscard]] auto copy() const -> Wall;

private:
    std::array<std::array<bool, 5>, 5> filled_;
    
    // Simple 2D array for wall pattern - much faster than maps
    static const std::array<std::array<int, 5>, 5> WALL_PATTERN;
    // COLOR_COLUMNS[row][color_index] = column, where color_index = 0-4 for BLUE,YELLOW,RED,BLACK,WHITE
    static const std::array<std::array<int, 5>, 5> COLOR_COLUMNS;
    
    [[nodiscard]] auto calculate_points(int row, int col) const -> int;
    [[nodiscard]] static auto color_to_index(TileColor color) -> int;
};

class PlayerBoard {
public:
    PlayerBoard();
    
    [[nodiscard]] auto can_place_tiles_on_pattern_line(int line_index, const std::vector<Tile>& tiles) const -> bool;
    [[nodiscard]] auto place_tiles_on_pattern_line(int line_index, const std::vector<Tile>& tiles) -> std::vector<Tile>;
    [[nodiscard]] auto place_tiles_on_floor_line(const std::vector<Tile>& tiles) -> std::vector<Tile>;
    
    [[nodiscard]] auto end_round_scoring() -> std::pair<int, std::vector<Tile>>;
    [[nodiscard]] auto final_scoring() const -> int;
    
    [[nodiscard]] auto has_first_player_marker() const -> bool;
    [[nodiscard]] auto remove_first_player_marker() -> bool;
    
    [[nodiscard]] auto copy() const -> PlayerBoard;
    
    // Getters
    [[nodiscard]] auto pattern_lines() const -> const std::array<PatternLine, 5>& { return pattern_lines_; }
    [[nodiscard]] auto wall() const -> const Wall& { return wall_; }
    [[nodiscard]] auto floor_line() const -> const std::vector<Tile>& { return floor_line_; }
    [[nodiscard]] auto score() const -> int { return score_; }

private:
    std::array<PatternLine, 5> pattern_lines_;
    Wall wall_;
    std::vector<Tile> floor_line_;
    int score_;
    
    static const std::array<int, 7> FLOOR_PENALTIES;
};

} // namespace azul 