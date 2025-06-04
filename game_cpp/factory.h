#pragma once

#include "tile.h"
#include <vector>

namespace azul {

class Factory {
public:
    Factory();
    
    void fill_from_bag(std::vector<Tile>& bag);
    [[nodiscard]] auto take_tiles(TileColor color) -> std::pair<std::vector<Tile>, std::vector<Tile>>;
    [[nodiscard]] auto is_empty() const -> bool;
    [[nodiscard]] auto has_color(TileColor color) const -> bool;
    [[nodiscard]] auto get_available_colors() const -> std::vector<TileColor>;
    
    [[nodiscard]] auto copy() const -> Factory;
    
    [[nodiscard]] auto tiles() const -> const std::vector<Tile>& { return tiles_; }

private:
    std::vector<Tile> tiles_;
};

class CenterArea {
public:
    CenterArea();
    
    void add_tiles(const std::vector<Tile>& tiles);
    void add_first_player_marker();
    [[nodiscard]] auto take_tiles(TileColor color) -> std::vector<Tile>;
    [[nodiscard]] auto is_empty() const -> bool;
    [[nodiscard]] auto has_color(TileColor color) const -> bool;
    [[nodiscard]] auto get_available_colors() const -> std::vector<TileColor>;
    void clear();
    
    [[nodiscard]] auto copy() const -> CenterArea;
    
    [[nodiscard]] auto tiles() const -> const std::vector<Tile>& { return tiles_; }
    [[nodiscard]] auto has_first_player_marker() const -> bool { return has_first_player_marker_; }

private:
    std::vector<Tile> tiles_;
    bool has_first_player_marker_;
    Tile first_player_marker_;
};

class FactoryArea {
public:
    explicit FactoryArea(int num_players);
    
    void setup_round(std::vector<Tile>& bag);
    [[nodiscard]] auto take_from_factory(int factory_index, TileColor color) -> std::vector<Tile>;
    [[nodiscard]] auto take_from_center(TileColor color) -> std::vector<Tile>;
    [[nodiscard]] auto is_round_over() const -> bool;
    [[nodiscard]] auto get_available_moves() const -> std::vector<std::pair<int, TileColor>>;
    
    [[nodiscard]] auto copy() const -> FactoryArea;
    
    [[nodiscard]] auto num_factories() const -> int { return num_factories_; }
    [[nodiscard]] auto factories() const -> const std::vector<Factory>& { return factories_; }
    [[nodiscard]] auto center() const -> const CenterArea& { return center_; }

private:
    int num_factories_;
    std::vector<Factory> factories_;
    CenterArea center_;
};

} // namespace azul 