#include "tile.h"
#include <unordered_map>

namespace azul {

std::unordered_map<TileColor, std::unique_ptr<Tile>> Tile::tile_pool_;

Tile::Tile(TileColor color) : color_(color) {}

auto Tile::to_string() const -> std::string {
    static const std::unordered_map<TileColor, std::string> color_names = {
        {TileColor::BLUE, "blue"},
        {TileColor::YELLOW, "yellow"},
        {TileColor::RED, "red"},
        {TileColor::BLACK, "black"},
        {TileColor::WHITE, "white"},
        {TileColor::FIRST_PLAYER, "first_player"}
    };
    
    auto it = color_names.find(color_);
    return (it != color_names.end()) ? it->second : "unknown";
}

auto Tile::get_tile(TileColor color) -> const Tile& {
    auto it = tile_pool_.find(color);
    if (it == tile_pool_.end()) {
        tile_pool_[color] = std::make_unique<Tile>(color);
    }
    return *tile_pool_[color];
}

auto Tile::create_standard_tiles() -> std::vector<Tile> {
    std::vector<Tile> tiles;
    tiles.reserve(100); // 20 * 5 colors
    
    for (const auto color : {TileColor::BLUE, TileColor::YELLOW, TileColor::RED, 
                            TileColor::BLACK, TileColor::WHITE}) {
        const Tile& tile_instance = get_tile(color);
        for (int i = 0; i < 20; ++i) {
            tiles.push_back(tile_instance);
        }
    }
    
    return tiles;
}

auto Tile::create_first_player_marker() -> const Tile& {
    return get_tile(TileColor::FIRST_PLAYER);
}

} // namespace azul 