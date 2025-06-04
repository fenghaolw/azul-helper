#include "factory.h"
#include <algorithm>
#include <unordered_set>

namespace azul {

// Factory implementation
Factory::Factory() = default;

void Factory::fill_from_bag(std::vector<Tile>& bag) {
    tiles_.clear();
    for (int i = 0; i < 4 && !bag.empty(); ++i) {
        tiles_.push_back(bag.back());
        bag.pop_back();
    }
}

std::pair<std::vector<Tile>, std::vector<Tile>> Factory::take_tiles(TileColor color) {
    std::vector<Tile> taken;
    std::vector<Tile> remaining;
    
    for (const auto& tile : tiles_) {
        if (tile.color() == color) {
            taken.push_back(tile);
        } else {
            remaining.push_back(tile);
        }
    }
    
    tiles_.clear();
    return {taken, remaining};
}

auto Factory::is_empty() const -> bool {
    return tiles_.empty();
}

auto Factory::has_color(TileColor color) const -> bool {
    return std::any_of(tiles_.begin(), tiles_.end(),
                      [color](const Tile& tile) { return tile.color() == color; });
}

auto Factory::get_available_colors() const -> std::vector<TileColor> {
    std::vector<TileColor> colors;
    colors.reserve(5); // At most 5 different colors
    
    // Use simple vector with duplicate check - faster than unordered_set for small collections
    for (const auto& tile : tiles_) {
        if (tile.color() != TileColor::FIRST_PLAYER) {
            // Check if color already added (linear search is faster than hash for small vectors)
            bool found = false;
            for (const auto& existing_color : colors) {
                if (existing_color == tile.color()) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                colors.push_back(tile.color());
            }
        }
    }
    return colors;
}

auto Factory::copy() const -> Factory {
    Factory new_factory;
    new_factory.tiles_ = tiles_;
    return new_factory;
}

// CenterArea implementation
CenterArea::CenterArea() 
    : has_first_player_marker_(false), first_player_marker_(Tile::create_first_player_marker()) {}

void CenterArea::add_tiles(const std::vector<Tile>& tiles) {
    tiles_.insert(tiles_.end(), tiles.begin(), tiles.end());
}

void CenterArea::add_first_player_marker() {
    has_first_player_marker_ = true;
}

auto CenterArea::take_tiles(TileColor color) -> std::vector<Tile> {
    std::vector<Tile> taken;
    std::vector<Tile> remaining;
    
    for (const auto& tile : tiles_) {
        if (tile.color() == color) {
            taken.push_back(tile);
        } else {
            remaining.push_back(tile);
        }
    }
    
    tiles_ = remaining;
    
    // If taking tiles from center for first time this round,
    // also take first player marker
    if (!taken.empty() && has_first_player_marker_) {
        taken.push_back(first_player_marker_);
        has_first_player_marker_ = false;
    }
    
    return taken;
}

auto CenterArea::is_empty() const -> bool {
    return tiles_.empty();
}

auto CenterArea::has_color(TileColor color) const -> bool {
    return std::any_of(tiles_.begin(), tiles_.end(),
                      [color](const Tile& tile) { return tile.color() == color; });
}

auto CenterArea::get_available_colors() const -> std::vector<TileColor> {
    std::vector<TileColor> colors;
    colors.reserve(5); // At most 5 different colors
    
    // Use simple vector with duplicate check - faster than unordered_set for small collections
    for (const auto& tile : tiles_) {
        if (tile.color() != TileColor::FIRST_PLAYER) {
            // Check if color already added (linear search is faster than hash for small vectors)
            bool found = false;
            for (const auto& existing_color : colors) {
                if (existing_color == tile.color()) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                colors.push_back(tile.color());
            }
        }
    }
    return colors;
}

void CenterArea::clear() {
    tiles_.clear();
    has_first_player_marker_ = false;
}

auto CenterArea::copy() const -> CenterArea {
    CenterArea new_center;
    new_center.tiles_ = tiles_;
    new_center.has_first_player_marker_ = has_first_player_marker_;
    new_center.first_player_marker_ = first_player_marker_;
    return new_center;
}

// FactoryArea implementation
FactoryArea::FactoryArea(int num_players) 
    : num_factories_((2 * num_players) + 1), factories_(num_factories_) {}

void FactoryArea::setup_round(std::vector<Tile>& bag) {
    // Clear center area
    center_.clear();
    
    // Clear all factories first
    for (auto& factory : factories_) {
        factory = Factory(); // Reset to empty state
    }
    
    // Add first player marker to center
    center_.add_first_player_marker();
    
    // Fill each factory with 4 tiles
    for (auto& factory : factories_) {
        factory.fill_from_bag(bag);
    }
}

auto FactoryArea::take_from_factory(int factory_index, TileColor color) -> std::vector<Tile> {
    if (factory_index < 0 || factory_index >= static_cast<int>(factories_.size())) {
        return {};
    }
    
    auto [taken, remaining] = factories_[factory_index].take_tiles(color);
    
    // Add remaining tiles to center
    if (!remaining.empty()) {
        center_.add_tiles(remaining);
    }
    
    return taken;
}

auto FactoryArea::take_from_center(TileColor color) -> std::vector<Tile> {
    return center_.take_tiles(color);
}

auto FactoryArea::is_round_over() const -> bool {
    bool all_factories_empty = std::all_of(factories_.begin(), factories_.end(),
                                          [](const Factory& factory) { return factory.is_empty(); });
    
    // Center is considered empty if it has no regular tiles
    bool center_has_regular_tiles = std::any_of(center_.tiles().begin(), center_.tiles().end(),
                                               [](const Tile& tile) { return tile.color() != TileColor::FIRST_PLAYER; });
    
    return all_factories_empty && !center_has_regular_tiles;
}

auto FactoryArea::get_available_moves() const -> std::vector<std::pair<int, TileColor>> {
    std::vector<std::pair<int, TileColor>> moves;
    moves.reserve(50); // Reasonable estimate to avoid reallocations
    
    // Check center first (like Python)
    auto center_colors = center_.get_available_colors();
    for (const auto color : center_colors) {
        moves.emplace_back(-1, color);
    }
    
    // Check factories (like Python) - no need to check is_empty()
    for (int i = 0; i < static_cast<int>(factories_.size()); ++i) {
        auto factory_colors = factories_[i].get_available_colors();
        for (const auto color : factory_colors) {
            moves.emplace_back(i, color);
        }
    }
    
    return moves;
}

auto FactoryArea::copy() const -> FactoryArea {
    FactoryArea new_area(0); // Dummy num_players, will be overwritten
    new_area.num_factories_ = num_factories_;
    new_area.factories_.clear();
    new_area.factories_.reserve(factories_.size());
    for (const auto& factory : factories_) {
        new_area.factories_.push_back(factory.copy());
    }
    new_area.center_ = center_.copy();
    return new_area;
}

} // namespace azul 