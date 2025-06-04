#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace azul {

enum class TileColor : int {
    BLUE = 0,
    YELLOW = 1,
    RED = 2,
    BLACK = 3,
    WHITE = 4,
    FIRST_PLAYER = 5
};

class Tile {
public:
    explicit Tile(TileColor color);
    
    [[nodiscard]] auto color() const -> TileColor { return color_; }
    [[nodiscard]] auto is_first_player_marker() const -> bool { return color_ == TileColor::FIRST_PLAYER; }
    
    [[nodiscard]] auto operator==(const Tile& other) const -> bool { return color_ == other.color_; }
    [[nodiscard]] auto to_string() const -> std::string;
    
    // Static factory methods for efficient tile management
    [[nodiscard]] static auto get_tile(TileColor color) -> const Tile&;
    [[nodiscard]] static auto create_standard_tiles() -> std::vector<Tile>;
    [[nodiscard]] static auto create_first_player_marker() -> const Tile&;

private:
    TileColor color_;
    
    // Tile pool for memory efficiency
    static std::unordered_map<TileColor, std::unique_ptr<Tile>> tile_pool_;
};

// Hash function for TileColor
struct TileColorHash {
    [[nodiscard]] auto operator()(const TileColor& color) const -> std::size_t {
        return static_cast<std::size_t>(color);
    }
};

} // namespace azul 