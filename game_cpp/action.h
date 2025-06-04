#pragma once

#include "tile.h"
#include <functional>
#include <string>

namespace azul {

class Action {
public:
    Action(int source, TileColor color, int destination);
    
    [[nodiscard]] auto source() const -> int { return source_; }
    [[nodiscard]] auto color() const -> TileColor { return color_; }
    [[nodiscard]] auto destination() const -> int { return destination_; }
    
    [[nodiscard]] auto operator==(const Action& other) const -> bool;
    [[nodiscard]] auto to_string() const -> std::string;
    
    // Hash support for use in std::unordered_set/map
    [[nodiscard]] auto hash() const -> std::size_t { return hash_; }

private:
    int source_;        // Factory index (-1 for center)
    TileColor color_;   // Color of tiles to take
    int destination_;   // Pattern line index (0-4) or -1 for floor line
    std::size_t hash_;  // Precomputed hash for performance
    
    void compute_hash();
};

} // namespace azul

// Hash specialization for Action
namespace std {
template<>
struct hash<azul::Action> {
    [[nodiscard]] auto operator()(const azul::Action& action) const -> std::size_t {
        return action.hash();
    }
};
} 