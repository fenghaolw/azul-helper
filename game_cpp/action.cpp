#include "action.h"
#include <sstream>

namespace azul {

Action::Action(int source, TileColor color, int destination) 
    : source_(source), color_(color), destination_(destination) {
    compute_hash();
}

void Action::compute_hash() {
    // Combine hash values efficiently
    hash_ = std::hash<int>{}(source_);
    hash_ ^= std::hash<int>{}(static_cast<int>(color_)) + 0x9e3779b9 + (hash_ << 6) + (hash_ >> 2);
    hash_ ^= std::hash<int>{}(destination_) + 0x9e3779b9 + (hash_ << 6) + (hash_ >> 2);
}

auto Action::operator==(const Action& other) const -> bool {
    // Fast path: compare hash first (most likely to differ)
    if (hash_ != other.hash_) {
        return false;
    }
    return source_ == other.source_ && color_ == other.color_ && destination_ == other.destination_;
}

auto Action::to_string() const -> std::string {
    std::ostringstream oss;
    std::string source_str = (source_ == -1) ? "center" : "factory_" + std::to_string(source_);
    std::string dest_str = (destination_ == -1) ? "floor" : "line_" + std::to_string(destination_);
    
    oss << "Action(" << source_str << ", ";
    
    // Convert TileColor to string
    switch (color_) {
        case TileColor::BLUE: oss << "blue"; break;
        case TileColor::YELLOW: oss << "yellow"; break;
        case TileColor::RED: oss << "red"; break;
        case TileColor::BLACK: oss << "black"; break;
        case TileColor::WHITE: oss << "white"; break;
        case TileColor::FIRST_PLAYER: oss << "first_player"; break;
        default: oss << "unknown"; break;
    }
    
    oss << ", " << dest_str << ")";
    return oss.str();
}

} // namespace azul 