#include "minimax_agent.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "azul.h"
#include "open_spiel/algorithms/minimax.h"

namespace azul {

// Floor line penalty points (from azul.cc)
const std::vector<int> kFloorPenalties = {-1, -1, -2, -2, -2, -3, -3};

MinimaxAgent::MinimaxAgent(int player_id, int depth)
    : player_id_(player_id), depth_(depth) {}

auto MinimaxAgent::get_action(const GameStateType& state) -> ActionType {
  if (state.IsTerminal()) {
    throw std::runtime_error("Cannot get action from terminal state");
  }

  // Reset statistics for this search
  nodes_explored_ = 0;

  auto legal_actions = state.LegalActions();

  if (legal_actions.empty()) {
    throw std::runtime_error("No legal actions available");
  }

  if (legal_actions.size() == 1) {
    return legal_actions[0];
  }

  // Create a custom evaluation function that captures our player_id
  auto evaluation_function = [this](const open_spiel::State& s) -> double {
    return this->evaluate_state(s);
  };

  auto game = state.GetGame();
  std::pair<double, open_spiel::Action> result;

  try {
    // OpenSpiel's minimax algorithms have specific requirements:
    // - AlphaBetaSearch: only works with kDeterministic games
    // - ExpectiminimaxSearch: works with stochastic games but may have its own
    // limitations

    auto game_type = game->GetType();

    if (game_type.chance_mode ==
        open_spiel::GameType::ChanceMode::kDeterministic) {
      // Use AlphaBetaSearch for deterministic games only
      result = open_spiel::algorithms::AlphaBetaSearch(
          *game, &state, evaluation_function, depth_, player_id_);
    } else if (game_type.utility == open_spiel::GameType::Utility::kZeroSum) {
      // Try ExpectiminimaxSearch for stochastic zero-sum games
      result = open_spiel::algorithms::ExpectiminimaxSearch(
          *game, &state, evaluation_function, depth_, player_id_);
    } else {
      // For general-sum stochastic games, OpenSpiel's algorithms may not work
      // Fall back to our own simple minimax-like evaluation
      throw std::runtime_error(
          "Unsupported game type for OpenSpiel minimax algorithms");
    }
    // Update node exploration count (approximation)
    nodes_explored_ = std::pow(legal_actions.size(), std::min(depth_, 3));
    return result.second;

  } catch (const std::exception& e) {
    throw std::runtime_error("OpenSpiel minimax search failed: " +
                             std::string(e.what()));
  }
}

void MinimaxAgent::reset_stats() {
  nodes_explored_ = 0;
}

auto MinimaxAgent::get_action_probabilities(const GameStateType& state)
    -> std::vector<double> {
  if (state.IsTerminal()) {
    return {};
  }

  auto legal_actions = state.LegalActions();

  if (legal_actions.empty()) {
    return {};
  }

  // Minimax is deterministic, so best action gets probability 1.0
  std::vector<double> probabilities(legal_actions.size(), 0.0);

  ActionType best_action = get_action(state);

  // Find the index of the best action
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    if (legal_actions[i] == best_action) {
      probabilities[i] = 1.0;
      break;
    }
  }

  return probabilities;
}

auto MinimaxAgent::evaluate_state(const GameStateType& state) const -> double {
  // Cast to AzulState to access Azul-specific methods
  const auto* azul_state =
      dynamic_cast<const open_spiel::azul::AzulState*>(&state);
  if (azul_state == nullptr) {
    return 0.0;  // Fallback if cast fails
  }

  if (state.IsTerminal()) {
    // Terminal state evaluation - return actual score difference
    auto returns = state.Returns();
    if (player_id_ >= 0 && player_id_ < static_cast<int>(returns.size())) {
      return returns[player_id_];
    }
    return 0.0;
  }

  // For non-terminal states, implement Azul heuristic evaluation
  // Simulate what would happen if the round ended immediately
  double our_projected_score =
      calculate_round_end_score(*azul_state, player_id_);

  // Calculate best opponent projected score
  double best_opponent_projected_score = -1000.0;
  int num_players = azul_state->PlayerBoards().size();

  for (int player = 0; player < num_players; ++player) {
    if (player != player_id_) {
      double opponent_score = calculate_round_end_score(*azul_state, player);
      best_opponent_projected_score =
          std::max(best_opponent_projected_score, opponent_score);
    }
  }

  return our_projected_score - best_opponent_projected_score;
}

auto MinimaxAgent::calculate_round_end_score(
    const open_spiel::azul::AzulState& state, int player) -> double {
  const auto& player_boards = state.PlayerBoards();
  if (player >= static_cast<int>(player_boards.size())) {
    return 0.0;
  }

  const auto& board = player_boards[player];
  double current_score = static_cast<double>(board.score);
  double projected_additional_score = 0.0;

  // Check each pattern line for completion and simulate scoring
  for (int line_idx = 0; line_idx < open_spiel::azul::kNumPatternLines;
       ++line_idx) {
    const auto& pattern_line = board.pattern_lines[line_idx];
    int line_capacity = line_idx + 1;

    if (pattern_line.count == line_capacity && pattern_line.count > 0) {
      // This line is complete, simulate placing on wall

      // Find the wall column for this color and line
      int wall_col = -1;
      for (int col = 0; col < open_spiel::azul::kWallSize; ++col) {
        if (open_spiel::azul::kWallPattern[line_idx][col] ==
            pattern_line.color) {
          wall_col = col;
          break;
        }
      }

      // Check if we can place this tile (wall position should be empty)
      if (wall_col != -1 && !board.wall[line_idx][wall_col]) {
        // Simulate the scoring for placing this tile
        double tile_score =
            simulate_wall_scoring(board.wall, line_idx, wall_col);
        projected_additional_score += tile_score;
      }
    }
  }

  // Calculate floor line penalties
  double floor_penalty = 0.0;
  size_t floor_size = board.floor_line.size();
  for (size_t i = 0; i < floor_size && i < kFloorPenalties.size(); ++i) {
    floor_penalty += static_cast<double>(kFloorPenalties[i]);
  }

  return current_score + projected_additional_score +
         floor_penalty;  // floor_penalty is already negative
}

auto MinimaxAgent::simulate_wall_scoring(
    const std::array<std::array<bool, open_spiel::azul::kWallSize>,
                     open_spiel::azul::kWallSize>& wall,
    int row, int col) -> double {
  double score = 1.0;  // Base score for the tile

  // Count horizontal connections
  int horizontal_length = 1;

  // Check left
  for (int c = col - 1; c >= 0; --c) {
    if (wall[row][c]) {
      horizontal_length++;
    } else {
      break;
    }
  }

  // Check right
  for (int c = col + 1; c < open_spiel::azul::kWallSize; ++c) {
    if (wall[row][c]) {
      horizontal_length++;
    } else {
      break;
    }
  }

  // Count vertical connections
  int vertical_length = 1;

  // Check up
  for (int r = row - 1; r >= 0; --r) {
    if (wall[r][col]) {
      vertical_length++;
    } else {
      break;
    }
  }

  // Check down
  for (int r = row + 1; r < open_spiel::azul::kWallSize; ++r) {
    if (wall[r][col]) {
      vertical_length++;
    } else {
      break;
    }
  }

  // Apply scoring rules (based on EndRoundScoring logic in azul.cc)
  if (horizontal_length > 1) {
    score = static_cast<double>(horizontal_length);
  }

  if (vertical_length > 1) {
    if (horizontal_length > 1) {
      // Both horizontal and vertical connections
      score += static_cast<double>(vertical_length - 1);
    } else {
      // Only vertical connections
      score = static_cast<double>(vertical_length);
    }
  }

  return score;
}

}  // namespace azul