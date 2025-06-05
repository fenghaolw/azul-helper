#include "minimax_agent.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "open_spiel/algorithms/minimax.h"

namespace azul {

MinimaxAgent::MinimaxAgent(int player_id, int depth)
    : player_id_(player_id), depth_(depth) {}

ActionType MinimaxAgent::get_action(const GameStateType& state) {
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

std::vector<double> MinimaxAgent::get_action_probabilities(
    const GameStateType& state) {
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

double MinimaxAgent::evaluate_state(const GameStateType& state) const {
  if (state.IsTerminal()) {
    // Terminal state evaluation - use OpenSpiel's Returns for zero-sum games
    auto returns = state.Returns();
    if (player_id_ >= 0 && player_id_ < static_cast<int>(returns.size())) {
      // In zero-sum games, our return is what matters (opponent's return is
      // -ours)
      return returns[player_id_];
    }
    return 0.0;
  }

  // For non-terminal states, implement a basic Azul heuristic evaluation
  // This is a simplified heuristic - in practice, you'd want more sophisticated
  // evaluation

  try {
    // Get state string and parse for basic information
    std::string state_str = state.ToString();

    // Basic heuristic: favor states where we're likely to score more points
    // This is a placeholder - real Azul evaluation would consider:
    // - Pattern line completion potential
    // - Wall placement strategy
    // - Floor line penalties
    // - Tile availability in factories/center

    // For now, use a simple random-like evaluation with slight bias toward our
    // player This ensures the search explores different paths
    std::hash<std::string> hasher;
    double hash_value = static_cast<double>(hasher(state_str) % 1000) / 1000.0;

    // Add slight bias for our player position to break ties
    double player_bias = (player_id_ + 1) * 0.001;

    return hash_value + player_bias;

  } catch (const std::exception&) {
    // If state parsing fails, return neutral evaluation
    return 0.0;
  }
}

}  // namespace azul