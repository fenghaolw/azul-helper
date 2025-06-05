#pragma once

#include <memory>
#include <string>

#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"

namespace azul {

/**
 * MCTS Agent for Azul using OpenSpiel's MCTS implementation
 */
class AzulMCTSAgent {
 public:
  AzulMCTSAgent(int player_id, int num_simulations = 1000, double uct_c = 1.4,
                int seed = -1);
  ~AzulMCTSAgent() = default;

  // Core agent interface
  [[nodiscard]] auto get_action(const open_spiel::State& state)
      -> open_spiel::Action;
  [[nodiscard]] static auto get_action_probabilities(
      const open_spiel::State& state, double temperature = 1.0)
      -> std::vector<double>;

  // Agent information
  [[nodiscard]] auto player_id() const -> int { return player_id_; }
  [[nodiscard]] auto get_name() const -> std::string { return name_; }

  // Performance tracking
  void reset();
  void reset_stats();
  [[nodiscard]] auto get_nodes_explored() const -> size_t {
    return nodes_explored_;
  }
  [[nodiscard]] auto get_thinking_time() const -> double {
    return total_thinking_time_;
  }

  // Configuration
  void set_num_simulations(int num_simulations) {
    num_simulations_ = num_simulations;
  }
  void set_uct_c(double uct_c) { uct_c_ = uct_c; }

 private:
  int player_id_;
  int num_simulations_;
  double uct_c_;
  int seed_;
  std::string name_;

  // Performance tracking
  size_t nodes_explored_;
  double total_thinking_time_;
  int moves_played_;

  // OpenSpiel MCTS components
  std::shared_ptr<const open_spiel::Game> game_;
  std::unique_ptr<open_spiel::algorithms::MCTSBot> mcts_bot_;
  std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator_;

  void initialize_mcts();
};

/**
 * Factory function to create MCTS agents
 */
std::unique_ptr<AzulMCTSAgent> create_mcts_agent(int player_id,
                                                 int num_simulations = 1000,
                                                 double uct_c = 1.4,
                                                 int seed = -1);

}  // namespace azul