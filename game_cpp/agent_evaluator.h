#pragma once

#include <memory>
#include <random>
#include <utility>

#include "azul.h"
#include "evaluation_config.h"
#include "mcts_agent.h"
#include "minimax_agent.h"
#include "open_spiel/spiel.h"
#include "random_agent.h"

// Include LibTorch AlphaZero headers for model loading
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"

namespace azul {

// Type aliases for OpenSpiel integration
using GameStateType = open_spiel::State;
using ActionType = open_spiel::Action;
using AzulGame = open_spiel::azul::AzulGame;

/**
 * Agent interface for evaluation.
 * All agents must implement this interface to be evaluated.
 */
class EvaluationAgent {
 public:
  virtual ~EvaluationAgent() = default;

  // Core agent interface - agents adapt to current player
  virtual ActionType get_action(const GameStateType& state, int player_id) = 0;
  virtual std::string get_name() const = 0;
  virtual void reset() = 0;

  // Performance statistics (optional, return 0 if not supported)
  virtual size_t get_nodes_explored() const { return 0; }
  virtual double get_thinking_time() const { return 0.0; }
  virtual void reset_stats() {}
};

/**
 * Wrapper for RandomAgent that works with OpenSpiel states.
 */
class RandomAgentWrapper : public EvaluationAgent {
 private:
  std::mt19937 rng_;
  std::string name_;
  int seed_;

 public:
  RandomAgentWrapper(int seed = -1, const std::string& name = "RandomAgent")
      : name_(name), seed_(seed) {
    if (seed != -1) {
      rng_.seed(seed);
    } else {
      std::random_device rd;
      rng_.seed(rd());
    }
  }

  ActionType get_action(const GameStateType& state, int player_id) override {
    auto legal_actions = state.LegalActions();
    if (legal_actions.empty()) {
      throw std::runtime_error("No legal actions available");
    }

    std::uniform_int_distribution<size_t> dist(0, legal_actions.size() - 1);
    return legal_actions[dist(rng_)];
  }

  std::string get_name() const override { return name_; }
  void reset() override {
    if (seed_ != -1) {
      rng_.seed(seed_);
    }
  }
  void reset_stats() override {}
};

/**
 * Wrapper for MinimaxAgent that works with OpenSpiel states.
 */
class MinimaxAgentWrapper : public EvaluationAgent {
 private:
  std::unique_ptr<MinimaxAgent> agent_;
  std::string name_;
  int depth_;

 public:
  MinimaxAgentWrapper(int depth = 4, const std::string& name = "MinimaxAgent")
      : agent_(std::make_unique<MinimaxAgent>(0, depth)),
        name_(name),
        depth_(depth) {}

  ActionType get_action(const GameStateType& state, int player_id) override {
    if (agent_->player_id() != player_id) {
      agent_ = std::make_unique<MinimaxAgent>(player_id, depth_);
    }

    try {
      return agent_->get_action(state);
    } catch (const std::exception& e) {
      auto legal_actions = state.LegalActions();
      if (!legal_actions.empty()) {
        return legal_actions[0];
      }
      throw;
    }
  }

  std::string get_name() const override { return name_; }
  void reset() override { agent_ = std::make_unique<MinimaxAgent>(0, depth_); }
  void reset_stats() override { agent_->reset_stats(); }

  size_t get_nodes_explored() const override {
    return agent_->nodes_explored();
  }
};

/**
 * Wrapper for MCTS agent that works with OpenSpiel states.
 */
class MCTSAgentWrapper : public EvaluationAgent {
 private:
  std::unique_ptr<AzulMCTSAgent> agent_;
  std::string name_;
  int num_simulations_;
  double uct_c_;
  int seed_;

 public:
  MCTSAgentWrapper(int num_simulations = 1000, double uct_c = 1.4,
                   int seed = -1, const std::string& name = "MCTSAgent")
      : agent_(
            std::make_unique<AzulMCTSAgent>(0, num_simulations, uct_c, seed)),
        name_(name),
        num_simulations_(num_simulations),
        uct_c_(uct_c),
        seed_(seed) {}

  ActionType get_action(const GameStateType& state, int player_id) override {
    if (agent_->player_id() != player_id) {
      agent_ = std::make_unique<AzulMCTSAgent>(player_id, num_simulations_,
                                               uct_c_, seed_);
    }

    try {
      return agent_->get_action(state);
    } catch (const std::exception& e) {
      auto legal_actions = state.LegalActions();
      if (!legal_actions.empty()) {
        return legal_actions[0];
      }
      throw;
    }
  }

  std::string get_name() const override { return name_; }
  void reset() override {
    agent_ =
        std::make_unique<AzulMCTSAgent>(0, num_simulations_, uct_c_, seed_);
  }
  void reset_stats() override { agent_->reset_stats(); }

  size_t get_nodes_explored() const override {
    return agent_->get_nodes_explored();
  }

  double get_thinking_time() const override {
    return agent_->get_thinking_time();
  }
};

/**
 * Chance-aware evaluator that wraps VPNetEvaluator to handle chance nodes.
 * Routes chance nodes to random rollout and regular states to neural network.
 */
class ChanceAwareEvaluator : public open_spiel::algorithms::Evaluator {
 private:
  std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>
      vpnet_evaluator_;
  std::shared_ptr<open_spiel::algorithms::RandomRolloutEvaluator>
      fallback_evaluator_;

 public:
  ChanceAwareEvaluator(
      std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>
          vpnet_evaluator,
      std::shared_ptr<open_spiel::algorithms::RandomRolloutEvaluator>
          fallback_evaluator)
      : vpnet_evaluator_(std::move(vpnet_evaluator)),
        fallback_evaluator_(std::move(fallback_evaluator)) {}

  auto Evaluate(const open_spiel::State& state)
      -> std::vector<double> override {
    // Handle chance nodes with fallback evaluator
    if (state.CurrentPlayer() == -1) {
      return fallback_evaluator_->Evaluate(state);
    }
    // Use neural network for regular states
    return vpnet_evaluator_->Evaluate(state);
  }

  auto Prior(const open_spiel::State& state)
      -> open_spiel::ActionsAndProbs override {
    // Handle chance nodes with fallback evaluator
    if (state.CurrentPlayer() == -1) {
      return fallback_evaluator_->Prior(state);
    }
    // Use neural network for regular states
    return vpnet_evaluator_->Prior(state);
  }
};

/**
 * Wrapper for MCTS agent that uses a trained LibTorch AlphaZero model as
 * evaluator.
 */
class AlphaZeroMCTSAgentWrapper : public EvaluationAgent {
 private:
  std::shared_ptr<const open_spiel::Game> game_;
  std::unique_ptr<open_spiel::algorithms::torch_az::DeviceManager>
      device_manager_;
  std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator> evaluator_;
  std::unique_ptr<open_spiel::algorithms::MCTSBot> mcts_bot_;
  std::string name_;
  int num_simulations_;
  double uct_c_;
  int seed_;
  std::string model_path_;

  // Performance tracking
  mutable size_t nodes_explored_{};
  mutable double total_thinking_time_{};
  mutable int moves_played_{};

 public:
  AlphaZeroMCTSAgentWrapper(const std::string& checkpoint_path,
                            int num_simulations = 400, double uct_c = 1.4,
                            int seed = -1, std::string name = "AlphaZero_MCTS")
      : name_(std::move(name)),
        num_simulations_(num_simulations),
        uct_c_(uct_c),
        seed_(seed),
        model_path_(checkpoint_path) {
    // Load the Azul game
    game_ = open_spiel::LoadGame("azul");
    if (!game_) {
      throw std::runtime_error(
          "Failed to load Azul game for AlphaZero MCTS agent");
    }

    // Create the device manager and load the trained model
    try {
      device_manager_ =
          std::make_unique<open_spiel::algorithms::torch_az::DeviceManager>();

      // Create VPNet model from the checkpoint directory
      // (extract directory from full checkpoint path)
      std::string model_dir = checkpoint_path;
      size_t last_slash = model_dir.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        model_dir = model_dir.substr(0, last_slash);
      }

      // Create the VPNet model
      open_spiel::algorithms::torch_az::VPNetModel vpnet_model(
          *game_, model_dir, "vpnet.pb", "cpu");

      // Load the specific checkpoint
      vpnet_model.LoadCheckpoint(checkpoint_path);

      // Add the model to device manager
      device_manager_->AddDevice(std::move(vpnet_model));

      // Create the VPNet evaluator with the loaded model
      auto vpnet_evaluator =
          std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
              device_manager_.get(),
              1,     // batch_size
              1,     // threads
              10000  // cache_size
          );

      // Create fallback evaluator for chance nodes
      auto fallback_evaluator =
          std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
              1, seed != -1 ? seed : 42);

      // Create chance-aware evaluator that combines both
      auto chance_aware_evaluator = std::make_shared<ChanceAwareEvaluator>(
          vpnet_evaluator, fallback_evaluator);

      // Create MCTS bot with the chance-aware evaluator
      mcts_bot_ = std::make_unique<open_spiel::algorithms::MCTSBot>(
          *game_, chance_aware_evaluator, uct_c, num_simulations,
          1000,   // max_memory_mb
          false,  // solve
          seed,   // seed
          false   // verbose
      );

      std::cout << "✅ Successfully loaded AlphaZero model with chance-aware "
                   "evaluator from: "
                << checkpoint_path << '\n';

    } catch (const std::exception& e) {
      throw std::runtime_error("❌ Failed to load AlphaZero model from " +
                               checkpoint_path + ": " + e.what());
    }
  }

  auto get_action(const GameStateType& state, int player_id)
      -> ActionType override {
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
      // Check if this is a chance node (should not happen in get_action)
      if (state.CurrentPlayer() == -1) {
        throw std::runtime_error(
            "AlphaZero MCTS agent called on chance node! This should be "
            "handled by the evaluator.");
      }

      // Check if state is terminal
      if (state.IsTerminal()) {
        throw std::runtime_error(
            "AlphaZero MCTS agent called on terminal state!");
      }

      // Validate the state and player
      if (state.CurrentPlayer() != player_id) {
        throw std::runtime_error("State current player (" +
                                 std::to_string(state.CurrentPlayer()) +
                                 ") doesn't match expected player (" +
                                 std::to_string(player_id) + ")");
      }

      // Check if state is valid
      if (state.CurrentPlayer() < 0) {
        throw std::runtime_error("Invalid state: current player is " +
                                 std::to_string(state.CurrentPlayer()));
      }

      // Get action from MCTS bot with trained model
      ActionType action = mcts_bot_->Step(state);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);

      // Update performance statistics
      total_thinking_time_ += duration.count() / 1000.0;
      moves_played_++;
      nodes_explored_ += num_simulations_;  // Approximate

      return action;

    } catch (const std::exception& e) {
      throw std::runtime_error("Error in AlphaZero MCTS action selection: " +
                               std::string(e.what()));
    }
  }

  auto get_name() const -> std::string override { return name_; }

  void reset() override {
    // Reset performance tracking
    reset_stats();

    // Recreate MCTS bot to clear any internal state
    if (evaluator_) {
      mcts_bot_ = std::make_unique<open_spiel::algorithms::MCTSBot>(
          *game_, evaluator_, uct_c_, num_simulations_,
          1000,   // max_memory_mb
          false,  // solve
          seed_,  // seed
          false   // verbose
      );
    }
  }

  void reset_stats() override {
    nodes_explored_ = 0;
    total_thinking_time_ = 0.0;
    moves_played_ = 0;
  }

  auto get_nodes_explored() const -> size_t override { return nodes_explored_; }

  auto get_thinking_time() const -> double override {
    return total_thinking_time_;
  }
};

/**
 * Main agent evaluator class with robust error handling.
 */
class AgentEvaluator {
 public:
  AgentEvaluator(const EvaluationConfig& config = EvaluationConfig());

  /**
   * Evaluate a test agent against a baseline agent.
   */
  auto evaluate_agent(EvaluationAgent& test_agent,
                      EvaluationAgent& baseline_agent) -> EvaluationResult;

  /**
   * Quick evaluation with fewer games for rapid testing.
   */
  auto quick_evaluation(EvaluationAgent& test_agent,
                        EvaluationAgent& baseline_agent, int num_games = 20)
      -> EvaluationResult;

 private:
  EvaluationConfig config_;

  /**
   * Run a single game between two agents.
   */
  auto run_single_game(EvaluationAgent& test_agent,
                       EvaluationAgent& baseline_agent, int game_id,
                       int test_agent_player, int baseline_agent_player,
                       int seed) const -> GameResult;

  /**
   * Plan which games to play, including player position swapping.
   */
  auto plan_games() const -> std::vector<std::tuple<int, int, int, int>>;

  /**
   * Calculate statistical significance using binomial test.
   */
  static auto calculate_statistical_significance(int test_wins, int total_games)
      -> std::pair<double, bool>;

  /**
   * Calculate confidence interval for win rate.
   */
  auto calculate_confidence_interval(int wins, int total_games) const
      -> std::pair<double, double>;
};

/**
 * Tournament system for evaluating multiple agents.
 */
class Tournament {
 public:
  Tournament(const EvaluationConfig& config = EvaluationConfig());

  /**
   * Add an agent to the tournament.
   */
  void add_agent(std::unique_ptr<EvaluationAgent> agent);

  /**
   * Run a complete round-robin tournament.
   */
  auto run_tournament() -> TournamentResult;

  /**
   * Get the number of agents in the tournament.
   */
  auto get_num_agents() const -> size_t { return agents_.size(); }

 private:
  EvaluationConfig config_;
  std::vector<std::unique_ptr<EvaluationAgent>> agents_;
  AgentEvaluator evaluator_;
};

// Factory functions for creating agent wrappers
auto create_random_evaluation_agent(int seed = -1,
                                    const std::string& name = "RandomAgent")
    -> std::unique_ptr<EvaluationAgent>;

auto create_minimax_evaluation_agent(int depth = 4,
                                     const std::string& name = "MinimaxAgent")
    -> std::unique_ptr<EvaluationAgent>;

auto create_mcts_evaluation_agent(int num_simulations = 1000,
                                  double uct_c = 1.4, int seed = -1,
                                  const std::string& name = "MCTSAgent")
    -> std::unique_ptr<EvaluationAgent>;

auto create_alphazero_mcts_evaluation_agent(
    const std::string& checkpoint_path, int num_simulations = 400,
    double uct_c = 1.4, int seed = -1,
    const std::string& name = "AlphaZero_MCTS")
    -> std::unique_ptr<EvaluationAgent>;

}  // namespace azul