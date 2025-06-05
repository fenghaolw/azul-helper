#pragma once

#include "evaluation_config.h"
#include "azul_openspiel.h"
#include "mcts_agent.h"
#include "random_agent.h"
#include "minimax_agent.h"
#include <memory>
#include <functional>
#include <map>
#include <random>
#include <iostream>

namespace azul {

using GameStateType = open_spiel::State;
using ActionType = open_spiel::Action;

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
    bool enable_alpha_beta_;
    int seed_;
    
public:
    MinimaxAgentWrapper(int depth = 4, bool enable_alpha_beta = true, 
                       int seed = -1, 
                       const std::string& name = "MinimaxAgent")
        : agent_(std::make_unique<MinimaxAgent>(0, depth, enable_alpha_beta, seed)), 
          name_(name), depth_(depth), enable_alpha_beta_(enable_alpha_beta),
          seed_(seed) {}
    
    ActionType get_action(const GameStateType& state, int player_id) override {
        if (agent_->player_id() != player_id) {
            agent_ = std::make_unique<MinimaxAgent>(player_id, depth_, enable_alpha_beta_, seed_);
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
        agent_ = std::make_unique<MinimaxAgent>(0, depth_, enable_alpha_beta_, seed_);
    }
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
                    int seed = -1, 
                    const std::string& name = "MCTSAgent")
        : agent_(std::make_unique<AzulMCTSAgent>(0, num_simulations, uct_c, seed)), 
          name_(name), num_simulations_(num_simulations), uct_c_(uct_c),
          seed_(seed) {}
    
    ActionType get_action(const GameStateType& state, int player_id) override {
        if (agent_->player_id() != player_id) {
            agent_ = std::make_unique<AzulMCTSAgent>(player_id, num_simulations_, uct_c_, seed_);
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
        agent_ = std::make_unique<AzulMCTSAgent>(0, num_simulations_, uct_c_, seed_);
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
 * Main agent evaluator class with robust error handling.
 */
class AgentEvaluator {
public:
    AgentEvaluator(const EvaluationConfig& config = EvaluationConfig());
    
    /**
     * Evaluate a test agent against a baseline agent.
     */
    EvaluationResult evaluate_agent(
        EvaluationAgent& test_agent,
        EvaluationAgent& baseline_agent
    );
    
    /**
     * Quick evaluation with fewer games for rapid testing.
     */
    EvaluationResult quick_evaluation(
        EvaluationAgent& test_agent,
        EvaluationAgent& baseline_agent,
        int num_games = 20
    );

private:
    EvaluationConfig config_;
    
    /**
     * Run a single game between two agents.
     */
    GameResult run_single_game(
        EvaluationAgent& test_agent,
        EvaluationAgent& baseline_agent,
        int game_id,
        int test_agent_player,
        int baseline_agent_player,
        int seed
    );
    
    /**
     * Plan which games to play, including player position swapping.
     */
    std::vector<std::tuple<int, int, int, int>> plan_games() const;
    
    /**
     * Calculate statistical significance using binomial test.
     */
    std::pair<double, bool> calculate_statistical_significance(
        int test_wins, int total_games
    ) const;
    
    /**
     * Calculate confidence interval for win rate.
     */
    std::pair<double, double> calculate_confidence_interval(
        int wins, int total_games
    ) const;
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
    TournamentResult run_tournament();
    
    /**
     * Get the number of agents in the tournament.
     */
    size_t get_num_agents() const { return agents_.size(); }

private:
    EvaluationConfig config_;
    std::vector<std::unique_ptr<EvaluationAgent>> agents_;
    AgentEvaluator evaluator_;
};

// Factory functions for creating agent wrappers
std::unique_ptr<EvaluationAgent> create_random_evaluation_agent(
    int seed = -1, const std::string& name = "RandomAgent"
);

std::unique_ptr<EvaluationAgent> create_minimax_evaluation_agent(
    int depth = 4, bool enable_alpha_beta = true, 
    int seed = -1, const std::string& name = "MinimaxAgent"
);

std::unique_ptr<EvaluationAgent> create_mcts_evaluation_agent(
    int num_simulations = 1000, double uct_c = 1.4,
    int seed = -1, const std::string& name = "MCTSAgent"
);

} // namespace azul 