#pragma once

#include "evaluation_config.h"
#include "game_state.h"
#include "random_agent.h"
#include "minimax_agent.h"
#include <memory>
#include <functional>
#include <map>

namespace azul {

/**
 * Agent interface for evaluation.
 * All agents must implement this interface to be evaluated.
 */
class EvaluationAgent {
public:
    virtual ~EvaluationAgent() = default;
    
    // Core agent interface - agents adapt to current player
    virtual Action get_action(const GameState& state, int player_id) = 0;
    virtual std::string get_name() const = 0;
    virtual void reset() = 0;
    
    // Performance statistics (optional, return 0 if not supported)
    virtual size_t get_nodes_explored() const { return 0; }
    virtual double get_thinking_time() const { return 0.0; }
    virtual void reset_stats() {}
};

/**
 * Wrapper for RandomAgent that adapts to any player ID.
 */
class RandomAgentWrapper : public EvaluationAgent {
private:
    std::unique_ptr<RandomAgent> agent_;
    std::string name_;
    
public:
    RandomAgentWrapper(int seed = -1, const std::string& name = "RandomAgent")
        : agent_(std::make_unique<RandomAgent>(0, seed)), name_(name) {}
    
    Action get_action(const GameState& state, int player_id) override {
        // Create a temporary agent for the correct player if needed
        if (agent_->player_id() != player_id) {
            agent_ = std::make_unique<RandomAgent>(player_id, agent_->seed());
        }
        return agent_->get_action(state);
    }
    
    std::string get_name() const override { return name_; }
    void reset() override { agent_->reset(); }
    void reset_stats() override { agent_->reset(); }
};

/**
 * Wrapper for MinimaxAgent that adapts to any player ID.
 */
class MinimaxAgentWrapper : public EvaluationAgent {
private:
    std::unique_ptr<MinimaxAgent> agent_;
    std::string name_;
    int depth_;
    bool enable_alpha_beta_;
    bool enable_memoization_;
    int seed_;
    
public:
    MinimaxAgentWrapper(int depth = 4, bool enable_alpha_beta = true, 
                       bool enable_memoization = true, int seed = -1, 
                       const std::string& name = "MinimaxAgent")
        : agent_(std::make_unique<MinimaxAgent>(0, depth, enable_alpha_beta, enable_memoization, seed)), 
          name_(name), depth_(depth), enable_alpha_beta_(enable_alpha_beta),
          enable_memoization_(enable_memoization), seed_(seed) {}
    
    Action get_action(const GameState& state, int player_id) override {
        // Create a fresh agent for the correct player if needed
        if (agent_->player_id() != player_id) {
            agent_ = std::make_unique<MinimaxAgent>(player_id, depth_, enable_alpha_beta_, enable_memoization_, seed_);
        }
        
        try {
            return agent_->get_action(state);
        } catch (const std::exception& e) {
            // Fallback to random action if minimax fails
            auto legal_actions = state.get_legal_actions(player_id);
            if (!legal_actions.empty()) {
                return legal_actions[0]; // Take first legal action as fallback
            }
            throw; // Re-throw if no legal actions
        }
    }
    
    std::string get_name() const override { return name_; }
    void reset() override { 
        // Reset with fresh agent to clear any state
        agent_ = std::make_unique<MinimaxAgent>(0, depth_, enable_alpha_beta_, enable_memoization_, seed_);
    }
    void reset_stats() override { agent_->reset_stats(); }
    
    size_t get_nodes_explored() const override {
        return agent_->nodes_explored();
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
    bool enable_memoization = true, int seed = -1, const std::string& name = "MinimaxAgent"
);

} // namespace azul 