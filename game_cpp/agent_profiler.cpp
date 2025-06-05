#include "agent_profiler.h"
#include "minimax_agent.h"
#include "game_state.h"
#include "action.h"
#include <algorithm>
#include <iomanip>
#include <sstream>

#include "mcts_agent.h"
#include "open_spiel/spiel.h"

namespace azul {

// AgentProfiler implementation
void AgentProfiler::start_function(const std::string& function_name) {
    if (!profiling_enabled_) return;
    
    active_timers_[function_name].reset();
}

void AgentProfiler::end_function(const std::string& function_name) {
    if (!profiling_enabled_) return;
    
    auto timer_it = active_timers_.find(function_name);
    if (timer_it != active_timers_.end()) {
        double elapsed_ms = timer_it->second.elapsed_ms();
        function_stats_[function_name].update(elapsed_ms);
        active_timers_.erase(timer_it);
    }
}

void AgentProfiler::track_memory_allocation(const std::string& context, size_t size_bytes) {
    if (!profiling_enabled_) return;
    
    memory_stats_.track_allocation(size_bytes);
}

void AgentProfiler::track_memory_deallocation(const std::string& context, size_t size_bytes) {
    if (!profiling_enabled_) return;
    
    memory_stats_.track_deallocation(size_bytes);
}

void AgentProfiler::profile_minimax_search(const std::function<Action()>& search_func, 
                                          const std::string& context) {
    if (!profiling_enabled_) return;
    
    Timer timer;
    Action result = search_func();
    double elapsed_ms = timer.elapsed_ms();
    
    function_stats_[context].update(elapsed_ms);
}

void AgentProfiler::profile_mcts_search(const std::function<Action()>& search_func,
                                       const std::string& context) {
    if (!profiling_enabled_) return;
    
    Timer timer;
    Action result = search_func();
    double elapsed_ms = timer.elapsed_ms();
    
    function_stats_[context].update(elapsed_ms);
}

void AgentProfiler::print_profile_report(std::ostream& os) const {
    if (function_stats_.empty()) {
        os << "No profiling data available.\n";
        return;
    }
    
    double total_time = get_total_time_ms();
    
    os << "\n=== AGENT PERFORMANCE PROFILE REPORT ===\n";
    os << "Total execution time: " << std::fixed << std::setprecision(2) << total_time << " ms\n";
    os << "Memory usage: Peak=" << memory_stats_.peak_memory_kb << " KB, "
       << "Current=" << memory_stats_.current_memory_kb << " KB\n";
    os << "Memory operations: " << memory_stats_.allocations << " allocations, "
       << memory_stats_.deallocations << " deallocations\n\n";
    
    // Create sorted list of functions by total time
    std::vector<std::pair<std::string, FunctionStats>> sorted_stats;
    for (const auto& [name, stats] : function_stats_) {
        sorted_stats.emplace_back(name, stats);
    }
    
    std::sort(sorted_stats.begin(), sorted_stats.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_time_ms > b.second.total_time_ms;
              });
    
    // Print detailed stats
    os << std::left << std::setw(30) << "Function"
       << std::setw(10) << "Calls"
       << std::setw(12) << "Total(ms)"
       << std::setw(12) << "Avg(ms)"
       << std::setw(12) << "Min(ms)"
       << std::setw(12) << "Max(ms)"
       << std::setw(8) << "% Time" << "\n";
    os << std::string(96, '-') << "\n";
    
    for (const auto& [name, stats] : sorted_stats) {
        os << std::left << std::setw(30) << name
           << std::setw(10) << stats.call_count
           << std::setw(12) << std::fixed << std::setprecision(3) << stats.total_time_ms
           << std::setw(12) << std::fixed << std::setprecision(3) << stats.avg_time_ms
           << std::setw(12) << std::fixed << std::setprecision(3) << stats.min_time_ms
           << std::setw(12) << std::fixed << std::setprecision(3) << stats.max_time_ms
           << std::setw(8) << std::fixed << std::setprecision(1) << stats.get_percentage(total_time) << "%\n";
    }
}

void AgentProfiler::save_profile_report(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        print_profile_report(file);
        file.close();
    }
}

void AgentProfiler::print_hotspots(std::ostream& os, size_t top_n) const {
    if (function_stats_.empty()) {
        os << "No profiling data available.\n";
        return;
    }
    
    // Create sorted list of functions by total time
    std::vector<std::pair<std::string, FunctionStats>> sorted_stats;
    for (const auto& [name, stats] : function_stats_) {
        sorted_stats.emplace_back(name, stats);
    }
    
    std::sort(sorted_stats.begin(), sorted_stats.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_time_ms > b.second.total_time_ms;
              });
    
    double total_time = get_total_time_ms();
    
    os << "\n=== TOP " << top_n << " PERFORMANCE HOTSPOTS ===\n";
    
    size_t count = 0;
    for (const auto& [name, stats] : sorted_stats) {
        if (count >= top_n) break;
        
        os << (count + 1) << ". " << name << "\n";
        os << "   Time: " << std::fixed << std::setprecision(2) << stats.total_time_ms << " ms ("
           << std::fixed << std::setprecision(1) << stats.get_percentage(total_time) << "% of total)\n";
        os << "   Calls: " << stats.call_count << " (avg: " 
           << std::fixed << std::setprecision(3) << stats.avg_time_ms << " ms/call)\n";
        os << "   Range: " << std::fixed << std::setprecision(3) << stats.min_time_ms 
           << " - " << stats.max_time_ms << " ms\n\n";
        
        count++;
    }
}

// ProfiledMinimaxAgent implementation
ProfiledMinimaxAgent::ProfiledMinimaxAgent(std::unique_ptr<MinimaxAgent> agent)
    : agent_(std::move(agent)) {}

Action ProfiledMinimaxAgent::get_action(const GameState& state) {
    PROFILE_FUNCTION();
    
    auto& profiler = AgentProfiler::instance();
    
    Action result(0, TileColor::RED, 0); // Initialize with dummy values
    profiler.profile_minimax_search([&]() -> Action {
        // Profile the main components
        {
            PROFILE_SCOPE("get_legal_actions");
            auto legal_actions = state.get_legal_actions(agent_->player_id());
            profiler.track_memory_allocation("legal_actions", legal_actions.size() * sizeof(Action));
        }
        
        {
            PROFILE_SCOPE("minimax_search_core");
            result = agent_->get_action(state);
        }
        
        return result;
    }, "ProfiledMinimaxAgent::get_action");
    
    return result;
}

std::vector<double> ProfiledMinimaxAgent::get_action_probabilities(const GameState& state) {
    PROFILE_FUNCTION();
    return agent_->get_action_probabilities(state);
}

void ProfiledMinimaxAgent::reset() {
    PROFILE_FUNCTION();
    agent_->reset();
}

// ProfiledMCTSAgent implementation
ProfiledMCTSAgent::ProfiledMCTSAgent(std::unique_ptr<AzulMCTSAgent> agent)
    : agent_(std::move(agent)) {}

open_spiel::Action ProfiledMCTSAgent::get_action(const open_spiel::State& state) {
    PROFILE_FUNCTION();
    
    auto& profiler = AgentProfiler::instance();
    
    open_spiel::Action result;
    profiler.profile_mcts_search([&]() -> Action {
        {
            PROFILE_SCOPE("mcts_search_core");
            result = agent_->get_action(state);
        }
        
        return Action(0, TileColor::RED, 0); // Return dummy Action for the lambda
    }, "ProfiledMCTSAgent::get_action");
    
    return result;
}

std::vector<double> ProfiledMCTSAgent::get_action_probabilities(const open_spiel::State& state, double temperature) {
    PROFILE_FUNCTION();
    return agent_->get_action_probabilities(state, temperature);
}

void ProfiledMCTSAgent::reset() {
    PROFILE_FUNCTION();
    agent_->reset();
}

// Factory functions
std::unique_ptr<ProfiledMinimaxAgent> create_profiled_minimax_agent(int player_id, int depth, 
                                                                   bool enable_alpha_beta,
                                                                   int seed) {
    auto agent = create_minimax_agent(player_id, depth, enable_alpha_beta, seed);
    return std::make_unique<ProfiledMinimaxAgent>(std::move(agent));
}

std::unique_ptr<ProfiledMCTSAgent> create_profiled_mcts_agent(int player_id, int num_simulations, 
                                                             double uct_c, int seed) {
    auto agent = create_mcts_agent(player_id, num_simulations, uct_c, seed);
    return std::make_unique<ProfiledMCTSAgent>(std::move(agent));
}

} // namespace azul 