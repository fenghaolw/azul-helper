#include "evaluation_config.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

namespace azul {

void EvaluationResult::finalize_results() {
  if (games_played > 0) {
    test_agent_win_rate = static_cast<double>(test_agent_wins) / games_played;
    baseline_agent_win_rate =
        static_cast<double>(baseline_agent_wins) / games_played;

    // Calculate average scores and score difference
    double test_total_score = 0.0;
    double baseline_total_score = 0.0;
    double total_score_diff = 0.0;
    double total_duration = 0.0;

    for (const auto& game_result : game_results) {
      if (game_result.final_scores.size() >= 2) {
        test_total_score += game_result.final_scores[0];
        baseline_total_score += game_result.final_scores[1];
        total_score_diff +=
            (game_result.final_scores[0] - game_result.final_scores[1]);
      }
      total_duration += game_result.game_duration;
    }

    test_agent_avg_score = test_total_score / games_played;
    baseline_agent_avg_score = baseline_total_score / games_played;
    average_score_difference = total_score_diff / games_played;
    average_game_duration = total_duration / games_played;
  }
}

std::string EvaluationResult::summary() const {
  std::ostringstream oss;

  oss << "Evaluation Results: " << test_agent_name << " vs "
      << baseline_agent_name << "\n";
  oss << "Timestamp: " << timestamp << "\n";
  oss << "Games Played: " << games_played << "\n\n";

  oss << "Win Rates:\n";
  oss << "  " << test_agent_name << ": " << std::fixed << std::setprecision(1)
      << (test_agent_win_rate * 100) << "% (" << test_agent_wins << " wins)\n";
  oss << "  " << baseline_agent_name << ": " << std::fixed
      << std::setprecision(1) << (baseline_agent_win_rate * 100) << "% ("
      << baseline_agent_wins << " wins)\n";
  oss << "  Draws: " << draws << "\n\n";

  oss << "Performance Metrics:\n";
  oss << "  Average Score Difference: " << std::showpos << std::fixed
      << std::setprecision(1) << average_score_difference << "\n"
      << std::noshowpos;
  oss << "  Average Game Duration: " << std::fixed << std::setprecision(1)
      << average_game_duration << "s\n\n";

  oss << "Statistical Analysis:\n";
  oss << "  P-value: " << std::fixed << std::setprecision(4) << p_value << "\n";
  oss << "  Statistically Significant: "
      << (is_statistically_significant ? "Yes" : "No") << "\n";
  oss << "  Confidence Interval: [" << std::fixed << std::setprecision(3)
      << confidence_interval.first << ", " << confidence_interval.second
      << "]\n";

  if (timeouts > 0 || errors > 0) {
    oss << "\nIssues:\n";
    oss << "  Timeouts: " << timeouts << "\n";
    oss << "  Errors: " << errors << "\n";
  }

  return oss.str();
}

void TournamentResult::calculate_rankings() {
  rankings.clear();

  // Calculate statistics for each agent
  std::map<std::string, std::tuple<int, int, double>>
      agent_stats_map;  // wins, games, score_diff

  for (const auto& result : matchup_results) {
    const auto& test_name = result.test_agent_name;
    const auto& baseline_name = result.baseline_agent_name;

    // Update test agent stats
    auto& test_stats = agent_stats_map[test_name];
    std::get<0>(test_stats) += result.test_agent_wins;
    std::get<1>(test_stats) += result.games_played;
    std::get<2>(test_stats) +=
        result.average_score_difference * result.games_played;

    // Update baseline agent stats
    auto& baseline_stats = agent_stats_map[baseline_name];
    std::get<0>(baseline_stats) += result.baseline_agent_wins;
    std::get<1>(baseline_stats) += result.games_played;
    std::get<2>(baseline_stats) -=
        result.average_score_difference * result.games_played;
  }

  // Convert to rankings format
  for (const auto& [agent_name, stats] : agent_stats_map) {
    int wins = std::get<0>(stats);
    int games = std::get<1>(stats);
    double total_score_diff = std::get<2>(stats);

    double win_rate = games > 0 ? static_cast<double>(wins) / games : 0.0;
    double avg_score_diff = games > 0 ? total_score_diff / games : 0.0;

    rankings.emplace_back(agent_name, win_rate, avg_score_diff);
  }

  // Sort by win rate (descending), then by average score difference
  // (descending)
  std::sort(rankings.begin(), rankings.end(), [](const auto& a, const auto& b) {
    if (std::abs(std::get<1>(a) - std::get<1>(b)) < 1e-6) {
      return std::get<2>(a) > std::get<2>(b);
    }
    return std::get<1>(a) > std::get<1>(b);
  });
}

std::string TournamentResult::summary() const {
  std::ostringstream oss;

  oss << "TOURNAMENT RESULTS\n";
  oss << std::string(50, '=') << "\n";
  oss << "Tournament Date: " << timestamp << "\n";
  oss << "Participants: " << agent_stats.size() << " agents\n";
  oss << "Total Matchups: " << matchup_results.size() << "\n\n";

  oss << "FINAL RANKINGS:\n";
  for (size_t i = 0; i < agent_stats.size(); ++i) {
    const auto& stats = agent_stats[i];
    oss << "  " << (i + 1) << ". " << stats.agent_name << ": " << std::fixed
        << std::setprecision(1) << (stats.win_rate * 100) << "% win rate, "
        << std::setprecision(1) << stats.avg_score << " avg score\n";
  }

  oss << "\nHEAD-TO-HEAD RESULTS:\n";
  for (const auto& result : matchup_results) {
    oss << result.test_agent_name << " vs " << result.baseline_agent_name
        << ": " << std::fixed << std::setprecision(1)
        << (result.test_agent_win_rate * 100) << "% win rate ("
        << result.test_agent_wins << "/" << result.games_played << " games)\n";
  }

  return oss.str();
}

}  // namespace azul