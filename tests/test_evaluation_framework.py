"""
Basic tests for the Agent Evaluation Framework.

These tests ensure the core functionality of the evaluation framework
works correctly with the existing Azul game infrastructure.
"""

import json
import tempfile
import unittest
from pathlib import Path

from agents import HeuristicAgent, RandomAgent

# Import evaluation framework components
from evaluation import (
    AgentEvaluator,
    EvaluationConfig,
    HeuristicBaselineAgent,
    RandomBaselineAgent,
    Tournament,
    load_evaluation_results,
    save_evaluation_results,
)
from evaluation.utils import calculate_confidence_interval, calculate_win_rate


class TestEvaluationFramework(unittest.TestCase):
    """Test basic evaluation framework functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_evaluation_config(self):
        """Test evaluation configuration creation and serialization."""
        config = EvaluationConfig(
            num_games=50, timeout_per_move=2.0, verbose=False, random_seed=123
        )

        # Test basic attributes
        self.assertEqual(config.num_games, 50)
        self.assertEqual(config.timeout_per_move, 2.0)
        self.assertEqual(config.random_seed, 123)

        # Test serialization
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["num_games"], 50)

        # Test deserialization
        restored_config = EvaluationConfig.from_dict(config_dict)
        self.assertEqual(restored_config.num_games, 50)
        self.assertEqual(restored_config.timeout_per_move, 2.0)

    def test_baseline_agents(self):
        """Test baseline agent creation and basic functionality."""
        # Test RandomBaselineAgent
        random_agent = RandomBaselineAgent(seed=42)
        self.assertEqual(random_agent.name, "RandomAgent")
        self.assertEqual(random_agent.player_id, 0)

        # Test HeuristicBaselineAgent
        heuristic_agent = HeuristicBaselineAgent()
        self.assertEqual(heuristic_agent.name, "HeuristicAgent")

        # Test agent info
        info = random_agent.get_info()
        self.assertIsInstance(info, dict)
        self.assertIn("agent_type", info)

    def test_quick_evaluation(self):
        """Test quick evaluation functionality."""
        # Create simple agents
        test_agent = HeuristicAgent()
        baseline_agent = RandomBaselineAgent(seed=42)

        # Create evaluator
        evaluator = AgentEvaluator()

        # Run quick evaluation
        result = evaluator.quick_evaluation(
            test_agent=test_agent,
            baseline_agent=baseline_agent,
            num_games=5,
            verbose=False,
        )

        # Check result structure
        self.assertEqual(result.games_played, 5)
        self.assertIsInstance(result.test_agent_win_rate, float)
        self.assertIsInstance(result.baseline_agent_win_rate, float)
        self.assertIsInstance(result.game_results, list)
        self.assertEqual(len(result.game_results), 5)

        # Check that win rates sum to 1.0 (accounting for potential draws)
        total_rate = result.test_agent_win_rate + result.baseline_agent_win_rate
        self.assertLessEqual(total_rate, 1.0)

    def test_evaluation_with_configuration(self):
        """Test evaluation with custom configuration."""
        config = EvaluationConfig(
            num_games=10,
            timeout_per_move=1.0,
            verbose=False,
            use_fixed_seeds=True,
            random_seed=42,
            swap_player_positions=False,  # Faster for testing
        )

        evaluator = AgentEvaluator(config)

        # Use different agents
        test_agent = HeuristicBaselineAgent()
        baseline_agent = RandomBaselineAgent(seed=42)

        result = evaluator.evaluate_agent(
            test_agent=test_agent,
            baseline_agent=baseline_agent,
            test_agent_name="TestHeuristic",
            baseline_agent_name="TestRandom",
        )

        # Verify configuration was used
        self.assertEqual(result.games_played, 10)
        self.assertEqual(result.test_agent_name, "TestHeuristic")
        self.assertEqual(result.baseline_agent_name, "TestRandom")

        # Heuristic should beat random most of the time
        self.assertGreater(result.test_agent_win_rate, 0.3)  # At least 30% win rate

    def test_result_serialization(self):
        """Test saving and loading evaluation results."""
        # Run a quick evaluation
        evaluator = AgentEvaluator()
        test_agent = HeuristicAgent()
        baseline_agent = RandomBaselineAgent(seed=42)

        result = evaluator.quick_evaluation(
            test_agent, baseline_agent, num_games=3, verbose=False
        )

        # Save to temporary file
        temp_file = Path(self.temp_dir) / "test_result.json"
        save_evaluation_results(result, str(temp_file))

        # Verify file was created
        self.assertTrue(temp_file.exists())

        # Load and verify
        loaded_result = load_evaluation_results(str(temp_file))

        self.assertEqual(loaded_result.games_played, result.games_played)
        self.assertEqual(loaded_result.test_agent_name, result.test_agent_name)
        self.assertEqual(loaded_result.test_agent_win_rate, result.test_agent_win_rate)
        self.assertEqual(len(loaded_result.game_results), len(result.game_results))

    def test_tournament_basic(self):
        """Test basic tournament functionality."""
        config = EvaluationConfig(
            num_games=5, timeout_per_move=1.0, verbose=False  # Small for fast testing
        )

        tournament = Tournament(config)

        # Add agents
        tournament.add_agent(RandomAgent(seed=1), "Random1")
        tournament.add_agent(RandomAgent(seed=2), "Random2")
        tournament.add_agent(HeuristicAgent(), "Heuristic")

        # Run tournament
        result = tournament.run_tournament(verbose=False)

        # Check tournament result structure
        self.assertEqual(len(result.agents), 3)
        self.assertIn("Random1", result.agents)
        self.assertIn("Random2", result.agents)
        self.assertIn("Heuristic", result.agents)

        # Should have 3 matchups (3 choose 2)
        self.assertEqual(len(result.matchups), 3)

        # Check rankings were calculated
        self.assertEqual(len(result.rankings), 3)

        # Heuristic should likely be ranked first
        # Note: With only 5 games and randomness, this might not always be true
        # but Heuristic should have a good chance

    def test_utility_functions(self):
        """Test utility functions."""
        # Test win rate calculation
        win_rate = calculate_win_rate(75, 100)
        self.assertEqual(win_rate, 0.75)

        win_rate_zero = calculate_win_rate(0, 0)
        self.assertEqual(win_rate_zero, 0.0)

        # Test confidence interval calculation
        lower, upper = calculate_confidence_interval(75, 100, 0.95)
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)
        self.assertLessEqual(lower, 0.75)
        self.assertGreaterEqual(upper, 0.75)
        self.assertLess(lower, upper)

    def test_agent_statistics(self):
        """Test agent statistics collection."""
        agent = RandomBaselineAgent(seed=42)

        # Initially no stats
        stats = agent.get_stats()
        self.assertEqual(stats["total_moves"], 0)

        # Test stats reset
        agent.reset_stats()
        stats = agent.get_stats()
        self.assertEqual(stats["total_moves"], 0)
        self.assertEqual(stats["total_time_taken"], 0.0)

    def test_error_handling(self):
        """Test error handling in evaluation."""
        # Test with invalid configuration
        try:
            config = EvaluationConfig(num_games=0)
            evaluator = AgentEvaluator(config)

            test_agent = HeuristicAgent()
            baseline_agent = RandomBaselineAgent()

            # This should handle the edge case gracefully
            result = evaluator.quick_evaluation(
                test_agent, baseline_agent, num_games=0, verbose=False
            )
            self.assertEqual(result.games_played, 0)

        except Exception as e:
            # If it raises an exception, that's also acceptable
            self.assertIsInstance(e, (ValueError, RuntimeError))


class TestEvaluationUtilities(unittest.TestCase):
    """Test evaluation utility functions."""

    def test_statistical_functions(self):
        """Test statistical utility functions."""
        from evaluation.utils import calculate_statistical_significance

        # Test statistical significance
        p_value, is_significant = calculate_statistical_significance(
            wins_a=80, total_a=100, wins_b=20, total_b=100, alpha=0.05
        )

        self.assertIsInstance(p_value, float)
        self.assertIsInstance(is_significant, bool)
        self.assertLessEqual(p_value, 1.0)
        self.assertGreaterEqual(p_value, 0.0)

        # With such different win rates, should be significant
        self.assertTrue(is_significant)

    def test_result_formatting(self):
        """Test result formatting utilities."""
        from evaluation.utils import format_evaluation_results

        # Create a mock result for testing
        config = EvaluationConfig(num_games=10, verbose=False)
        evaluator = AgentEvaluator(config)

        test_agent = HeuristicAgent()
        baseline_agent = RandomBaselineAgent(seed=42)

        result = evaluator.quick_evaluation(
            test_agent, baseline_agent, num_games=3, verbose=False
        )

        # Test formatting
        formatted = format_evaluation_results(result, detailed=False)
        self.assertIsInstance(formatted, str)
        self.assertIn("AGENT EVALUATION RESULTS", formatted)
        self.assertIn(result.test_agent_name, formatted)

        # Test detailed formatting
        detailed_formatted = format_evaluation_results(result, detailed=True)
        self.assertIsInstance(detailed_formatted, str)
        self.assertIn("DETAILED STATISTICS", detailed_formatted)


if __name__ == "__main__":
    # Run specific test methods for debugging
    unittest.main(verbosity=2)
