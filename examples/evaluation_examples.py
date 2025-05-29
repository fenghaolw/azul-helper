"""
Example scripts demonstrating the Agent Evaluation Framework.

This module provides practical examples of how to use the evaluation framework
for different scenarios including basic evaluation, tournaments, and analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents import CheckpointAgent, HeuristicAgent, RandomAgent
from evaluation import (
    AgentEvaluator,
    EvaluationConfig,
    Tournament,
    format_evaluation_results,
    save_evaluation_results,
)
from evaluation.baseline_agents import create_baseline_agent


def example_basic_evaluation():
    """Basic example of evaluating an agent against a baseline."""
    print("=" * 60)
    print("BASIC EVALUATION EXAMPLE")
    print("=" * 60)

    # Create a simple evaluation configuration
    config = EvaluationConfig(
        num_games=20,  # Small number for quick demo
        timeout_per_move=2.0,
        verbose=True,
        use_fixed_seeds=True,
        random_seed=42,
    )

    # Create agents
    test_agent = HeuristicAgent(player_id=0)
    baseline_agent = RandomAgent(player_id=1, seed=42)

    # Create evaluator and run evaluation
    evaluator = AgentEvaluator(config)

    result = evaluator.evaluate_agent(
        test_agent=test_agent,
        baseline_agent=baseline_agent,
        test_agent_name="HeuristicAgent",
        baseline_agent_name="RandomAgent",
    )

    # Print results
    print(format_evaluation_results(result, detailed=True))

    # Save results
    save_evaluation_results(result, "evaluation_results/heuristic_vs_random.json")
    print("\nResults saved to evaluation_results/heuristic_vs_random.json")


def example_agent_comparison():
    """Example comparing different baseline agents."""
    print("\n" + "=" * 60)
    print("AGENT COMPARISON EXAMPLE")
    print("=" * 60)

    config = EvaluationConfig(
        num_games=15,
        timeout_per_move=1.5,
        verbose=True,
    )

    evaluator = AgentEvaluator(config)

    # Create baseline agents to compare - using baseline factory
    agents_to_test = [
        ("HeuristicBaseline", create_baseline_agent("heuristic")),
        ("RandomBaseline", create_baseline_agent("random")),
    ]

    baseline_agent = RandomAgent(seed=42)

    results = []

    for agent_name, agent in agents_to_test:
        print(f"\nEvaluating {agent_name}...")

        result = evaluator.evaluate_agent(
            test_agent=agent,
            baseline_agent=baseline_agent,
            test_agent_name=agent_name,
            baseline_agent_name="Random",
        )

        results.append(result)
        print(f"{agent_name}: {result.test_agent_win_rate:.1%} win rate")

    # Compare results
    print("\n" + "=" * 40)
    print("COMPARISON SUMMARY")
    print("=" * 40)

    for result in results:
        print(
            f"{result.test_agent_name}: {result.test_agent_win_rate:.1%} "
            f"({result.test_agent_wins}/{result.games_played}) "
            f"avg score diff: {result.average_score_difference:+.1f}"
        )


def example_tournament():
    """Example of running a tournament between multiple agents."""
    print("\n" + "=" * 60)
    print("TOURNAMENT EXAMPLE")
    print("=" * 60)

    # Create tournament configuration
    config = EvaluationConfig(
        num_games=10,  # Small for demo
        timeout_per_move=1.0,
        verbose=True,
    )

    # Create tournament
    tournament = Tournament(config)

    # Add agents
    tournament.add_agent(RandomAgent(seed=1), "Random")
    tournament.add_agent(create_baseline_agent("heuristic"), "HeuristicBaseline")
    tournament.add_agent(HeuristicAgent(), "HeuristicDirect")

    # Run tournament
    tournament.run_tournament()

    # Save tournament results
    tournament.save_results("evaluation_results/tournament_demo")


def example_quick_evaluation():
    """Example of quick evaluation for testing."""
    print("\n" + "=" * 60)
    print("QUICK EVALUATION EXAMPLE")
    print("=" * 60)

    evaluator = AgentEvaluator()

    # Create agents
    test_agent = HeuristicAgent()
    baseline_agent = RandomAgent()

    # Run quick evaluation (only 5 games)
    result = evaluator.quick_evaluation(
        test_agent=test_agent, baseline_agent=baseline_agent, num_games=5, verbose=True
    )

    print(result.summary())


def example_with_checkpoints():
    """Example of evaluating against previous model checkpoints."""
    print("\n" + "=" * 60)
    print("CHECKPOINT EVALUATION EXAMPLE")
    print("=" * 60)

    # This example assumes you have some model checkpoints available
    checkpoint_paths = [
        "models/checkpoint_100.pth",
        "models/checkpoint_200.pth",
        "models/checkpoint_300.pth",
    ]

    # Check if any checkpoints exist
    existing_checkpoints = []
    for path in checkpoint_paths:
        if Path(path).exists():
            existing_checkpoints.append(path)

    if not existing_checkpoints:
        print("No model checkpoints found. Skipping checkpoint evaluation example.")
        print(
            "To run this example, you need trained model checkpoints in the models/ directory."
        )
        return

    try:
        config = EvaluationConfig(num_games=10, verbose=True)
        evaluator = AgentEvaluator(config)

        # Load latest checkpoint
        latest_checkpoint = existing_checkpoints[-1]
        checkpoint_agent = CheckpointAgent(latest_checkpoint)

        # Compare against heuristic baseline
        baseline_agent = create_baseline_agent("heuristic")

        result = evaluator.evaluate_agent(
            test_agent=checkpoint_agent,
            baseline_agent=baseline_agent,
            test_agent_name=f"Checkpoint_{len(existing_checkpoints)}",
            baseline_agent_name="Heuristic",
        )

        print(format_evaluation_results(result))

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("This might happen if the checkpoint format is incompatible.")


def example_statistical_analysis():
    """Example of statistical analysis of evaluation results."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS EXAMPLE")
    print("=" * 60)

    config = EvaluationConfig(
        num_games=50,  # Larger sample for better statistics
        confidence_interval=0.95,
        verbose=False,  # Less verbose for cleaner output
    )

    evaluator = AgentEvaluator(config)

    # Compare two different approaches
    test_agent = create_baseline_agent("heuristic")
    baseline_agent = RandomAgent()

    result = evaluator.evaluate_agent(
        test_agent=test_agent,
        baseline_agent=baseline_agent,
        test_agent_name="HeuristicBaseline",
        baseline_agent_name="Random",
    )

    print(f"Evaluation Results:")
    print(f"Games Played: {result.games_played}")
    print(f"Win Rate: {result.test_agent_win_rate:.1%}")

    if result.confidence_interval:
        lower, upper = result.confidence_interval
        print(f"95% Confidence Interval: [{lower:.1%}, {upper:.1%}]")

    if result.p_value is not None:
        print(f"P-value: {result.p_value:.4f}")
        significance = "Yes" if result.is_statistically_significant else "No"
        print(f"Statistically Significant: {significance}")

    print(f"Average Score Difference: {result.average_score_difference:+.1f}")


def example_configuration_options():
    """Example showing different configuration options."""
    print("\n" + "=" * 60)
    print("CONFIGURATION OPTIONS EXAMPLE")
    print("=" * 60)

    # Example 1: Fast evaluation for development
    print("1. Fast Development Configuration:")
    fast_config = EvaluationConfig(
        num_games=5,
        timeout_per_move=0.5,
        num_workers=1,
        verbose=True,
        swap_player_positions=False,  # Faster
        use_fixed_seeds=True,
    )
    print(
        f"   Games: {fast_config.num_games}, Timeout: {fast_config.timeout_per_move}s"
    )

    # Example 2: Thorough evaluation for final results
    print("\n2. Thorough Evaluation Configuration:")
    thorough_config = EvaluationConfig(
        num_games=200,
        timeout_per_move=5.0,
        num_workers=4,  # Parallel processing
        verbose=True,
        swap_player_positions=True,  # More robust
        save_detailed_logs=True,
        save_game_replays=True,
        confidence_interval=0.99,  # Higher confidence
    )
    print(
        f"   Games: {thorough_config.num_games}, Workers: {thorough_config.num_workers}"
    )

    # Example 3: Tournament configuration
    print("\n3. Tournament Configuration:")
    tournament_config = EvaluationConfig(
        num_games=30,  # Per matchup
        timeout_per_move=3.0,
        num_workers=2,
        verbose=True,
        deterministic_evaluation=True,  # More consistent for comparisons
    )
    print(f"   Games per matchup: {tournament_config.num_games}")

    print("\nConfiguration objects created successfully!")


def main():
    """Run all examples."""
    print("Agent Evaluation Framework Examples")
    print("=" * 80)

    # Create output directory
    Path("evaluation_results").mkdir(exist_ok=True)

    try:
        # Run examples in order
        example_basic_evaluation()
        example_agent_comparison()
        example_quick_evaluation()
        example_tournament()
        example_statistical_analysis()
        example_configuration_options()

        # Optional: checkpoint example (only if checkpoints exist)
        example_with_checkpoints()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("Check the evaluation_results/ directory for saved results.")

    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
