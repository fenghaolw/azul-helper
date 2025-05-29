#!/usr/bin/env python3
"""
Main entry point for the Agent Evaluation Framework.

This script provides easy access to common evaluation tasks including
quick tests, comprehensive evaluations, and tournaments.
"""

import argparse
import sys
from pathlib import Path

# Add evaluation module to path
sys.path.append(str(Path(__file__).parent))

from evaluation import (
    AgentEvaluator,
    EvaluationConfig,
    Tournament,
    format_evaluation_results,
    save_evaluation_results,
)
from evaluation.baseline_agents import HeuristicBaselineAgent, RandomBaselineAgent
from evaluation.utils import ensure_evaluation_dir


def run_heuristic_vs_random(num_games: int = 100, verbose: bool = True):
    """Run evaluation: Heuristic Agent vs Random Agent."""
    print(f"Running Heuristic vs Random evaluation ({num_games} games)...")

    config = EvaluationConfig(
        num_games=num_games,
        timeout_per_move=3.0,
        verbose=verbose,
        use_fixed_seeds=True,
        random_seed=42,
    )

    # Create agents
    from agents.heuristic_agent import HeuristicAgent

    test_agent = HeuristicAgent()
    baseline_agent = RandomBaselineAgent(seed=42)

    # Run evaluation
    evaluator = AgentEvaluator(config)
    result = evaluator.evaluate_agent(
        test_agent=test_agent,
        baseline_agent=baseline_agent,
        test_agent_name="HeuristicAgent",
        baseline_agent_name="RandomAgent",
    )

    # Print and save results
    print(format_evaluation_results(result))

    ensure_evaluation_dir()
    save_evaluation_results(result, "evaluation_results/heuristic_vs_random.json")
    print(f"\nResults saved to evaluation_results/heuristic_vs_random.json")

    return result


def run_baseline_comparison(num_games: int = 50, verbose: bool = True):
    """Compare all baseline agents against each other."""
    print(f"Running baseline comparison tournament ({num_games} games per matchup)...")

    config = EvaluationConfig(
        num_games=num_games,
        timeout_per_move=2.0,
        verbose=verbose,
    )

    # Create tournament
    tournament = Tournament(config)

    # Add baseline agents
    tournament.add_agent(RandomBaselineAgent(seed=1), "Random")
    tournament.add_agent(HeuristicBaselineAgent(), "Heuristic")

    # Run tournament
    result = tournament.run_tournament(verbose=verbose)

    # Save results
    ensure_evaluation_dir()
    tournament.save_results("evaluation_results/baseline_comparison")

    return result


def run_quick_test(agent_type: str = "heuristic"):
    """Run a quick test evaluation."""
    print(f"Running quick test with {agent_type} agent...")

    # Create test agent
    if agent_type.lower() == "heuristic":
        from agents.heuristic_agent import HeuristicAgent

        test_agent = HeuristicAgent()
    else:
        print(f"Unknown agent type: {agent_type}")
        return None

    # Quick evaluation
    evaluator = AgentEvaluator()
    result = evaluator.quick_evaluation(
        test_agent=test_agent,
        baseline_agent=RandomBaselineAgent(seed=42),
        num_games=10,
        verbose=True,
    )

    print(result.summary())
    return result


def run_comprehensive_evaluation(
    test_agent_type: str = "heuristic", num_games: int = 200
):
    """Run comprehensive evaluation with multiple baselines."""
    print(f"Running comprehensive evaluation of {test_agent_type} agent...")

    # Create test agent
    if test_agent_type.lower() == "heuristic":
        from agents.heuristic_agent import HeuristicAgent

        test_agent = HeuristicAgent()
        test_name = "HeuristicAgent"
    else:
        print(f"Unknown agent type: {test_agent_type}")
        return None

    config = EvaluationConfig(
        num_games=num_games,
        timeout_per_move=5.0,
        verbose=True,
        num_workers=2,  # Use some parallelism
        save_detailed_logs=True,
        confidence_interval=0.95,
    )

    evaluator = AgentEvaluator(config)

    # Test against multiple baselines
    baselines = [
        ("Random", RandomBaselineAgent(seed=42)),
        ("Heuristic", HeuristicBaselineAgent()),
    ]

    results = []

    for baseline_name, baseline_agent in baselines:
        print(f"\n--- Evaluating against {baseline_name} ---")

        result = evaluator.evaluate_agent(
            test_agent=test_agent,
            baseline_agent=baseline_agent,
            test_agent_name=test_name,
            baseline_agent_name=baseline_name,
        )

        results.append(result)

        print(f"Result: {result.test_agent_win_rate:.1%} win rate")
        if result.is_statistically_significant:
            print(f"Statistically significant (p={result.p_value:.4f})")

        # Save individual result
        ensure_evaluation_dir()
        filename = f"evaluation_results/{test_name}_vs_{baseline_name}.json"
        save_evaluation_results(result, filename)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 60)

    for result in results:
        print(
            f"{result.baseline_agent_name}: {result.test_agent_win_rate:.1%} "
            f"({result.test_agent_wins}/{result.games_played}) "
            f"avg score diff: {result.average_score_difference:+.1f}"
        )

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agent Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py quick
  python run_evaluation.py heuristic-vs-random --games 200
  python run_evaluation.py baseline-comparison
  python run_evaluation.py comprehensive --agent heuristic --games 500
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Evaluation commands")

    # Quick test command
    quick_parser = subparsers.add_parser(
        "quick", help="Quick evaluation test (10 games)"
    )
    quick_parser.add_argument(
        "--agent",
        default="heuristic",
        choices=["heuristic", "simple"],
        help="Agent type to test",
    )

    # Heuristic vs Random command
    hvr_parser = subparsers.add_parser(
        "heuristic-vs-random", help="Heuristic agent vs Random agent"
    )
    hvr_parser.add_argument(
        "--games", type=int, default=100, help="Number of games to play"
    )
    hvr_parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    # Baseline comparison command
    baseline_parser = subparsers.add_parser(
        "baseline-comparison", help="Tournament between baseline agents"
    )
    baseline_parser.add_argument(
        "--games", type=int, default=50, help="Games per matchup"
    )
    baseline_parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    # Comprehensive evaluation command
    comp_parser = subparsers.add_parser(
        "comprehensive", help="Comprehensive evaluation against multiple baselines"
    )
    comp_parser.add_argument(
        "--agent", default="heuristic", help="Agent type to evaluate"
    )
    comp_parser.add_argument(
        "--games", type=int, default=200, help="Number of games per baseline"
    )

    # Command line interface
    cli_parser = subparsers.add_parser("cli", help="Use the full CLI interface")
    cli_parser.add_argument("args", nargs="*", help="Arguments to pass to CLI")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "quick":
            run_quick_test(args.agent)

        elif args.command == "heuristic-vs-random":
            run_heuristic_vs_random(args.games, args.verbose)

        elif args.command == "baseline-comparison":
            run_baseline_comparison(args.games, args.verbose)

        elif args.command == "comprehensive":
            run_comprehensive_evaluation(args.agent, args.games)

        elif args.command == "cli":
            # Import and run the full CLI
            from evaluation.cli import main as cli_main

            sys.argv = ["evaluation.cli"] + args.args
            return cli_main()

        else:
            print(f"Unknown command: {args.command}")
            return 1

        print("\nEvaluation completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return 1

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
