"""
Command-line interface for the Agent Evaluation Framework.

This module provides easy-to-use command line tools for running agent evaluations,
tournaments, and analyzing results.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from agents.openspiel_agents import OpenSpielMCTSAgent as MCTSAgent
from evaluation.agent_evaluator import AgentEvaluator
from evaluation.baseline_agents import create_baseline_agent
from evaluation.evaluation_config import EvaluationConfig
from evaluation.tournament import Tournament, run_baseline_comparison_tournament
from evaluation.utils import (
    create_evaluation_summary,
    ensure_evaluation_dir,
    format_evaluation_results,
    save_evaluation_results,
)


def create_test_agent(agent_type: str, **kwargs):
    """Create a test agent of the specified type."""
    if agent_type == "heuristic":
        from agents.heuristic_agent import HeuristicAgent

        return HeuristicAgent(**kwargs)
    elif agent_type == "mcts":
        # OpenSpiel MCTS doesn't require neural networks
        # Extract common MCTS parameters from kwargs and map to OpenSpiel parameters
        num_simulations = kwargs.get("num_simulations", 400)
        uct_c = kwargs.get(
            "uct_c", kwargs.get("c_puct", 1.4)
        )  # Map old c_puct to uct_c
        solve = kwargs.get("solve", False)

        # Remove parameters that don't apply to OpenSpiel MCTS
        openspiel_kwargs = {
            "num_simulations": num_simulations,
            "uct_c": uct_c,
            "solve": solve,
        }

        return MCTSAgent(**openspiel_kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def cmd_evaluate(args):
    """Run a single agent evaluation."""
    # Create configuration
    config = EvaluationConfig(
        num_games=args.num_games,
        timeout_per_move=args.timeout,
        num_workers=args.workers,
        verbose=args.verbose,
        use_fixed_seeds=not args.no_fixed_seeds,
        random_seed=args.seed,
        swap_player_positions=not args.no_swap_positions,
        save_detailed_logs=args.detailed_logs,
        save_game_replays=args.save_replays,
    )

    # Create agents
    try:
        test_agent = create_test_agent(
            args.test_agent, **parse_agent_args(args.test_agent_args)
        )
        baseline_agent = create_baseline_agent(
            args.baseline_agent, **parse_agent_args(args.baseline_agent_args)
        )
    except Exception as e:
        print(f"Error creating agents: {e}")
        return 1

    # Run evaluation
    evaluator = AgentEvaluator(config)

    try:
        result = evaluator.evaluate_agent(
            test_agent=test_agent,
            baseline_agent=baseline_agent,
            test_agent_name=args.test_agent_name,
            baseline_agent_name=args.baseline_agent_name,
        )

        # Print results
        print(format_evaluation_results(result, detailed=args.detailed))

        # Save results if requested
        if args.output:
            ensure_evaluation_dir(str(Path(args.output).parent))
            save_evaluation_results(result, args.output)
            print(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


def cmd_tournament(args):
    """Run a tournament between multiple agents."""
    # Create configuration
    config = EvaluationConfig(
        num_games=args.num_games,
        timeout_per_move=args.timeout,
        num_workers=args.workers,
        verbose=args.verbose,
        use_fixed_seeds=not args.no_fixed_seeds,
        random_seed=args.seed,
    )

    # Parse agent specifications
    agents = {}

    if args.agents:
        for agent_spec in args.agents:
            try:
                parts = agent_spec.split(":")
                if len(parts) < 2:
                    print(f"Invalid agent specification: {agent_spec}")
                    print("Format: name:type[:arg1=val1,arg2=val2,...]")
                    return 1

                name = parts[0]
                agent_type = parts[1]
                agent_args = parse_agent_args(parts[2] if len(parts) > 2 else "")

                if agent_type in [
                    "random",
                    "simple_heuristic",
                    "heuristic",
                    "checkpoint",
                ]:
                    agent = create_baseline_agent(agent_type, **agent_args)
                else:
                    agent = create_test_agent(agent_type, **agent_args)

                agents[name] = agent

            except Exception as e:
                print(f"Error creating agent {agent_spec}: {e}")
                return 1

    # Add baseline agents if requested
    if args.include_baselines:
        baseline_types = (
            args.baseline_types.split(",")
            if args.baseline_types
            else ["random", "heuristic"]
        )
        for baseline_type in baseline_types:
            try:
                baseline_agent = create_baseline_agent(baseline_type.strip())
                agents[f"{baseline_type.strip()}_baseline"] = baseline_agent
            except Exception as e:
                print(f"Error creating baseline {baseline_type}: {e}")
                return 1

    if len(agents) < 2:
        print("Need at least 2 agents for a tournament")
        return 1

    # Run tournament
    tournament = Tournament(config)

    for name, agent in agents.items():
        tournament.add_agent(agent, name)

    try:
        tournament.run_tournament(verbose=args.verbose)

        # Save results if requested
        if args.output:
            ensure_evaluation_dir(str(Path(args.output).parent))
            tournament.save_results(args.output)

        return 0

    except Exception as e:
        print(f"Error during tournament: {e}")
        return 1


def cmd_summary(args):
    """Generate a summary of evaluation results."""
    try:
        summary = create_evaluation_summary(args.results_dir, args.output)
        print(summary)
        return 0
    except Exception as e:
        print(f"Error generating summary: {e}")
        return 1


def cmd_quick(args):
    """Run a quick evaluation for testing."""
    # Create simple config
    config = EvaluationConfig(
        num_games=args.num_games or 10,
        timeout_per_move=2.0,
        verbose=True,
        num_workers=1,
    )

    # Create agents
    try:
        test_agent = create_test_agent(args.test_agent)
        baseline_agent = create_baseline_agent(args.baseline_agent or "random")
    except Exception as e:
        print(f"Error creating agents: {e}")
        return 1

    # Run quick evaluation
    evaluator = AgentEvaluator(config)

    try:
        result = evaluator.quick_evaluation(
            test_agent, baseline_agent, num_games=args.num_games or 10
        )
        print(result.summary())
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


def parse_agent_args(args_str: str) -> dict:
    """Parse agent arguments from string format 'arg1=val1,arg2=val2'."""
    if not args_str:
        return {}

    args_dict: dict = {}
    for arg_pair in args_str.split(","):
        if "=" in arg_pair:
            key, value = arg_pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to convert to appropriate type
            if value.lower() == "true":
                args_dict[key] = True
            elif value.lower() == "false":
                args_dict[key] = False
            elif value.isdigit():
                args_dict[key] = int(value)
            elif value.replace(".", "").isdigit():
                args_dict[key] = float(value)
            else:
                args_dict[key] = value

    return args_dict


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agent Evaluation Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick evaluation
  python -m evaluation.cli quick --test-agent heuristic

  # Full evaluation with custom settings
  python -m evaluation.cli evaluate \\
    --test-agent mcts --baseline-agent random \\
    --num-games 100 --output results.json

  # Tournament between multiple agents
  python -m evaluation.cli tournament \\
    --agents "MCTS:mcts" "Heuristic:heuristic" \\
    --include-baselines --output tournament_results

  # Generate summary of results
  python -m evaluation.cli summary evaluation_results/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a single agent against a baseline"
    )
    eval_parser.add_argument(
        "--test-agent",
        required=True,
        choices=["heuristic", "mcts", "checkpoint"],
        help="Type of test agent",
    )
    eval_parser.add_argument(
        "--baseline-agent",
        required=True,
        choices=["random", "simple_heuristic", "heuristic", "checkpoint"],
        help="Type of baseline agent",
    )
    eval_parser.add_argument("--test-agent-name", help="Name for test agent")
    eval_parser.add_argument("--baseline-agent-name", help="Name for baseline agent")
    eval_parser.add_argument(
        "--test-agent-args",
        default="",
        help="Arguments for test agent (key=value,key=value)",
    )
    eval_parser.add_argument(
        "--baseline-agent-args", default="", help="Arguments for baseline agent"
    )
    eval_parser.add_argument(
        "--num-games", type=int, default=100, help="Number of games to play"
    )
    eval_parser.add_argument(
        "--timeout", type=float, default=5.0, help="Timeout per move in seconds"
    )
    eval_parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers"
    )
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    eval_parser.add_argument(
        "--no-fixed-seeds", action="store_true", help="Disable fixed seeds"
    )
    eval_parser.add_argument(
        "--no-swap-positions", action="store_true", help="Disable position swapping"
    )
    eval_parser.add_argument(
        "--detailed-logs", action="store_true", help="Save detailed logs"
    )
    eval_parser.add_argument(
        "--save-replays", action="store_true", help="Save game replays"
    )
    eval_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed results"
    )
    eval_parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )
    eval_parser.add_argument("--output", help="Output file for results")

    # Tournament command
    tournament_parser = subparsers.add_parser(
        "tournament", help="Run a tournament between multiple agents"
    )
    tournament_parser.add_argument(
        "--agents", nargs="*", help="Agent specifications (name:type[:args])"
    )
    tournament_parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include standard baseline agents",
    )
    tournament_parser.add_argument(
        "--baseline-types",
        default="random,heuristic",
        help="Comma-separated baseline types to include",
    )
    tournament_parser.add_argument(
        "--num-games", type=int, default=50, help="Games per matchup"
    )
    tournament_parser.add_argument(
        "--timeout", type=float, default=5.0, help="Timeout per move"
    )
    tournament_parser.add_argument(
        "--workers", type=int, default=1, help="Parallel workers"
    )
    tournament_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    tournament_parser.add_argument(
        "--no-fixed-seeds", action="store_true", help="Disable fixed seeds"
    )
    tournament_parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )
    tournament_parser.add_argument("--output", help="Output file base name for results")

    # Summary command
    summary_parser = subparsers.add_parser(
        "summary", help="Generate summary of evaluation results"
    )
    summary_parser.add_argument(
        "results_dir", help="Directory containing evaluation results"
    )
    summary_parser.add_argument("--output", help="Output file for summary")

    # Quick command
    quick_parser = subparsers.add_parser("quick", help="Quick evaluation for testing")
    quick_parser.add_argument(
        "--test-agent",
        required=True,
        choices=["heuristic", "mcts"],
        help="Test agent type",
    )
    quick_parser.add_argument(
        "--baseline-agent",
        choices=["random", "simple_heuristic", "heuristic"],
        help="Baseline agent type (default: random)",
    )
    quick_parser.add_argument(
        "--num-games", type=int, help="Number of games (default: 10)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command
    if args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "tournament":
        return cmd_tournament(args)
    elif args.command == "summary":
        return cmd_summary(args)
    elif args.command == "quick":
        return cmd_quick(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
