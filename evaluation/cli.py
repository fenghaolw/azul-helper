"""
Command-line interface for the Agent Evaluation Framework.

This module provides easy-to-use command line tools for running agent evaluations,
tournaments, and analyzing results.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from agents import (  # Import from unified agents module
    HeuristicAgent,
    ImprovedHeuristicAgent,
    MinimaxAgent,
    OpenSpielMinimaxAgent,
    RandomAgent,
)
from agents.openspiel_agents import OpenSpielMCTSAgent as MCTSAgent
from evaluation.agent_evaluator import AgentEvaluator
from evaluation.evaluation_config import EvaluationConfig
from evaluation.tournament import Tournament
from evaluation.utils import (
    create_evaluation_summary,
    ensure_evaluation_dir,
    format_evaluation_results,
    save_evaluation_results,
)


def create_agent(agent_type: str, **kwargs):
    """Create an agent of the specified type."""
    agent_type = agent_type.lower()

    if agent_type == "heuristic":
        return HeuristicAgent(**kwargs)
    elif agent_type == "improved_heuristic":
        return ImprovedHeuristicAgent(**kwargs)
    elif agent_type == "random":
        return RandomAgent(**kwargs)
    elif agent_type == "mcts":
        # OpenSpiel MCTS - filter parameters to only those supported
        # Extract common MCTS parameters from kwargs and map to OpenSpiel parameters
        num_simulations = kwargs.get("num_simulations", 400)
        uct_c = kwargs.get(
            "uct_c", kwargs.get("c_puct", 1.4)
        )  # Map old c_puct to uct_c
        solve = kwargs.get("solve", False)
        seed = kwargs.get("seed", None)
        player_id = kwargs.get("player_id", 0)
        name = kwargs.get("name", None)
        max_memory = kwargs.get("max_memory", 1000000)
        evaluator = kwargs.get("evaluator", None)

        # Create with only supported parameters
        openspiel_kwargs = {
            "num_simulations": num_simulations,
            "uct_c": uct_c,
            "solve": solve,
            "seed": seed,
            "player_id": player_id,
            "name": name,
            "max_memory": max_memory,
            "evaluator": evaluator,
        }

        return MCTSAgent(**openspiel_kwargs)
    elif agent_type == "minimax":
        # Custom minimax agent with alpha-beta pruning
        time_limit = kwargs.get("time_limit", 1.0)
        max_depth = kwargs.get("max_depth", 4)
        player_id = kwargs.get("player_id", 0)
        name = kwargs.get("name", None)

        return MinimaxAgent(
            player_id=player_id, time_limit=time_limit, max_depth=max_depth, name=name
        )
    elif agent_type == "openspiel_minimax":
        # OpenSpiel minimax agent
        depth = kwargs.get("depth", 4)
        enable_alpha_beta = kwargs.get("enable_alpha_beta", True)
        time_limit = kwargs.get("time_limit", None)
        seed = kwargs.get("seed", None)
        player_id = kwargs.get("player_id", 0)
        name = kwargs.get("name", None)

        return OpenSpielMinimaxAgent(
            depth=depth,
            enable_alpha_beta=enable_alpha_beta,
            time_limit=time_limit,
            seed=seed,
            player_id=player_id,
            name=name,
        )
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: heuristic, improved_heuristic, random, mcts, minimax, openspiel_minimax"
        )


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
        if args.verbose:
            print(f"Creating {args.agent_a} agent...")
        agent_a = create_agent(args.agent_a, **parse_agent_args(args.agent_a_args))

        if args.verbose:
            print(f"Creating {args.agent_b} agent...")
        agent_b = create_agent(args.agent_b, **parse_agent_args(args.agent_b_args))
    except Exception as e:
        print(f"Error creating agents: {e}")
        return 1

    # Run evaluation
    evaluator = AgentEvaluator(config)

    try:
        if args.verbose:
            print(f"Starting evaluation: {agent_a.name} vs {agent_b.name}")
            print(
                f"Configuration: {config.num_games} games, {config.timeout_per_move}s timeout"
            )

        result = evaluator.evaluate_agent(
            test_agent=agent_a,
            baseline_agent=agent_b,
            test_agent_name=args.agent_a_name,
            baseline_agent_name=args.agent_b_name,
        )

        # Print results
        print(format_evaluation_results(result, detailed=args.detailed))

        # Save results if requested
        if args.output:
            ensure_evaluation_dir(str(Path(args.output).parent))
            save_evaluation_results(result, args.output)
            if args.verbose:
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

                agent = create_agent(agent_type, **agent_args)

                agents[name] = agent

            except Exception as e:
                print(f"Error creating agent {agent_spec}: {e}")
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
        test_agent = create_agent(args.test_agent)
        # Default opponent is random agent
        opponent_agent = create_agent(args.opponent or "random")
    except Exception as e:
        print(f"Error creating agents: {e}")
        return 1

    # Run quick evaluation
    evaluator = AgentEvaluator(config)

    try:
        result = evaluator.evaluate_agent(
            test_agent=test_agent, baseline_agent=opponent_agent
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
    --agent-a mcts --agent-b random \\
    --num-games 100 --output results.json

  # Verbose evaluation (show progress)
  python -m evaluation.cli evaluate \\
    --agent-a mcts --agent-b random \\
    --num-games 50 --verbose

  # Tournament between multiple agents
  python -m evaluation.cli tournament \\
    --agents "MCTS:mcts" "Heuristic:heuristic" "Random:random" \\
    --output tournament_results

  # Generate summary of results
  python -m evaluation.cli summary evaluation_results/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single agent")
    eval_parser.add_argument(
        "--agent-a",
        required=True,
        choices=[
            "heuristic",
            "improved_heuristic",
            "mcts",
            "minimax",
            "openspiel_minimax",
            "random",
        ],
        help="Type of first agent",
    )
    eval_parser.add_argument("--agent-a-name", help="Name for first agent")
    eval_parser.add_argument(
        "--agent-a-args",
        default="",
        help="Arguments for first agent (key=value,key=value)",
    )
    eval_parser.add_argument(
        "--agent-b",
        required=True,
        choices=[
            "heuristic",
            "improved_heuristic",
            "mcts",
            "minimax",
            "openspiel_minimax",
            "random",
        ],
        help="Type of second agent",
    )
    eval_parser.add_argument("--agent-b-name", help="Name for second agent")
    eval_parser.add_argument(
        "--agent-b-args",
        default="",
        help="Arguments for second agent (key=value,key=value)",
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
        "--verbose", action="store_true", help="Enable verbose output"
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
        "--verbose", action="store_true", help="Enable verbose output"
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
        choices=[
            "heuristic",
            "improved_heuristic",
            "mcts",
            "minimax",
            "openspiel_minimax",
            "random",
        ],
        help="Test agent type",
    )
    quick_parser.add_argument(
        "--opponent",
        choices=[
            "heuristic",
            "improved_heuristic",
            "mcts",
            "minimax",
            "openspiel_minimax",
            "random",
        ],
        help="Type of opponent agent",
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
