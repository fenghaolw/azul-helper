#!/usr/bin/env python3
"""
Flask API Server for Azul AI Agent

This server provides a REST API interface to the Python MCTS agent,
allowing the webapp to use the advanced Python AI instead of the
client-side TypeScript AI.
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from agents.heuristic_agent import HeuristicAgent
from agents.improved_heuristic_agent import ImprovedHeuristicAgent
from agents.minimax_agent import MinimaxAgent, MinimaxConfig
from agents.openspiel_agents import OpenSpielMCTSAgent
from game.game_state import Action, GameState, TileColor
from game.tile import Tile

app = Flask(__name__)
CORS(app)  # Enable CORS for webapp integration

# Global agent instances
mcts_agent: Optional[OpenSpielMCTSAgent] = None
heuristic_agent: Optional[HeuristicAgent] = None
improved_heuristic_agent: Optional[ImprovedHeuristicAgent] = None
minimax_agent: Optional[MinimaxAgent] = None
current_agent_type: str = "minimax"  # Changed default to minimax


def find_available_port(start_port: int = 5000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}"
    )


def kill_process_on_port(port: int) -> bool:
    """Try to kill any process running on the specified port."""
    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    print(f"🔄 Killing process {pid} on port {port}...")
                    subprocess.run(["kill", "-9", pid], check=False)
            return True
        return False
    except Exception as e:
        print(f"⚠️  Could not kill process on port {port}: {e}")
        return False


def setup_server_port(preferred_port: int = 5000, kill_existing: bool = False) -> int:
    """Setup server port, handling conflicts intelligently."""
    # First, try the preferred port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", preferred_port))
            print(f"✅ Port {preferred_port} is available")
            return preferred_port
    except OSError:
        print(f"⚠️  Port {preferred_port} is already in use")

        if kill_existing:
            print(f"🔄 Attempting to free port {preferred_port}...")
            if kill_process_on_port(preferred_port):
                # Wait a moment for the port to be freed
                time.sleep(1)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("localhost", preferred_port))
                        print(f"✅ Successfully freed and using port {preferred_port}")
                        return preferred_port
                except OSError:
                    print(
                        f"❌ Could not free port {preferred_port}, finding alternative..."
                    )
            else:
                print(
                    f"❌ Could not kill process on port {preferred_port}, finding alternative..."
                )

        # Find alternative port
        try:
            alternative_port = find_available_port(preferred_port + 1)
            print(f"🔄 Using alternative port {alternative_port}")
            return alternative_port
        except RuntimeError:
            print("❌ Could not find any available port")
            raise


def initialize_agent(
    agent_type: str = "auto",
    network_config: str = "medium",
    simulations: int = 800,
    minimax_difficulty: str = "medium",
    minimax_config: Optional[dict] = None,
) -> tuple:
    """Initialize the appropriate agent type."""
    global mcts_agent, heuristic_agent, improved_heuristic_agent, minimax_agent, current_agent_type

    current_agent_type = agent_type

    # Always initialize heuristic agent as fallback
    heuristic_agent = HeuristicAgent(player_id=1)
    print("✅ Initialized heuristic agent")

    # Initialize improved heuristic agent
    improved_heuristic_agent = ImprovedHeuristicAgent(player_id=1)
    print("✅ Initialized improved heuristic agent")

    # Initialize minimax agent
    if minimax_config:
        # Create custom config from dictionary
        config = MinimaxConfig(**minimax_config)
        minimax_agent = MinimaxAgent(player_id=1, config=config)
    else:
        # Use difficulty preset
        minimax_agent = MinimaxAgent(
            player_id=1,
            config=MinimaxConfig.create_difficulty_preset(minimax_difficulty),
        )
    print(f"✅ Initialized minimax agent (difficulty: {minimax_difficulty})")

    if agent_type == "heuristic":
        mcts_agent = None
        print("🧠 Using heuristic agent only")
        return heuristic_agent, None

    if agent_type == "improved_heuristic":
        mcts_agent = None
        print("🚀 Using improved heuristic agent only")
        return improved_heuristic_agent, None

    if agent_type == "minimax":
        mcts_agent = None
        print("🎯 Using minimax agent only")
        return minimax_agent, None

    # Try to initialize MCTS agent
    try:
        # Note: OpenSpiel MCTS doesn't require neural networks, it uses random rollouts
        # Initialize OpenSpiel MCTS agent
        mcts_agent = OpenSpielMCTSAgent(
            num_simulations=simulations,
            uct_c=1.4,  # UCT exploration constant
            solve=False,  # Don't use MCTS-Solver for speed
            seed=None,  # Use random seed
        )

        print(f"🤖 Initialized OpenSpiel MCTS agent with {simulations} simulations")

        if agent_type == "mcts":
            print("🎯 Using OpenSpiel MCTS agent only")
        else:  # auto
            print("⚡ Using OpenSpiel MCTS agent with heuristic fallback")

        return mcts_agent, None

    except Exception as e:
        print(f"⚠️  Error initializing MCTS agent: {e}")

        if agent_type == "mcts":
            print("❌ MCTS agent required but failed to initialize")
            return None, None
        else:  # auto fallback
            mcts_agent = None
            print("🔄 Falling back to heuristic agent")
            return heuristic_agent, None


def get_active_agent() -> Optional[Any]:
    """Get the currently active agent."""
    if current_agent_type == "heuristic":
        return heuristic_agent
    elif current_agent_type == "improved_heuristic":
        return improved_heuristic_agent
    elif current_agent_type == "minimax":
        return minimax_agent
    elif current_agent_type == "mcts" and mcts_agent:
        return mcts_agent
    else:  # auto mode - prefer minimax, then MCTS, then improved heuristic
        return (
            minimax_agent
            if minimax_agent
            else (mcts_agent if mcts_agent else improved_heuristic_agent)
        )


def webapp_move_to_action(move_data: Dict[str, Any]) -> Action:
    """Convert webapp move format to Action object."""
    try:
        factory_index = move_data.get("factoryIndex", 0)
        tile_color_str = move_data.get("tile", "red")
        line_index = move_data.get("lineIndex", 0)

        # Convert tile color string to TileColor enum
        tile_color = TileColor(tile_color_str.lower())

        # Create Action object using correct parameters
        # source: factory index (-1 for center)
        # color: TileColor enum
        # destination: pattern line index (0-4) or -1 for floor line
        action = Action(source=factory_index, color=tile_color, destination=line_index)

        return action

    except Exception as e:
        raise ValueError(f"Invalid move format: {e}")


def action_to_webapp_move(action: Action) -> Dict[str, Any]:
    """Convert Action object to webapp move format."""
    return {
        "factoryIndex": action.source,
        "tile": action.color.value,
        "lineIndex": action.destination,
    }


def webapp_gamestate_to_python(gamestate_data: Dict[str, Any]) -> GameState:
    """Convert webapp game state to Python GameState object."""
    try:
        # Extract basic game info
        num_players = len(gamestate_data.get("playerBoards", []))
        current_player = gamestate_data.get("currentPlayer", 0)
        round_number = gamestate_data.get("round", 1)
        game_over = gamestate_data.get("gameOver", False)

        # Create new game state
        game = GameState(num_players=num_players)

        # Set basic game state (using correct attribute names)
        game.current_player = current_player
        game.round_number = round_number
        game.game_over = game_over

        # Helper function to convert webapp tile strings to Python TileColor enums
        def convert_tile_color(tile_str: str) -> TileColor:
            """Convert webapp tile string to Python TileColor enum."""
            color_map = {
                "red": TileColor.RED,
                "blue": TileColor.BLUE,
                "yellow": TileColor.YELLOW,
                "black": TileColor.BLACK,
                "white": TileColor.WHITE,
            }

            tile_color = color_map.get(tile_str.lower())
            if tile_color:
                return tile_color
            else:
                raise ValueError(f"Unknown tile color: {tile_str}")

        # Reconstruct factory area
        webapp_factories = gamestate_data.get("factories", [])
        webapp_center = gamestate_data.get("center", [])

        # Set up factories
        for i, webapp_factory in enumerate(webapp_factories):
            if i < len(game.factory_area.factories):
                factory_tiles = []
                for tile_str in webapp_factory:
                    if (
                        tile_str and tile_str.lower() != "firstplayer"
                    ):  # Skip empty strings and first player
                        tile_color = convert_tile_color(tile_str)
                        factory_tiles.append(Tile(tile_color))
                game.factory_area.factories[i].tiles = factory_tiles

        # Set up center area
        center_tiles = []
        has_first_player = False
        for tile_str in webapp_center:
            if tile_str:  # Skip empty strings
                if tile_str.lower() == "firstplayer":
                    has_first_player = True
                else:
                    tile_color = convert_tile_color(tile_str)
                    center_tiles.append(Tile(tile_color))

        game.factory_area.center.tiles = center_tiles
        game.factory_area.center.has_first_player_marker = has_first_player

        # Reconstruct player boards
        webapp_player_boards = gamestate_data.get("playerBoards", [])
        for i, webapp_board in enumerate(webapp_player_boards):
            if i < len(game.players):
                player_board = game.players[i]

                # Set score
                player_board.score = webapp_board.get("score", 0)

                # Reconstruct pattern lines
                webapp_lines = webapp_board.get("lines", [])
                for line_idx, webapp_line in enumerate(webapp_lines):
                    if line_idx < len(player_board.pattern_lines):
                        pattern_line = player_board.pattern_lines[line_idx]

                        # Clear existing tiles
                        pattern_line.tiles = []
                        pattern_line.color = None

                        # Add tiles from webapp
                        if webapp_line:
                            for tile_str in webapp_line:
                                if tile_str:  # Skip empty strings
                                    tile_color = convert_tile_color(tile_str)
                                    pattern_line.tiles.append(Tile(tile_color))

                            # Set color if tiles exist
                            if pattern_line.tiles:
                                pattern_line.color = pattern_line.tiles[0].color

                # Reconstruct wall
                webapp_wall = webapp_board.get("wall", [])
                for row_idx, webapp_row in enumerate(webapp_wall):
                    if row_idx < 5:
                        for col_idx, tile_present in enumerate(webapp_row):
                            if col_idx < 5:
                                # In webapp, wall[row][col] might be a tile or null/empty
                                # If it's a tile string, mark as filled
                                if (
                                    tile_present
                                    and tile_present != "null"
                                    and tile_present != ""
                                ):
                                    player_board.wall.filled[row_idx][col_idx] = True
                                else:
                                    player_board.wall.filled[row_idx][col_idx] = False

                # Reconstruct floor line
                webapp_floor = webapp_board.get("floor", [])
                player_board.floor_line = []
                has_first_player_on_floor = False

                for tile_str in webapp_floor:
                    if tile_str:  # Skip empty strings
                        if tile_str.lower() == "firstplayer":
                            # Create first player marker tile
                            first_player_tile = Tile.create_first_player_marker()
                            player_board.floor_line.append(first_player_tile)
                            has_first_player_on_floor = True
                        else:
                            tile_color = convert_tile_color(tile_str)
                            player_board.floor_line.append(Tile(tile_color))

                # If this player has first player marker, they'll be first next round
                if has_first_player_on_floor:
                    pass  # Track first player marker was found

        # The game state is now reconstructed with the visible board state
        # Note: We can't reconstruct the exact bag state, but agents only need
        # the current visible state for decision making

        return game

    except Exception as e:
        raise ValueError(f"Failed to convert game state: {e}")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    active_agent = get_active_agent()
    agent_type = "none"

    if active_agent == mcts_agent:
        agent_type = "mcts"
    elif active_agent == minimax_agent:
        agent_type = "minimax"
    elif active_agent == improved_heuristic_agent:
        agent_type = "improved_heuristic"
    elif active_agent == heuristic_agent:
        agent_type = "heuristic"

    # Get current server info
    server_info = {
        "host": request.host.split(":")[0],  # Remove port from host
        "port": request.environ.get("SERVER_PORT", "5000"),
        "url": request.url_root.rstrip("/"),
    }

    return jsonify(
        {
            "status": "healthy",
            "agent_initialized": active_agent is not None,
            "current_agent_type": current_agent_type,
            "active_agent_type": agent_type,
            "mcts_available": mcts_agent is not None,
            "minimax_available": minimax_agent is not None,
            "heuristic_available": heuristic_agent is not None,
            "server": server_info,
            "timestamp": time.time(),
        }
    )


@app.route("/agent/info", methods=["GET"])
def get_agent_info():
    """Get information about the current agent."""
    active_agent = get_active_agent()

    if not active_agent:
        return jsonify({"error": "No agent initialized"}), 500

    info: Dict[str, Any] = {
        "current_agent_type": current_agent_type,
        "active_agent": (
            "mcts"
            if active_agent == mcts_agent
            else (
                "minimax"
                if active_agent == minimax_agent
                else (
                    "improved_heuristic"
                    if active_agent == improved_heuristic_agent
                    else "heuristic"
                )
            )
        ),
    }

    if active_agent == mcts_agent:
        info.update(
            {
                "simulations": str(mcts_agent.num_simulations),
                "exploration_constant": str(mcts_agent.uct_c),
                "algorithm": "OpenSpiel MCTS",
            }
        )
    elif active_agent == minimax_agent:
        stats = minimax_agent.get_stats()
        config = minimax_agent.get_info()["config"]
        info["algorithm"] = "Minimax Alpha-Beta"
        info["features"] = (
            f"Time limit: {config['time_limit']}s, Max depth: {config['max_depth'] or 'adaptive'}"
        )
        info["max_depth_reached"] = stats.get("max_depth_reached", 0)
        info["time_limit"] = config["time_limit"]
        info["max_depth"] = config["max_depth"]
        info["config"] = config
    elif active_agent == improved_heuristic_agent:
        stats = improved_heuristic_agent.get_stats()
        info["algorithm"] = stats.get("algorithm", "Improved Heuristic")
        info["features"] = stats.get("features", "Unknown")
    elif active_agent == heuristic_agent:
        stats = heuristic_agent.get_stats()
        info["algorithm"] = stats.get("algorithm", "Heuristic")
        info["features"] = stats.get("features", "Unknown")

    return jsonify(info)


@app.route("/agent/move", methods=["POST"])
def get_best_move():
    """Get the best move for the current game state."""
    active_agent = get_active_agent()

    if not active_agent:
        return jsonify({"error": "No agent initialized"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract parameters
        gamestate_data = data.get("gameState")

        if not gamestate_data:
            return jsonify({"error": "No game state provided"}), 400

        # Convert webapp game state to Python GameState
        # Note: This is a placeholder - you'll need to implement full conversion
        try:
            game_state = webapp_gamestate_to_python(gamestate_data)
        except Exception as e:
            return jsonify({"error": f"Failed to parse game state: {str(e)}"}), 400

        # Get the best move
        start_time = time.time()
        try:
            if active_agent == mcts_agent:  # MCTS agent
                # OpenSpiel MCTS doesn't dynamically adjust simulations during runtime
                # We use the pre-configured simulation count
                action = mcts_agent.select_action(game_state, deterministic=True)

                # OpenSpiel MCTS doesn't expose nodes_evaluated, approximate with simulations
                nodes_evaluated = mcts_agent.num_simulations
                algorithm_info = (
                    f"OpenSpiel MCTS ({mcts_agent.num_simulations} simulations)"
                )
                agent_type_name = "mcts"

            elif active_agent == improved_heuristic_agent:  # Improved Heuristic agent
                action = improved_heuristic_agent.select_action(game_state)  # type: ignore[union-attr]
                stats = improved_heuristic_agent.get_stats()  # type: ignore[union-attr]
                nodes_evaluated = stats["nodesEvaluated"]
                algorithm_info = f"Improved Heuristic ({stats['algorithm']})"
                agent_type_name = "improved_heuristic"

            elif active_agent == minimax_agent:  # Minimax agent
                action = minimax_agent.select_action(game_state)  # type: ignore[union-attr]
                stats = minimax_agent.get_stats()  # type: ignore[union-attr]
                nodes_evaluated = stats["nodes_evaluated"]
                max_depth = stats["max_depth_reached"]
                algorithm_info = f"Minimax α-β (depth {max_depth})"
                agent_type_name = "minimax"

            else:  # Original Heuristic agent
                action = heuristic_agent.select_action(game_state)  # type: ignore[union-attr]
                stats = heuristic_agent.get_stats()  # type: ignore[union-attr]
                nodes_evaluated = stats["nodesEvaluated"]
                algorithm_info = f"Heuristic ({stats['algorithm']})"
                agent_type_name = "heuristic"

            search_time = time.time() - start_time

            # Convert action back to webapp format
            move = action_to_webapp_move(action)

            # Get search statistics
            response_stats = {
                "nodesEvaluated": nodes_evaluated,
                "searchTime": search_time,
                "simulations": (
                    mcts_agent.num_simulations if active_agent == mcts_agent else 0
                ),
                "algorithm": algorithm_info,
                "agent_type": agent_type_name,
            }

            return jsonify({"move": move, "stats": response_stats, "success": True})

        except Exception as e:
            return jsonify({"error": f"Agent failed to select move: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Request processing failed: {str(e)}"}), 500


@app.route("/agent/configure", methods=["POST"])
def configure_agent():
    """Configure agent parameters."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400

        # Extract configuration parameters
        agent_type = data.get("agentType", "auto")
        network_config = data.get("networkConfig", "medium")
        simulations = data.get("simulations", 800)
        minimax_difficulty = data.get("minimaxDifficulty", "medium")
        minimax_config = data.get("minimaxConfig", None)

        # Reinitialize agent with new parameters
        active_agent, network = initialize_agent(
            agent_type, network_config, simulations, minimax_difficulty, minimax_config
        )

        if not active_agent:
            return jsonify({"error": "Failed to initialize requested agent type"}), 500

        return jsonify(
            {
                "success": True,
                "message": f"Agent reconfigured successfully",
                "agent_type": agent_type,
                "active_agent": (
                    "mcts"
                    if active_agent == mcts_agent
                    else (
                        "minimax"
                        if active_agent == minimax_agent
                        else (
                            "improved_heuristic"
                            if active_agent == improved_heuristic_agent
                            else "heuristic"
                        )
                    )
                ),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Configuration failed: {str(e)}"}), 500


@app.route("/agent/types", methods=["GET"])
def get_agent_types():
    """Get available agent types."""
    return jsonify(
        {
            "available_types": [
                {
                    "id": "auto",
                    "name": "Auto (Minimax with Fallbacks)",
                    "description": "Uses Minimax when available, falls back to MCTS or improved heuristic agent",
                },
                {
                    "id": "minimax",
                    "name": "Minimax Alpha-Beta",
                    "description": "Minimax algorithm with alpha-beta pruning and iterative deepening",
                },
                {
                    "id": "mcts",
                    "name": "MCTS Only",
                    "description": "Monte Carlo Tree Search with neural network guidance",
                },
                {
                    "id": "improved_heuristic",
                    "name": "Improved Heuristic",
                    "description": "Advanced rule-based agent with strategic guidelines for competitive play",
                },
                {
                    "id": "heuristic",
                    "name": "Original Heuristic",
                    "description": "Basic rule-based agent with Azul strategy heuristics",
                },
            ],
            "current_type": current_agent_type,
        }
    )


@app.route("/agent/minimax/presets", methods=["GET"])
def get_minimax_presets():
    """Get available minimax difficulty presets."""
    presets = {
        "easy": {
            "name": "Easy",
            "description": "Quick decisions, shallow search",
            "time_limit": 0.3,
            "max_depth": 2,
            "features": ["Basic minimax", "No move ordering"],
        },
        "medium": {
            "name": "Medium",
            "description": "Balanced performance and strength",
            "time_limit": 0.7,
            "max_depth": 4,
            "features": ["Iterative deepening", "Move ordering", "Alpha-beta pruning"],
        },
        "hard": {
            "name": "Hard",
            "description": "Strong play with longer thinking",
            "time_limit": 1.5,
            "max_depth": 6,
            "features": ["Adaptive depth", "Advanced pruning", "Smart node limits"],
        },
        "expert": {
            "name": "Expert",
            "description": "Maximum strength, extended analysis",
            "time_limit": 3.0,
            "max_depth": 8,
            "features": ["Deep search", "Full feature set", "Optimal play"],
        },
        "custom": {
            "name": "Custom",
            "description": "User-defined configuration",
            "time_limit": "configurable",
            "max_depth": "configurable",
            "features": ["All features configurable"],
        },
    }

    return jsonify(
        {"presets": presets, "current_preset": "medium", "success": True}  # Default
    )


@app.route("/agent/minimax/configure", methods=["POST"])
def configure_minimax():
    """Configure minimax agent with specific parameters."""
    if not minimax_agent:
        return jsonify({"error": "Minimax agent not initialized"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400

        difficulty = data.get("difficulty")
        custom_config = data.get("config")

        if difficulty and difficulty != "custom":
            # Use preset
            print(f"🔧 Configuring minimax agent to {difficulty} difficulty")
            minimax_agent.set_difficulty_preset(difficulty)
            config_info = minimax_agent.get_info()["config"]
            print(f"✅ New minimax config: {config_info}")

            return jsonify(
                {
                    "success": True,
                    "message": f"Minimax agent configured to {difficulty} difficulty",
                    "difficulty": difficulty,
                    "config": config_info,
                }
            )

        elif custom_config:
            # Use custom configuration
            minimax_agent.update_config(**custom_config)
            config_info = minimax_agent.get_info()["config"]

            return jsonify(
                {
                    "success": True,
                    "message": "Minimax agent configured with custom settings",
                    "difficulty": "custom",
                    "config": config_info,
                }
            )

        else:
            return (
                jsonify(
                    {"error": "Either difficulty preset or custom config required"}
                ),
                400,
            )

    except Exception as e:
        return jsonify({"error": f"Configuration failed: {str(e)}"}), 500


@app.route("/agent/minimax/info", methods=["GET"])
def get_minimax_info():
    """Get detailed minimax agent information and current configuration."""
    if not minimax_agent:
        return jsonify({"error": "Minimax agent not initialized"}), 500

    info = minimax_agent.get_info()
    stats = minimax_agent.get_stats()

    return jsonify(
        {
            "info": info,
            "stats": stats,
            "performance": {
                "nodes_per_second": stats["nodes_evaluated"]
                / max(0.001, stats.get("last_move_time", 1.0)),
                "effective_branching_factor": "calculated_from_search_tree",
                "average_depth": stats["max_depth_reached"],
            },
            "success": True,
        }
    )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Azul AI API Server")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Preferred port number (default: 5000)",
    )
    parser.add_argument(
        "--kill-existing",
        "-k",
        action="store_true",
        help="Kill existing process on preferred port if occupied",
    )
    parser.add_argument(
        "--agent-type",
        "-a",
        choices=["auto", "mcts", "heuristic", "improved_heuristic", "minimax"],
        default="minimax",
        help="Agent type to initialize (default: minimax)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    print("🎮 Starting Azul AI API Server...")

    # Setup port intelligently
    try:
        server_port = setup_server_port(args.port, args.kill_existing)
    except Exception as e:
        print(f"❌ Failed to setup server port: {e}")
        sys.exit(1)

    # Initialize the agent
    try:
        active_agent, network = initialize_agent(args.agent_type)
        if active_agent:
            print("✅ Agent initialized successfully")
        else:
            print("❌ Failed to initialize any agent")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize agent: {e}")
        print("Server will start but agent functionality may be limited")

    # Update webapp with the actual port being used
    if server_port != 5000:
        print(f"\n📝 Important: Server is running on port {server_port}, not 5000!")
        print(f"   You may need to update your webapp to use:")
        print(f"   http://localhost:{server_port}")

    # Start the server
    print(f"\n🚀 Server starting on http://{args.host}:{server_port}")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /agent/info - Agent information")
    print("  GET  /agent/types - Available agent types")
    print("  POST /agent/move - Get best move")
    print("  POST /agent/configure - Configure agent")
    print("  GET  /agent/minimax/presets - Get minimax difficulty presets")
    print("  POST /agent/minimax/configure - Configure minimax agent")
    print("  GET  /agent/minimax/info - Get minimax agent information")

    if args.kill_existing:
        print("\n⚠️  Note: --kill-existing was used. Be careful with this option.")

    print(f"\n🎯 Ready to serve intelligent moves!")
    print(f"💡 Pro tip: Use 'python3 api_server.py --help' to see all options")

    try:
        app.run(host=args.host, port=server_port, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print(f"\n👋 Server shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)
