#!/usr/bin/env python3
"""
Flask API Server for Azul AI Agent

This server provides a REST API interface to the Python MCTS agent,
allowing the webapp to use the advanced Python AI instead of the
client-side TypeScript AI.
"""

import argparse
import json
import time
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from agents.mcts import MCTSAgent
from agents.heuristic_agent import HeuristicAgent
from game.game_state import Action, GameState, TileColor
from training.neural_network import AzulNeuralNetwork
from game.tile import Tile

app = Flask(__name__)
CORS(app)  # Enable CORS for webapp integration

# Global agent instances
agent: Optional[MCTSAgent] = None
heuristic_agent: Optional[HeuristicAgent] = None
neural_network: Optional[AzulNeuralNetwork] = None
current_agent_type: str = "auto"  # "mcts", "heuristic", or "auto"


def find_available_port(start_port: int = 5000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")


def kill_process_on_port(port: int) -> bool:
    """Try to kill any process running on the specified port."""
    try:
        # Find process using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"🔄 Killing process {pid} on port {port}...")
                    subprocess.run(['kill', '-9', pid], check=False)
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
            s.bind(('localhost', preferred_port))
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
                        s.bind(('localhost', preferred_port))
                        print(f"✅ Successfully freed and using port {preferred_port}")
                        return preferred_port
                except OSError:
                    print(f"❌ Could not free port {preferred_port}, finding alternative...")
            else:
                print(f"❌ Could not kill process on port {preferred_port}, finding alternative...")
        
        # Find alternative port
        try:
            alternative_port = find_available_port(preferred_port + 1)
            print(f"🔄 Using alternative port {alternative_port}")
            return alternative_port
        except RuntimeError:
            print("❌ Could not find any available port")
            raise


def initialize_agent(agent_type: str = "auto", network_config: str = "medium", simulations: int = 800) -> tuple:
    """Initialize the appropriate agent type."""
    global agent, heuristic_agent, neural_network, current_agent_type
    
    current_agent_type = agent_type
    
    # Always initialize heuristic agent as fallback
    heuristic_agent = HeuristicAgent(player_id=1)
    print("✅ Initialized heuristic agent")
    
    if agent_type == "heuristic":
        agent = None
        neural_network = None
        print("🧠 Using heuristic agent only")
        return heuristic_agent, None
    
    # Try to initialize MCTS agent
    try:
        # Initialize neural network
        neural_network = AzulNeuralNetwork(network_config=network_config)
        print(f"✅ Initialized neural network: {neural_network.get_model_info()}")
        
        # Initialize MCTS agent
        agent = MCTSAgent(
            neural_network=neural_network,
            simulations=simulations,
            exploration_constant=1.4,
            temperature=1.0,
            use_dirichlet_noise=False  # Disable for deterministic play
        )
        
        print(f"🤖 Initialized MCTS agent with {simulations} simulations")
        
        if agent_type == "mcts":
            print("🎯 Using MCTS agent only")
        else:  # auto
            print("⚡ Using MCTS agent with heuristic fallback")
            
        return agent, neural_network
        
    except Exception as e:
        print(f"⚠️  Error initializing MCTS agent: {e}")
        
        if agent_type == "mcts":
            print("❌ MCTS agent required but failed to initialize")
            return None, None
        else:  # auto fallback
            agent = None
            neural_network = None
            print("🔄 Falling back to heuristic agent")
            return heuristic_agent, None


def get_active_agent() -> Optional[Any]:
    """Get the currently active agent."""
    if current_agent_type == "heuristic":
        return heuristic_agent
    elif current_agent_type == "mcts" and agent:
        return agent
    else:  # auto mode
        return agent if agent else heuristic_agent


def webapp_move_to_action(move_data: Dict[str, Any]) -> Action:
    """Convert webapp move format to Action object."""
    try:
        factory_index = move_data.get('factoryIndex', 0)
        tile_color_str = move_data.get('tile', 'red')
        line_index = move_data.get('lineIndex', 0)
        
        # Convert tile color string to TileColor enum
        tile_color = TileColor(tile_color_str.lower())
        
        # Create Action object using correct parameters
        # source: factory index (-1 for center)
        # color: TileColor enum
        # destination: pattern line index (0-4) or -1 for floor line
        action = Action(
            source=factory_index,
            color=tile_color,
            destination=line_index
        )
        
        return action
        
    except Exception as e:
        raise ValueError(f"Invalid move format: {e}")


def action_to_webapp_move(action: Action) -> Dict[str, Any]:
    """Convert Action object to webapp move format."""
    return {
        'factoryIndex': action.source,
        'tile': action.color.value,
        'lineIndex': action.destination
    }


def webapp_gamestate_to_python(gamestate_data: Dict[str, Any]) -> GameState:
    """Convert webapp game state to Python GameState object."""
    try:
        from game.tile import Tile, TileColor
        
        # Extract basic game info
        num_players = len(gamestate_data.get('playerBoards', []))
        current_player = gamestate_data.get('currentPlayer', 0)
        round_number = gamestate_data.get('round', 1)
        game_over = gamestate_data.get('gameOver', False)
        first_player_index = gamestate_data.get('firstPlayerIndex', 0)
        
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
                'red': TileColor.RED,
                'blue': TileColor.BLUE,
                'yellow': TileColor.YELLOW,
                'black': TileColor.BLACK,
                'white': TileColor.WHITE
            }
            
            tile_color = color_map.get(tile_str.lower())
            if tile_color:
                return tile_color
            else:
                raise ValueError(f"Unknown tile color: {tile_str}")
        
        # Reconstruct factory area
        webapp_factories = gamestate_data.get('factories', [])
        webapp_center = gamestate_data.get('center', [])
        
        # Set up factories
        for i, webapp_factory in enumerate(webapp_factories):
            if i < len(game.factory_area.factories):
                factory_tiles = []
                for tile_str in webapp_factory:
                    if tile_str and tile_str.lower() != 'firstplayer':  # Skip empty strings and first player
                        tile_color = convert_tile_color(tile_str)
                        factory_tiles.append(Tile(tile_color))
                game.factory_area.factories[i].tiles = factory_tiles
        
        # Set up center area
        center_tiles = []
        has_first_player = False
        for tile_str in webapp_center:
            if tile_str:  # Skip empty strings
                if tile_str.lower() == 'firstplayer':
                    has_first_player = True
                else:
                    tile_color = convert_tile_color(tile_str)
                    center_tiles.append(Tile(tile_color))
        
        game.factory_area.center.tiles = center_tiles
        game.factory_area.center.has_first_player_marker = has_first_player
        
        # Reconstruct player boards
        webapp_player_boards = gamestate_data.get('playerBoards', [])
        for i, webapp_board in enumerate(webapp_player_boards):
            if i < len(game.players):
                player_board = game.players[i]
                
                # Set score
                player_board.score = webapp_board.get('score', 0)
                
                # Reconstruct pattern lines
                webapp_lines = webapp_board.get('lines', [])
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
                webapp_wall = webapp_board.get('wall', [])
                for row_idx, webapp_row in enumerate(webapp_wall):
                    if row_idx < 5:
                        for col_idx, tile_present in enumerate(webapp_row):
                            if col_idx < 5:
                                # In webapp, wall[row][col] might be a tile or null/empty
                                # If it's a tile string, mark as filled
                                if tile_present and tile_present != 'null' and tile_present != '':
                                    player_board.wall.filled[row_idx][col_idx] = True
                                else:
                                    player_board.wall.filled[row_idx][col_idx] = False
                
                # Reconstruct floor line
                webapp_floor = webapp_board.get('floor', [])
                player_board.floor_line = []
                has_first_player_on_floor = False
                
                for tile_str in webapp_floor:
                    if tile_str:  # Skip empty strings
                        if tile_str.lower() == 'firstplayer':
                            # Create first player marker tile
                            first_player_tile = Tile.create_first_player_marker()
                            player_board.floor_line.append(first_player_tile)
                            has_first_player_on_floor = True
                        else:
                            tile_color = convert_tile_color(tile_str)
                            player_board.floor_line.append(Tile(tile_color))
                
                # If this player has first player marker, they'll be first next round
                if has_first_player_on_floor:
                    first_player_index = i
        
        # The game state is now reconstructed with the visible board state
        # Note: We can't reconstruct the exact bag state, but agents only need
        # the current visible state for decision making
        
        return game
        
    except Exception as e:
        raise ValueError(f"Failed to convert game state: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    active_agent = get_active_agent()
    agent_type = "none"
    
    if active_agent == agent:
        agent_type = "mcts"
    elif active_agent == heuristic_agent:
        agent_type = "heuristic"
    
    # Get current server info
    server_info = {
        'host': request.host.split(':')[0],  # Remove port from host
        'port': request.environ.get('SERVER_PORT', '5000'),
        'url': request.url_root.rstrip('/')
    }
    
    return jsonify({
        'status': 'healthy',
        'agent_initialized': active_agent is not None,
        'neural_network_available': neural_network is not None,
        'current_agent_type': current_agent_type,
        'active_agent_type': agent_type,
        'mcts_available': agent is not None,
        'heuristic_available': heuristic_agent is not None,
        'server': server_info,
        'timestamp': time.time()
    })


@app.route('/agent/info', methods=['GET'])
def get_agent_info():
    """Get information about the current agent."""
    active_agent = get_active_agent()
    
    if not active_agent:
        return jsonify({'error': 'No agent initialized'}), 500
    
    info = {
        'current_agent_type': current_agent_type,
        'active_agent': 'mcts' if active_agent == agent else 'heuristic',
        'neural_network_available': neural_network is not None
    }
    
    if active_agent == agent and agent:
        info.update({
            'simulations': agent.simulations,
            'exploration_constant': agent.exploration_constant,
            'temperature': agent.temperature,
            'algorithm': 'MCTS'
        })
    elif active_agent == heuristic_agent:
        stats = heuristic_agent.get_stats()
        info.update({
            'algorithm': stats['algorithm'],
            'features': stats['features']
        })
    
    if neural_network:
        info['neural_network_info'] = neural_network.get_model_info()
    
    return jsonify(info)


@app.route('/agent/move', methods=['POST'])
def get_best_move():
    """Get the best move for the current game state."""
    active_agent = get_active_agent()
    
    if not active_agent:
        return jsonify({'error': 'No agent initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        gamestate_data = data.get('gameState')
        thinking_time = data.get('thinkingTime', 2000)  # milliseconds
        
        if not gamestate_data:
            return jsonify({'error': 'No game state provided'}), 400
        
        # Convert webapp game state to Python GameState
        # Note: This is a placeholder - you'll need to implement full conversion
        try:
            game_state = webapp_gamestate_to_python(gamestate_data)
        except Exception as e:
            return jsonify({'error': f'Failed to parse game state: {str(e)}'}), 400
        
        # Get the best move
        start_time = time.time()
        try:
            if active_agent == agent:  # MCTS agent
                # Adjust agent simulations based on thinking time
                # Rough estimate: 1000 simulations per second
                estimated_simulations = max(100, min(5000, thinking_time))
                agent.simulations = estimated_simulations
                
                action = agent.select_action(game_state)
                nodes_evaluated = getattr(agent, 'nodes_evaluated', estimated_simulations)
                algorithm_info = f"MCTS ({agent.simulations} simulations)"
                
            else:  # Heuristic agent
                action = heuristic_agent.select_action(game_state)
                stats = heuristic_agent.get_stats()
                nodes_evaluated = stats['nodesEvaluated']
                algorithm_info = f"Heuristic ({stats['algorithm']})"
            
            search_time = time.time() - start_time
            
            # Convert action back to webapp format
            move = action_to_webapp_move(action)
            
            # Get search statistics
            response_stats = {
                'nodesEvaluated': nodes_evaluated,
                'searchTime': search_time,
                'simulations': getattr(active_agent, 'simulations', 0),
                'algorithm': algorithm_info,
                'agent_type': 'mcts' if active_agent == agent else 'heuristic'
            }
            
            return jsonify({
                'move': move,
                'stats': response_stats,
                'success': True
            })
            
        except Exception as e:
            return jsonify({'error': f'Agent failed to select move: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500


@app.route('/agent/configure', methods=['POST'])
def configure_agent():
    """Configure agent parameters."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Extract configuration parameters
        agent_type = data.get('agentType', 'auto')
        network_config = data.get('networkConfig', 'medium')
        simulations = data.get('simulations', 800)
        
        # Reinitialize agent with new parameters
        active_agent, network = initialize_agent(agent_type, network_config, simulations)
        
        if not active_agent:
            return jsonify({'error': 'Failed to initialize requested agent type'}), 500
        
        return jsonify({
            'success': True,
            'message': f'Agent reconfigured successfully',
            'agent_type': agent_type,
            'active_agent': 'mcts' if active_agent == agent else 'heuristic'
        })
        
    except Exception as e:
        return jsonify({'error': f'Configuration failed: {str(e)}'}), 500


@app.route('/agent/types', methods=['GET'])
def get_agent_types():
    """Get available agent types."""
    return jsonify({
        'available_types': [
            {
                'id': 'auto',
                'name': 'Auto (MCTS with Heuristic Fallback)',
                'description': 'Uses MCTS when available, falls back to heuristic agent'
            },
            {
                'id': 'mcts',
                'name': 'MCTS Only',
                'description': 'Monte Carlo Tree Search with neural network guidance'
            },
            {
                'id': 'heuristic',
                'name': 'Heuristic Only', 
                'description': 'Rule-based agent with Azul strategy heuristics'
            }
        ],
        'current_type': current_agent_type
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Azul AI API Server')
    parser.add_argument('--port', '-p', type=int, default=5000, 
                       help='Preferred port number (default: 5000)')
    parser.add_argument('--kill-existing', '-k', action='store_true',
                       help='Kill existing process on preferred port if occupied')
    parser.add_argument('--agent-type', '-a', choices=['auto', 'mcts', 'heuristic'], 
                       default='auto', help='Agent type to initialize (default: auto)')
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Host to bind to (default: 0.0.0.0)')
    
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