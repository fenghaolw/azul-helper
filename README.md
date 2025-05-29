# Azul Helper

A comprehensive Azul board game AI research platform featuring multiple AI agent types, reinforcement learning capabilities, and both web application and browser extension interfaces.

## 🚀 Quick Start

### One-Command Setup

```bash
# Start everything (Python API server + TypeScript webapp)
python3 start.py

# Check what's running
python3 start.py --check-only

# Start only the API server (for development)
python3 start.py --server-only

# Start only the webapp (if server already running)
python3 start.py --ui-only
```

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm

## 🏗️ Architecture

This project features a **hybrid architecture** combining Python AI backends with TypeScript frontends:

### 🐍 Python Backend
- **Flask API Server**: Serves multiple AI agent types via REST API
- **Game Engine**: Pure Python implementation of Azul game mechanics
- **AI Agents**: Multiple sophisticated AI implementations
- **Training Infrastructure**: Reinforcement learning and neural network training
- **Utilities**: Game analysis, benchmarking, and testing tools

### 🌐 TypeScript Frontend
- **Web Application**: Full-featured game interface with canvas rendering
- **Browser Extension**: BGA integration for online play assistance
- **AI Integration**: Communicates with Python backend via HTTP API

## 🤖 AI Agent Types

The system supports multiple AI approaches, each with different strengths:

### 1. **Auto Agent (Recommended)**
- **Hybrid Approach**: MCTS with heuristic fallback
- **Adaptive**: Automatically switches based on position complexity
- **Robust**: Always finds a move even in difficult positions
- **Performance**: Best overall playing strength

### 2. **MCTS Agent**
- **Algorithm**: Monte Carlo Tree Search with neural network guidance
- **Strength**: Excellent positional understanding
- **Deep Search**: Explores many possible futures
- **Requirements**: More computationally intensive

### 3. **Heuristic Agent**
- **Algorithm**: Rule-based strategic evaluation
- **Speed**: Very fast move generation
- **Reliability**: Consistent performance
- **Strategy**: Hand-crafted game knowledge

### 4. **TypeScript Minimax Agent** (Legacy)
- **Algorithm**: Minimax with alpha-beta pruning
- **Local**: Runs entirely in the browser
- **Educational**: Great for understanding game tree search
- **Performance**: Good for learning and development

## 🎮 Features

### Web Application
- 🎯 **Complete Azul Implementation**: Full game rules and mechanics
- 🤖 **Multiple AI Opponents**: Choose from 4 different AI agent types
- 🎨 **Beautiful UI**: Portuguese azulejo-inspired design
- ⚡ **Real-time Gameplay**: Smooth canvas-based rendering
- 📊 **AI Statistics**: Live performance metrics and analysis
- 🎛️ **Flexible Difficulty**: Adjustable thinking time from 0.5s to 5s
- 🔄 **Auto-Discovery**: Frontend automatically finds backend server

### Browser Extension
- 🌐 **BGA Integration**: Works with Board Game Arena's Azul
- 🧠 **AI Analysis**: Get move suggestions while playing online
- 📈 **Position Evaluation**: See how AI evaluates current position
- ⚙️ **Configurable**: Adjust analysis depth and agent type

### Training & Research
- 🧪 **RL Training**: Reinforcement learning agent development
- 📊 **Benchmarking**: Compare different AI approaches
- 🔬 **Analysis Tools**: Game state analysis and visualization
- 💾 **Model Management**: Save, load, and version AI models

## 🎯 How to Play

Azul is a tile-laying game where players compete to decorate palace walls with Portuguese tiles.

### Basic Rules
1. **Setup**: Factories filled with 4 random tiles each round
2. **Selection**: Pick all tiles of one color from a factory or center
3. **Placement**: Place tiles in pattern lines (1-5 tiles per line)
4. **Scoring**: Complete lines move to wall and score points
5. **Penalties**: Excess tiles go to floor for penalty points
6. **Victory**: Game ends when someone completes a horizontal row

### Scoring System
- **Basic**: 1 point per tile + adjacency bonuses
- **Row Bonus**: 2 points per completed horizontal row
- **Column Bonus**: 7 points per completed vertical column  
- **Color Bonus**: 10 points per complete color set (5 tiles)
- **Floor Penalties**: -1, -1, -2, -2, -2, -3, -3 for floor tiles

## 🔧 Advanced Setup

### Python Environment

```bash
# Install Python dependencies
pip install -r requirements.txt

# Development dependencies (for training/research)
pip install -r requirements-dev.txt
```

### Manual Server Control

```bash
# Start with specific agent type
python3 api_server.py --agent-type mcts

# Custom port
python3 api_server.py --port 5001

# Kill existing and restart
python3 api_server.py --port 5000 --kill-existing
```

### TypeScript Development

```bash
# Install Node.js dependencies
cd webapp && npm install

# Development server
npm run dev

# Build for production
npm run build
```

## 📁 Project Structure

```
azul-helper/
├── 🐍 Python Backend
│   ├── api_server.py           # Flask API server
│   ├── start.py               # Unified startup script
│   ├── agents/                # AI agent implementations
│   │   ├── heuristic_agent.py # Rule-based AI
│   │   ├── mcts_agent.py      # Monte Carlo Tree Search
│   │   └── base_agent.py      # Agent interface
│   ├── game/                  # Core game engine
│   │   ├── game_state.py      # Game state management
│   │   ├── player_board.py    # Player board logic
│   │   ├── tile.py           # Tile definitions
│   │   └── factory.py        # Factory mechanics
│   ├── training/              # RL training infrastructure
│   │   ├── trainers/          # Training algorithms
│   │   ├── environments/      # Game environments
│   │   └── networks/          # Neural network models
│   ├── models/                # Saved AI models
│   ├── utils/                 # Utilities and tools
│   └── tests/                 # Test suites
├── 🌐 TypeScript Frontend
│   ├── webapp/                # Web application
│   │   ├── src/
│   │   │   ├── main.ts        # Application entry point
│   │   │   ├── GameState.ts   # TypeScript game state
│   │   │   ├── GameRenderer.ts # Canvas rendering
│   │   │   ├── AI.ts          # Legacy minimax AI
│   │   │   ├── PythonAI.ts    # Python API client
│   │   │   └── types.ts       # Type definitions
│   │   ├── index.html         # Main HTML file
│   │   └── package.json       # Frontend dependencies
│   └── extension/             # Browser extension
│       ├── src/
│       │   ├── background.ts   # Extension background
│       │   ├── content.ts      # BGA content script
│       │   └── popup.tsx       # Extension popup
│       └── manifest.json       # Extension manifest
├── 📊 Configuration & Docs
│   ├── requirements.txt        # Python dependencies
│   ├── requirements-dev.txt    # Development dependencies
│   ├── pyproject.toml         # Python project config
│   ├── README.md              # This file
│   ├── README_RL.md           # RL training guide
│   ├── README_TESTING.md      # Testing guide
│   └── DEVELOPMENT.md         # Development guide
└── 🎯 Static Assets
    └── static/
        ├── imgs/              # Game tile images
        └── bga.html          # BGA test page
```

## 🔌 API Endpoints

The Flask server provides RESTful API endpoints:

### Game Analysis
- `POST /api/get_best_move` - Get AI move recommendation
- `POST /api/evaluate_position` - Get position evaluation
- `GET /api/agent/info` - Get current agent information
- `POST /api/agent/configure` - Configure agent type

### Health & Status
- `GET /api/health` - Server health check
- `GET /api/stats` - Performance statistics

## 🧠 AI Performance

Different agents excel in different scenarios:

| Agent Type | Speed | Strength | Memory | Best For |
|------------|-------|----------|---------|----------|
| Auto | Fast | Excellent | Low | General play |
| MCTS | Medium | Excellent | Medium | Research |
| Heuristic | Very Fast | Good | Very Low | Speed tests |
| Minimax | Fast | Good | Low | Learning |

## 🏃‍♂️ Development Workflow

### Local Development
```bash
# Terminal 1: Start Python API server
python3 start.py --server-only

# Terminal 2: Start frontend development server
cd webapp && npm run dev
```

### Testing
```bash
# Run Python tests
python -m pytest tests/

# Run TypeScript tests
cd webapp && npm test

# Integration tests
python -m pytest tests/integration/
```

### Training New Models
```bash
# Train a new RL agent
python -m training.train_agent --agent-type mcts --episodes 10000

# Benchmark agents
python -m utils.benchmark --agents auto,mcts,heuristic
```

## 🎯 Usage Examples

### Web Application
1. Run `python3 start.py`
2. Open browser to displayed URL
3. Select AI agent type from dropdown
4. Choose difficulty level
5. Play against AI or watch AI vs AI

### Browser Extension
1. Build extension: `cd extension && npm run build`
2. Load in browser as unpacked extension
3. Visit Board Game Arena Azul game
4. Use extension popup for move analysis

### API Integration
```python
import requests

# Get move recommendation
response = requests.post('http://localhost:5000/api/get_best_move', 
                        json={'game_state': game_state_dict})
best_move = response.json()['move']
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Install dependencies: `pip install -r requirements-dev.txt`
3. Run tests: `python -m pytest`
4. Create feature branch
5. Submit pull request

### Research Contributions
- New AI agent implementations
- Improved training algorithms
- Performance optimizations
- Game analysis tools

## 📚 Documentation

- **[Development Guide](DEVELOPMENT.md)**: Detailed development setup
- **[RL Training Guide](README_RL.md)**: Reinforcement learning details
- **[Testing Guide](README_TESTING.md)**: Testing infrastructure
- **[UI Guide](README_UI.md)**: Frontend development

## 🏆 Credits

- **Game Design**: Michael Kiesling (original Azul board game)
- **AI Algorithms**: Multiple academic sources and research papers
- **Implementation**: Modern Python and TypeScript stack
- **Visual Design**: Portuguese azulejo tile work inspiration

## 📄 License

Educational and research use. Azul is a trademark of Plan B Games.

---

🎲 **Happy Playing and Researching!** ✨
