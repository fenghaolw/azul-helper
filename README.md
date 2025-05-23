# Azul - Board Game with AI

A web implementation of the popular board game Azul featuring an advanced AI opponent that uses minimax algorithm with alpha-beta pruning, as described in [Dom Wilson's excellent article](https://domwil.co.uk/posts/azul-ai/).

## Features

- 🎮 **Complete Azul Implementation**: Full game rules and mechanics
- 🤖 **Advanced AI Opponent**: Minimax with alpha-beta pruning and iterative deepening
- 🎨 **Beautiful UI**: Modern, responsive design with intuitive controls
- ⚡ **Real-time Gameplay**: Smooth canvas-based rendering
- 📊 **AI Statistics**: See how many game states the AI evaluates
- 🎛️ **Adjustable Difficulty**: From 0.5 seconds to 5 seconds thinking time

## How to Play

Azul is a tile-laying game where players compete to decorate their palace walls with beautiful Portuguese tiles.

### Basic Rules

1. **Setup**: Each round, factories are filled with 4 random tiles each
2. **Tile Selection**: On your turn, pick all tiles of one color from a factory or the center
3. **Placement**: Place tiles in your pattern lines (1-5 tiles per line)
4. **Scoring**: Complete lines move to your wall and score points
5. **Penalties**: Excess tiles go to your floor line for penalty points
6. **Victory**: Game ends when someone completes a horizontal row

### Scoring System

- **Basic Scoring**: 1 point per tile, plus bonuses for adjacent tiles
- **Row Bonus**: 2 points for each completed horizontal row
- **Column Bonus**: 7 points for each completed vertical column  
- **Color Bonus**: 10 points for each complete set of 5 tiles of one color
- **Floor Penalties**: -1, -1, -2, -2, -2, -3, -3 points for floor tiles

## AI Implementation

The AI uses sophisticated algorithms based on game theory:

### Minimax Algorithm
- Explores possible future game states to find optimal moves
- Assumes both players play optimally
- Evaluates positions using a heuristic function

### Alpha-Beta Pruning
- Dramatically reduces the number of positions to evaluate
- Can search much deeper in the same time
- Typically 16-400x faster than basic minimax

### Iterative Deepening
- Searches progressively deeper until time runs out
- Always has a best move available even if time expires
- Allows for consistent response times

### Move Ordering
- Sorts moves based on previous search results
- Improves alpha-beta pruning efficiency
- Better moves are examined first for more cutoffs

## Technical Stack

- **TypeScript**: Type-safe implementation with modern ES6+ features
- **HTML5 Canvas**: Hardware-accelerated 2D graphics rendering
- **Vite**: Fast development server and build tool
- **Modular Architecture**: Clean separation of game logic, AI, and UI

## Installation & Running

### Prerequisites
- Node.js 16+ installed on your system

### Quick Start

1. **Install dependencies**:
```bash
npm install
```

2. **Start development server**:
```bash
npm run dev
```

3. **Open your browser** and navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Game Controls

- **Click factories or center**: Select tiles of one color
- **Click pattern lines**: Place selected tiles (lines 1-5)
- **Click floor**: Place tiles on floor (penalty points)
- **New Game button**: Start a fresh game
- **AI Toggle**: Enable/disable AI opponent
- **Difficulty slider**: Adjust AI thinking time

## AI Difficulty Levels

- **Easy (0.5s)**: Good for beginners, searches 2-3 moves deep
- **Medium (1s)**: Balanced opponent, searches 3-4 moves deep  
- **Hard (2s)**: Challenging play, searches 4-5 moves deep
- **Expert (5s)**: Very strong opponent, searches 5+ moves deep

## Project Structure

```
src/
├── types.ts          # TypeScript interfaces and enums
├── PlayerBoard.ts    # Individual player board logic
├── GameState.ts      # Main game state and rules
├── AI.ts            # Minimax AI implementation
├── GameRenderer.ts   # Canvas-based UI rendering
└── main.ts          # Application entry point and UI
```

## Implementation Details

### Game State Management
- Immutable game state with deep cloning for AI simulation
- Efficient move generation and validation
- Proper round and game ending conditions

### AI Evaluation Function
- Considers potential score differences between players
- Accounts for completed lines, floor penalties, and tile placement
- Balances immediate gains vs. long-term positioning

### Performance Optimizations
- Alpha-beta pruning with move ordering
- Iterative deepening with time management
- Efficient game state cloning
- Canvas rendering optimizations

## Algorithm Performance

Based on testing with the original implementation:

| Method | Depth 2 | Depth 3 | Depth 4 |
|--------|---------|---------|---------|
| Minimax | 6,552 nodes | 396,552 nodes | Out of memory |
| Alpha-Beta | 393 nodes | 9,401 nodes | 41,275 nodes |
| + Move Ordering | 316 nodes | 7,373 nodes | 36,754 nodes |

The alpha-beta implementation is **16-400x faster** than basic minimax!

## Credits

- **Game Design**: Michael Kiesling (original Azul board game)
- **AI Algorithm**: Based on [Dom Wilson's article](https://domwil.co.uk/posts/azul-ai/)
- **Implementation**: Built with modern web technologies

## License

This project is for educational purposes. Azul is a trademark of Plan B Games.

---

Enjoy playing against the AI! 🎲✨ 