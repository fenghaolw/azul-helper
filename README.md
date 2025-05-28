# Azul Helper

A comprehensive Azul board game project featuring both a web application and browser extension with advanced AI capabilities using minimax algorithm with alpha-beta pruning, as described in [Dom Wilson's excellent article](https://domwil.co.uk/posts/azul-ai/).

## Features

### Web Application
- 🎮 **Complete Azul Implementation**: Full game rules and mechanics
- 🤖 **Advanced AI Opponent**: Minimax with alpha-beta pruning and iterative deepening
- 🎨 **Beautiful UI**: Portuguese azulejo-inspired design with ceramic coaster aesthetics
- ⚡ **Real-time Gameplay**: Smooth canvas-based rendering with high-DPI support
- 📊 **AI Statistics**: See how many game states the AI evaluates
- 🎛️ **Adjustable Difficulty**: From 0.5 seconds to 5 seconds thinking time

### Browser Extension
- 🌐 **BGA Integration**: Works with Board Game Arena's Azul implementation
- 🧠 **AI Analysis**: Get move suggestions while playing online
- 📈 **Position Evaluation**: See how the AI evaluates the current position
- ⚙️ **Configurable**: Adjust AI thinking time and analysis depth

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

## UI Features

The game features a beautiful Portuguese azulejo-inspired design:

### Visual Elements
- **Ceramic Coaster Factories**: Elegant circular factory displays with decorative borders
- **Ornate Center Table**: Traditional azulejo patterns and decorative corners
- **Player Boards**: Clean layout with proper spacing and alignment
- **Tile Design**: High-quality SVG tiles with rounded corners and shadows
- **Interactive Elements**: Hover effects and visual feedback for all actions

### Design Details
- **Color Scheme**: Traditional Portuguese tile colors
- **Typography**: Classic serif fonts for titles and modern sans-serif for game info
- **Patterns**: Subtle azulejo-inspired decorative elements
- **Spacing**: Carefully balanced layout with proper visual hierarchy
- **Responsiveness**: High-DPI support for crisp rendering on all displays

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
- **HTML5 Canvas**: Hardware-accelerated 2D graphics rendering with high-DPI support
- **Vite**: Fast development server and build tool
- **Modular Architecture**: Clean separation of game logic, AI, and UI
- **SVG Assets**: High-quality tile images with proper scaling

## Installation & Running

### Prerequisites
- Node.js 16+ installed on your system

### Web Application

1. **Navigate to webapp directory**:
```bash
cd webapp
```

2. **Install dependencies**:
```bash
npm install
```

3. **Start development server**:
```bash
npm run dev
```

4. **Open your browser** and navigate to `http://localhost:3000`

### Browser Extension

1. **Navigate to extension directory**:
```bash
cd extension
```

2. **Install dependencies**:
```bash
npm install
```

3. **Build the extension**:
```bash
npm run build
```

4. **Load the extension in Chrome**:
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select `extension/dist`

### Building for Production

```bash
# Build webapp
cd webapp && npm run build

# Build extension
cd extension && npm run build
```

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
├── webapp/                    # Web application
│   ├── src/
│   │   ├── types.ts          # TypeScript interfaces and enums
│   │   ├── PlayerBoard.ts    # Individual player board logic
│   │   ├── GameState.ts      # Game state with web app and BGA support
│   │   ├── AI.ts            # Minimax AI implementation
│   │   ├── GameRenderer.ts   # Canvas-based UI rendering
│   │   └── main.ts          # Application entry point
│   ├── index.html           # Main HTML file
│   └── package.json         # Webapp dependencies
├── extension/                 # Browser extension
│   ├── src/
│   │   ├── background.ts     # Extension background script
│   │   ├── content.ts        # Content script for BGA
│   │   ├── popup.tsx         # Extension popup UI
│   │   └── manifest.json     # Extension manifest
│   └── package.json         # Extension dependencies
└── static/                   # Shared static assets
    ├── imgs/                # Game tile images
    └── bga.html            # BGA test page
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
- **Visual Design**: Inspired by traditional Portuguese azulejo tile work

## License

This project is for educational purposes. Azul is a trademark of Plan B Games.

---

Enjoy playing against the AI! 🎲✨
