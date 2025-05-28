# Azul Helper Extension

A Chrome extension that provides AI-powered move suggestions for the Azul board game on Board Game Arena.

## Tech Stack

- **Preact** - Lightweight React alternative for the UI
- **TypeScript** - Type safety and better development experience
- **Tailwind CSS** - Utility-first CSS framework with Material Design styling
- **Vite** - Fast build tool and development server
- **Chrome Extension Manifest V3** - Latest extension format
- **Material Design** - Modern, clean UI following Google's design principles

## Development

### Prerequisites

- Node.js (v16 or higher)
- npm

### Setup

1. Install dependencies:
```bash
npm install
```

2. Build the extension:
```bash
npm run build
```

3. Load the extension in Chrome:
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select the `dist` folder

### Development Workflow

1. Make changes to the source code in the `src/` directory
2. Run `npm run build` to rebuild the extension
3. Reload the extension in Chrome (click the refresh icon on the extension card)
4. Test your changes on Board Game Arena

### Project Structure

```
src/
├── components/          # Preact components
│   ├── App.tsx         # Main app component
│   ├── AnalyzeButton.tsx
│   ├── MoveSuggestion.tsx
│   ├── GameState.tsx
│   ├── PlayerBoards.tsx
│   ├── Settings.tsx
│   ├── ErrorDisplay.tsx
│   └── TileIcon.tsx
├── popup.tsx           # Entry point for the side panel
├── popup.html          # HTML template
├── content.ts          # Content script for BGA integration
├── background.ts       # Service worker
├── manifest.json       # Extension manifest
├── styles.css          # Tailwind CSS styles
└── types.ts           # TypeScript type definitions
```

## Features

- **Material Design Interface** - Clean, modern UI following Google's design principles
- **Side Panel Integration** - Seamlessly integrates with Chrome's side panel
- **Automatic Game State Detection** - Extracts game state from Board Game Arena automatically
- **AI Move Suggestions** - Provides optimal move recommendations with configurable difficulty
- **Real-time Updates** - Updates game state as the game progresses
- **Responsive Design** - Optimized for side panel with proper spacing and typography
- **Visual Hierarchy** - Clear information organization with cards, colors, and shadows

## Usage

1. Navigate to an Azul game on Board Game Arena
2. Click the extension icon to open the side panel
3. The extension will automatically extract the current game state
4. Click "Analyze Position & Suggest Move" to get AI recommendations
5. Adjust the AI difficulty slider as needed

## Building for Production

```bash
npm run build
```

The built extension will be in the `dist/` directory, ready to be packaged or loaded into Chrome.
