import { useState, useEffect } from 'preact/hooks';
import { GameState } from '../types';

interface GameProps {
    gameContainer: HTMLElement;
}

export function Game({ gameContainer }: GameProps) {
    console.log('Game component rendering with gameContainer:', gameContainer);
    const [gameState, setGameState] = useState<GameState | null>(null);

    // Listen for game state updates
    useEffect(() => {
        console.log('Setting up gameStateUpdate listener on:', gameContainer);
        const handleGameStateUpdate = (event: CustomEvent<GameState>) => {
            console.log('Received gameStateUpdate:', event.detail);
            setGameState(event.detail);
        };

        gameContainer.addEventListener('gameStateUpdate', handleGameStateUpdate as EventListener);

        return () => {
            gameContainer.removeEventListener('gameStateUpdate', handleGameStateUpdate as EventListener);
        };
    }, [gameContainer]);

    if (!gameState) {
        console.log('No gameState, rendering loading...');
        return (
            <div className="game-container" style={{ padding: '20px', background: 'white', minHeight: '100vh' }}>
                <div className="loading" style={{ color: 'black', fontSize: '24px' }}>Loading game...</div>
            </div>
        );
    }

    return (
        <div className="game-container" style={{ padding: '20px', background: 'lightblue', minHeight: '100vh' }}>
            <div style={{ color: 'black', fontSize: '18px' }}>
                <h1>Azul Game</h1>
                <p>Round: {gameState.round}</p>
                <p>Current Player: {gameState.currentPlayerIndex}</p>
                <p>Factories: {gameState.factories.length}</p>
                <p>Center Tiles: {gameState.centerTiles.length}</p>
            </div>
        </div>
    );
}
