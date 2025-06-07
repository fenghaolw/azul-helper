import { Player } from '../types';

interface GameInfoProps {
    round: number;
    currentPlayerIndex: number;
    players: Player[];
    gamePhase: 'playing' | 'scoring' | 'finished';
}

export function GameInfo({ round, currentPlayerIndex, players, gamePhase }: GameInfoProps) {
    const getGamePhaseText = () => {
        switch (gamePhase) {
            case 'playing':
                return 'Jogando';
            case 'scoring':
                return 'Pontuando';
            case 'finished':
                return 'Jogo Terminado';
            default:
                return 'Aguardando';
        }
    };

    const getGamePhaseClass = () => {
        switch (gamePhase) {
            case 'playing':
                return 'game-info__status--playing';
            case 'scoring':
                return 'game-info__status--playing';
            case 'finished':
                return 'game-info__status--finished';
            default:
                return 'game-info__status--waiting';
        }
    };

    return (
        <div className="game-info">
            <h2 className="game-info__title">Informações do Jogo</h2>

            <div className="game-info__content">
                <div className="game-info__section">
                    <div className="game-info__round">
                        Rodada
                        <span className="round-number">{round}</span>
                    </div>
                </div>

                <div className="game-info__section game-info__current-player">
                    <div className="game-info__section-title">Jogador Atual</div>
                    <div className="game-info__section-content">
                        <span className="player-name">{players[currentPlayerIndex]?.name || 'Desconhecido'}</span>
                        <span className="turn-indicator">É sua vez</span>
                    </div>
                </div>

                <div className="game-info__section game-info__scores">
                    <div className="game-info__section-title">Pontuações</div>
                    <div className="game-info__section-content">
                        <div className="score-list">
                            {players.map((player, index) => (
                                <div
                                    key={index}
                                    className={`score-item ${index === currentPlayerIndex ? 'score-item--current' : ''
                                        }`}
                                >
                                    <span className="player-name">{player.name}</span>
                                    <span className="player-score">{player.score}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                <div className={`game-info__status ${getGamePhaseClass()}`}>
                    {getGamePhaseText()}
                </div>
            </div>
        </div>
    );
}
