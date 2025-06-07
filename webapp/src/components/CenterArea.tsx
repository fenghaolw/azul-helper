import { CenterTile, TileColor } from '../types';
import { Tile } from './Tile';

interface CenterAreaProps {
    centerTiles: CenterTile[];
    selectedGroup: number | null;
    selectedColor: TileColor | null;
    onCenterClick: (groupIndex: number, color: TileColor) => void;
}

export function CenterArea({ centerTiles, selectedGroup, selectedColor, onCenterClick }: CenterAreaProps) {
    const handleGroupClick = (groupIndex: number, color: TileColor) => {
        onCenterClick(groupIndex, color);
    };

    return (
        <div className="center-area">
            <h2 className="center-area__title">Centro</h2>
            <div className="center-area__content">
                {centerTiles.length === 0 ? (
                    <div className="center-area__empty">Nenhum azulejo no centro</div>
                ) : (
                    centerTiles.map((group, index) => (
                        <div
                            key={`${group.color}-${index}`}
                            className={`center-area__group ${group.count === 0 ? 'center-area__group--empty' : ''
                                } ${selectedGroup === index ? 'center-area__group--selected' : ''
                                }`}
                            onClick={() => handleGroupClick(index, group.color)}
                        >
                            {Array.from({ length: Math.min(group.count, 4) }).map((_, tileIndex) => (
                                <Tile
                                    key={`${group.color}-${index}-${tileIndex}`}
                                    color={group.color}
                                    isSelected={selectedGroup === index && selectedColor === group.color}
                                    className="stacked"
                                />
                            ))}
                            <span className="center-area__count">{group.count}</span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
