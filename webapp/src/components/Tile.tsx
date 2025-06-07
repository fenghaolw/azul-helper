import { TileColor } from '../types';

interface TileProps {
    color: TileColor;
    isSelected?: boolean;
    isValidDrop?: boolean;
    onClick?: () => void;
    className?: string;
}

export function Tile({ color, isSelected, isValidDrop, onClick, className = '' }: TileProps) {
    const classNames = [
        'tile',
        `tile--${color}`,
        isSelected && 'tile--selected',
        isValidDrop && 'tile--valid-drop',
        className
    ].filter(Boolean).join(' ');

    return (
        <div
            className={classNames}
            onClick={onClick}
            role={onClick ? 'button' : undefined}
            tabIndex={onClick ? 0 : undefined}
            onKeyDown={onClick ? (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onClick();
                }
            } : undefined}
        />
    );
}
