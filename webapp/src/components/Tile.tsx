import { TileColor } from "../types";

interface TileProps {
  color: TileColor;
  isSelected?: boolean;
  isValidDrop?: boolean;
  onClick?: (e?: MouseEvent) => void;
  className?: string;
  style?: any;
}

export function Tile({
  color,
  isSelected,
  isValidDrop,
  onClick,
  className = "",
  style,
}: TileProps) {
  const getTileImagePath = (color: TileColor): string => {
    const basePath = "/imgs/";
    switch (color) {
      case "red":
        return `${basePath}tile-red.svg`;
      case "blue":
        return `${basePath}tile-blue.svg`;
      case "yellow":
        return `${basePath}tile-yellow.svg`;
      case "black":
        return `${basePath}tile-black.svg`;
      case "white":
        return `${basePath}tile-turquoise.svg`; // Using turquoise for white
      case "first-player":
        return `${basePath}tile-overlay-dark.svg`; // Special tile for first player
      default:
        return `${basePath}tile-turquoise.svg`;
    }
  };

  const classNames = [
    "tile",
    `tile--${color}`,
    isSelected && "tile--selected",
    isValidDrop && "tile--valid-drop",
    onClick && "tile--clickable",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const tileStyle = {
    backgroundImage: `url("${getTileImagePath(color)}")`,
    backgroundSize: "contain",
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
    backgroundColor: "transparent", // Don't use CSS background colors when using SVG images
    cursor: onClick ? "pointer" : "default",
    ...style,
  };

  return (
    <div
      className={classNames}
      style={tileStyle}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
    />
  );
}
