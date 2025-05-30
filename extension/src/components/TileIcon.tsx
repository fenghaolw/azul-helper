interface TileIconProps {
  tile: string;
  size?: number;
  className?: string;
}

const tileSVGs: { [key: string]: string } = {
  red: 'tile-red.svg',
  blue: 'tile-blue.svg',
  yellow: 'tile-yellow.svg',
  black: 'tile-black.svg',
  white: 'tile-turquoise.svg',
  firstplayer: 'tile-overlay-dark.svg',
};

export default function TileIcon({ tile, size = 16, className = '' }: TileIconProps) {
  const tileKey = tile.toLowerCase();
  const svgFile = tileSVGs[tileKey];

  if (!svgFile) {
    console.error(`No SVG file found for tile key: '${tileKey}' (original: '${tile}')`);
  }

  const src = chrome.runtime.getURL(`icons/${svgFile || 'tile-overlay-dark.svg'}`);

  return (
    <img src={src} alt={tile} width={size} height={size} className={`tile-icon ${className}`} />
  );
}
