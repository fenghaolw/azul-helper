import { analysisResult, isAnalyzing } from './App';
import TileIcon from './TileIcon';
import { TILE_ICON_SIZE } from '../constants';
import { Move } from '../types';

export default function MoveSuggestion() {
  const formatMove = (move: Move | null): string => {
    if (!move) return '';
    const source = move.factoryIndex === -1 ? 'center' : `factory ${move.factoryIndex + 1}`;
    const destination = move.lineIndex === -1 ? 'floor' : `pattern line ${move.lineIndex + 1}`;
    return `Take ${move.tile} tiles from ${source} and place them in ${destination}`;
  };

  const getSuggestionText = (): string => {
    if (isAnalyzing.value) return 'Analyzing position...';
    if (analysisResult.value?.error) return 'Error analyzing position';
    if (analysisResult.value?.move) return formatMove(analysisResult.value.move);
    return 'Click "Analyze Position" to get AI suggestion';
  };

  const getStats = () => {
    if (!analysisResult.value || analysisResult.value.error) return null;
    return analysisResult.value.stats;
  };

  return (
    <div className="md-card p-3 sm:p-4 lg:p-5">
      <div className="flex items-center gap-2 mb-3 sm:mb-4">
        <div className="w-1 h-6 bg-blue-500 rounded-full"></div>
        <h3 className="font-semibold text-base text-gray-900">Suggested Move</h3>
      </div>

      <div className="text-sm text-gray-700 mb-3">{getSuggestionText()}</div>

      {analysisResult.value?.move && !analysisResult.value.error && (
        <>
          <div className="flex gap-2 mb-3">
            <TileIcon tile={analysisResult.value.move.tile} size={TILE_ICON_SIZE * 1.5} />
          </div>
        </>
      )}
    </div>
  );
}
