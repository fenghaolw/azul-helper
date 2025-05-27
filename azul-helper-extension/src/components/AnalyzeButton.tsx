import { isAnalyzing } from './App';

interface AnalyzeButtonProps {
  onAnalyze: () => void;
}

export default function AnalyzeButton({ onAnalyze }: AnalyzeButtonProps) {
  return (
    <button
      onClick={onAnalyze}
      disabled={isAnalyzing.value}
      className={`
        md-button w-full py-3 px-6 text-sm text-white font-medium
        ${
          isAnalyzing.value
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
        }
      `}
    >
      {isAnalyzing.value ? 'Analyzing...' : 'Analyze Position & Suggest Move'}
    </button>
  );
}
