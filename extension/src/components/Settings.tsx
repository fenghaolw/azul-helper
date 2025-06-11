import { difficulty } from './App';

const difficultyLabels: { [key: number]: string } = {
  1: 'Easy',
  2: 'Medium',
  3: 'Hard',
  4: 'Expert',
};

export default function Settings() {
  const handleDifficultyChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    difficulty.value = parseInt(target.value);
  };

  return (
    <div className="md-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1 h-6 bg-orange-500 rounded-full"></div>
        <h3 className="font-semibold text-base text-gray-900">AI Settings</h3>
      </div>
      <div className="flex items-center gap-3">
        <label htmlFor="difficulty" className="text-sm font-medium text-gray-700">
          Difficulty:
        </label>
        <input
          type="range"
          id="difficulty"
          min="1"
          max="4"
          value={difficulty.value}
          onChange={handleDifficultyChange}
          className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
        />
        <span className="text-sm font-medium text-gray-900 min-w-[60px]">
          {difficultyLabels[difficulty.value]}
        </span>
      </div>
    </div>
  );
}
