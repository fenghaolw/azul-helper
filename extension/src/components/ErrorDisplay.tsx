import { error } from './App';

export default function ErrorDisplay() {
  if (!error.value) {
    return null;
  }

  return (
    <div className="md-card p-4 border-l-4 border-red-500 bg-red-50">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">
          <span className="text-white text-xs font-bold">!</span>
        </div>
        <h4 className="font-semibold text-red-800">Error</h4>
      </div>
      <div className="text-red-700 text-sm">{error.value}</div>
    </div>
  );
}
