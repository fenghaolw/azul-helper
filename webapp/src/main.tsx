// filepath: /Users/fenghaolw/Documents/azul-helper/webapp/src/main.tsx
import "./styles/main.scss";
import { render } from "preact";
import { useState } from "preact/hooks";
import Router, { Route } from "preact-router";
import { createContext } from "preact";
import { GameView } from "./components/GameView";
import { ReplayView } from "./components/ReplayView";

// Create a context for sharing replay data between routes
interface ReplayDataContextType {
  replayData: any;
  setReplayData: (data: any) => void;
}

export const ReplayDataContext = createContext<ReplayDataContextType>({
  replayData: null,
  setReplayData: () => {},
});

function App() {
  const [aiEnabled, setAiEnabled] = useState(true);
  const [replayData, setReplayData] = useState<any>(null);

  const handleToggleAI = () => {
    setAiEnabled(!aiEnabled);
  };

  return (
    <ReplayDataContext.Provider value={{ replayData, setReplayData }}>
      <div className="azul-app">
        <Router>
          <Route
            path="/"
            component={GameView}
            aiEnabled={aiEnabled}
            onToggleAI={handleToggleAI}
          />
          <Route path="/replay" component={ReplayView} />
        </Router>
      </div>
    </ReplayDataContext.Provider>
  );
}

// Initialize the app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  console.log("Starting Azul app...");
  const container = document.getElementById("app");
  if (container) {
    render(<App />, container);
    console.log("Azul app rendered successfully");
  }
});
