import { useEffect, useState } from "preact/hooks";
import { Sidebar } from "./Sidebar";

interface AISettingsProps {
  aiEnabled: boolean;
  onToggleAI: () => void;
  onNewGame: () => void;
  round?: number;
  onSwitchToReplay?: () => void;
}

interface AIStats {
  totalMoves: number;
  nodesEvaluated: number;
  searchTime: number;
  averageSearchTime: number;
  lastMoveTime?: Date;
}

interface ServerInfo {
  agentType?: string;
  agentName?: string;
  version?: string;
  port?: number;
}

export function AISettings({
  aiEnabled,
  onToggleAI,
  onNewGame,
  round = 1,
  onSwitchToReplay,
}: AISettingsProps) {
  const [aiStats, setAiStats] = useState<AIStats | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);
  const [serverPort, setServerPort] = useState<number>(5000);

  // Check actual server connection status
  useEffect(() => {
    const checkConnection = async () => {
      if (!aiEnabled) {
        setAiStats(null);
        setIsConnected(false);
        return;
      }

      try {
        // Try ports 5000-5009
        for (let port = 5000; port < 5010; port++) {
          try {
            const testUrl = `http://localhost:${port}`;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1000); // 1 second timeout per port

            const response = await Promise.race([
              fetch(`${testUrl}/health`, {
                method: "GET",
                signal: controller.signal,
              }),
              new Promise<never>((_, reject) => {
                setTimeout(() => reject(new Error("Timeout")), 1000);
              }),
            ]);

            clearTimeout(timeoutId);

            if (response.ok) {
              const healthData = await response.json();
              if (healthData.status === "healthy") {
                setIsConnected(true);
                setServerInfo(healthData);
                setServerPort(port);
                return;
              }
            }
          } catch (error) {
            // Port not responding, continue to next port
            continue;
          }
        }

        // No server found
        setIsConnected(false);
        setAiStats(null);
        setServerInfo(null);
      } catch (error) {
        // All ports failed
        setIsConnected(false);
        setAiStats(null);
        setServerInfo(null);
      }
    };

    // Only check connection when component mounts or AI is enabled/disabled
    checkConnection();
  }, [aiEnabled]);

  const formatTime = (seconds: number): string => {
    return `${(seconds * 1000).toFixed(1)}ms`;
  };

  const getPerformanceIndicator = (avgTime: number) => {
    if (avgTime < 0.1)
      return { icon: "⚡", text: "Lightning Fast", color: "#4caf50" };
    if (avgTime < 0.5) return { icon: "🚀", text: "Fast", color: "#2196f3" };
    if (avgTime < 2.0) return { icon: "🏃", text: "Normal", color: "#ff9800" };
    return { icon: "🐌", text: "Slow", color: "#ff9800" };
  };

  return (
    <Sidebar title="Game Controls" subtitle={`Round ${round}`}>
      <div className="ai-settings">
        <div className="ai-settings__controls">
          <button className="button button--primary" onClick={onNewGame}>
            New Game
          </button>

          <button
            className={`button ${aiEnabled ? "button--danger" : "button--success"}`}
            onClick={onToggleAI}
          >
            {aiEnabled ? "Disable AI" : "Enable AI"}
          </button>

          {onSwitchToReplay && (
            <button
              className="button button--primary"
              onClick={onSwitchToReplay}
            >
              Load Replay
            </button>
          )}
        </div>

        <div className="ai-settings__section">
          <div className="section-header">
            <div className="section-header__accent section-header__accent--purple"></div>
            <h3 className="section-header__title">AI Status</h3>
          </div>

          {!aiEnabled ? (
            <div className="ai-settings__status-text">
              🤖 AI is disabled - Human vs Human mode
            </div>
          ) : (
            <div className="ai-settings__status">
              <div
                className={`ai-settings__status-indicator ${isConnected ? "ai-settings__status-indicator--connected" : "ai-settings__status-indicator--disconnected"}`}
              >
                <strong>Status:</strong>{" "}
                {isConnected ? "🚀 Connected" : "❌ Disconnected"}
              </div>

              {isConnected && (
                <div className="ai-settings__info">
                  <strong>AI Agent:</strong> 🤖{" "}
                  {serverInfo?.agentName ||
                    serverInfo?.agentType ||
                    "Unknown Agent"}
                </div>
              )}

              {isConnected && (
                <div className="ai-settings__info">
                  <strong>Server:</strong> localhost:{serverPort}
                </div>
              )}

              {!isConnected && (
                <div className="ai-settings__info">
                  💡 Try: python start.py --server-only
                </div>
              )}
            </div>
          )}
        </div>

        {aiEnabled && aiStats && isConnected && (
          <div className="ai-settings__section">
            <div className="section-header">
              <div className="section-header__accent section-header__accent--green"></div>
              <h3 className="section-header__title">Performance Statistics</h3>
            </div>

            <div className="ai-settings__stats">
              <div className="ai-settings__stat">
                <strong>Last Search:</strong>{" "}
                {aiStats.nodesEvaluated.toLocaleString()} nodes
              </div>

              <div className="ai-settings__stat">
                <strong>Last Time:</strong> {formatTime(aiStats.searchTime)}
              </div>

              <div className="ai-settings__stat">
                <strong>Avg Time:</strong>{" "}
                {formatTime(aiStats.averageSearchTime)}
              </div>

              <div className="ai-settings__stat">
                <strong>Moves Made:</strong> {aiStats.totalMoves}
              </div>

              {aiStats.lastMoveTime && (
                <div className="ai-settings__stat">
                  <strong>Last Move:</strong>{" "}
                  {Math.floor(
                    (Date.now() - aiStats.lastMoveTime.getTime()) / 1000,
                  )}
                  s ago
                </div>
              )}

              {(() => {
                const perf = getPerformanceIndicator(aiStats.averageSearchTime);
                return (
                  <div
                    className="ai-settings__stat ai-settings__stat--performance"
                    style={{ color: perf.color }}
                  >
                    {perf.icon} <strong>Speed:</strong> {perf.text}
                  </div>
                );
              })()}
            </div>
          </div>
        )}

        {aiEnabled && isConnected && (
          <div className="ai-settings__section">
            <div className="section-header">
              <div className="section-header__accent section-header__accent--orange"></div>
              <h3 className="section-header__title">Configuration</h3>
            </div>
            <div className="ai-settings__config">
              <div className="ai-settings__config-item">
                💡 AI difficulty controlled by server configuration
              </div>
              <div className="ai-settings__config-item">
                🔧 Restart server with different flags to change settings
              </div>
            </div>
          </div>
        )}
      </div>
    </Sidebar>
  );
}
