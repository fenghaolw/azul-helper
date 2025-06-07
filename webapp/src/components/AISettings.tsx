import { useEffect, useState } from 'preact/hooks';

interface AISettingsProps {
  aiEnabled: boolean;
  onToggleAI: () => void;
  onNewGame: () => void;
  round?: number;
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
}

export function AISettings({
  aiEnabled,
  onToggleAI,
  onNewGame,
  round = 1,
}: AISettingsProps) {
  const [aiStats, setAiStats] = useState<AIStats | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);

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
                method: 'GET',
                signal: controller.signal,
              }),
              new Promise<never>((_, reject) => {
                setTimeout(() => reject(new Error('Timeout')), 1000);
              }),
            ]);

            clearTimeout(timeoutId);

            if (response.ok) {
              const healthData = await response.json();
              if (healthData.status === 'healthy') {
                setIsConnected(true);
                setServerInfo(healthData);
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

    checkConnection();
    const interval = setInterval(checkConnection, 3000); // Check every 3 seconds

    return () => clearInterval(interval);
  }, [aiEnabled]);

  const formatTime = (seconds: number): string => {
    return `${(seconds * 1000).toFixed(1)}ms`;
  };

  const getPerformanceIndicator = (avgTime: number) => {
    if (avgTime < 0.1)
      return { icon: '⚡', text: 'Lightning Fast', color: '#4caf50' };
    if (avgTime < 0.5) return { icon: '🚀', text: 'Fast', color: '#2196f3' };
    if (avgTime < 2.0) return { icon: '🏃', text: 'Normal', color: '#ff9800' };
    return { icon: '🐌', text: 'Slow', color: '#ff9800' };
  };

  return (
    <div className="ai-settings">
      <div className="ai-settings__header">
        <h2>Game Controls</h2>
        <div className="ai-settings__round">Round {round}</div>
      </div>

      <div className="ai-settings__controls">
        <button
          className="ai-settings__button ai-settings__button--primary"
          onClick={onNewGame}
        >
          New Game
        </button>

        <button
          className={`ai-settings__button ${aiEnabled
              ? 'ai-settings__button--danger'
              : 'ai-settings__button--success'
            }`}
          onClick={onToggleAI}
        >
          {aiEnabled ? 'Disable AI' : 'Enable AI'}
        </button>
      </div>

      <div className="ai-settings__status">
        <h3>AI Status</h3>

        {!aiEnabled ? (
          <div className="ai-settings__disabled">
            🤖 AI is disabled - Human vs Human mode
          </div>
        ) : (
          <div className="ai-settings__enabled">
            <div
              className={`ai-settings__connection ${isConnected ? 'connected' : 'disconnected'}`}
            >
              <strong>Status:</strong>{' '}
              {isConnected ? '🚀 Connected' : '❌ Disconnected'}
            </div>

            {isConnected && (
              <div className="ai-settings__agent">
                <strong>AI Agent:</strong> 🤖 {serverInfo?.agentName || serverInfo?.agentType || 'Unknown Agent'}
              </div>
            )}

            {isConnected && (
              <div className="ai-settings__server">
                <strong>Server:</strong> localhost:5000
              </div>
            )}

            {!isConnected && (
              <div className="ai-settings__help">
                💡 Try: python start.py --server-only
              </div>
            )}
          </div>
        )}
      </div>

      {aiEnabled && aiStats && isConnected && (
        <div className="ai-settings__stats">
          <h3>📊 Performance Statistics</h3>

          <div className="ai-settings__stat">
            <strong>Last Search:</strong>{' '}
            {aiStats.nodesEvaluated.toLocaleString()} nodes
          </div>

          <div className="ai-settings__stat">
            <strong>Last Time:</strong> {formatTime(aiStats.searchTime)}
          </div>

          <div className="ai-settings__stat">
            <strong>Avg Time:</strong> {formatTime(aiStats.averageSearchTime)}
          </div>

          <div className="ai-settings__stat">
            <strong>Moves Made:</strong> {aiStats.totalMoves}
          </div>

          {aiStats.lastMoveTime && (
            <div className="ai-settings__stat">
              <strong>Last Move:</strong>{' '}
              {Math.floor((Date.now() - aiStats.lastMoveTime.getTime()) / 1000)}
              s ago
            </div>
          )}

          {(() => {
            const perf = getPerformanceIndicator(aiStats.averageSearchTime);
            return (
              <div
                className="ai-settings__performance"
                style={{ color: perf.color }}
              >
                {perf.icon} <strong>Speed:</strong> {perf.text}
              </div>
            );
          })()}
        </div>
      )}

      {aiEnabled && isConnected && (
        <div className="ai-settings__config">
          <h3>⚙️ Configuration</h3>
          <div className="ai-settings__config-note">
            💡 AI difficulty controlled by server configuration
          </div>
          <div className="ai-settings__config-note">
            🔧 Restart server with different flags to change settings
          </div>
        </div>
      )}
    </div>
  );
}
