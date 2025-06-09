import { useState, useEffect } from "preact/hooks";

interface ActionLogProps {
  gameState: any;
}

interface ActionLogEntry {
  player: number;
  action: string;
  timestamp: number;
}

export function ActionLog({ gameState }: ActionLogProps) {
  const [logs, setLogs] = useState<ActionLogEntry[]>([]);

  useEffect(() => {
    if (gameState) {
      // Add new log entry when game state changes
      const newLog: ActionLogEntry = {
        player: gameState.currentPlayer,
        action: gameState.lastAction || "Game started",
        timestamp: Date.now(),
      };
      setLogs((prevLogs) => [...prevLogs, newLog]);
    }
  }, [gameState]);

  return (
    <div className="action-log">
      <div className="action-log__content">
        {logs.map((log, index) => (
          <div key={index} className="action-log__entry">
            <div className="action-log__timestamp">
              {new Date(log.timestamp).toLocaleTimeString()}
            </div>
            <div className="action-log__details">
              <span className="action-log__player">
                Player {log.player + 1}
              </span>
              <span className="action-log__separator">•</span>
              <span className="action-log__action">{log.action}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
