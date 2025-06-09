import { useContext, useEffect } from "preact/hooks";
import { route } from "preact-router";
import { GameReplay } from "./GameReplay";
import { ReplayDataContext } from "../main";

export function ReplayView() {
  const { replayData, setReplayData } = useContext(ReplayDataContext);

  // Redirect to home if no replay data is available
  useEffect(() => {
    if (!replayData) {
      route("/");
    }
  }, [replayData]);

  (event: Event) => {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          setReplayData(data);
          // No need to route again as we're already on the replay page
        } catch (error) {
          console.error("❌ Error parsing replay file:", error);
          alert("Invalid replay file format");
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="azul-app__game" id="gameContainer">
      {replayData && <GameReplay replayData={replayData} />}
    </div>
  );
}
