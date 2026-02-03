"use client";

import { GameState } from "@/lib/types";

interface GameStatusProps {
  gameState: GameState;
  connected: boolean;
}

export default function GameStatus({ gameState, connected }: GameStatusProps) {
  const { current_turn, status, in_check, player_color, ai_type } = gameState;

  const statusText = () => {
    if (status === "red_win") return "Red Wins!";
    if (status === "black_win") return "Black Wins!";
    if (status === "draw") return "Draw!";
    if (in_check) return `${current_turn === "red" ? "Red" : "Black"} is in CHECK!`;
    return `${current_turn === "red" ? "Red" : "Black"}'s turn`;
  };

  const isGameOver = status !== "playing";
  const playerWon =
    (status === "red_win" && player_color === "red") ||
    (status === "black_win" && player_color === "black");
  const playerLost =
    (status === "red_win" && player_color === "black") ||
    (status === "black_win" && player_color === "red");

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span
          className={`inline-block w-3 h-3 rounded-full ${
            connected ? "bg-green-500" : "bg-red-500"
          }`}
        />
        <span className="text-sm text-stone-400">
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>

      <div
        className={`text-xl font-bold ${
          isGameOver
            ? playerWon
              ? "text-green-400"
              : playerLost
              ? "text-red-400"
              : "text-yellow-400"
            : in_check
            ? "text-red-400 animate-pulse"
            : "text-stone-100"
        }`}
      >
        {statusText()}
      </div>

      <div className="text-sm text-stone-400 space-y-1">
        <div>
          You: <span className={player_color === "red" ? "text-red-400" : "text-blue-400"}>
            {player_color === "red" ? "Red" : "Black"}
          </span>
        </div>
        <div>Opponent: {ai_type}</div>
      </div>
    </div>
  );
}
