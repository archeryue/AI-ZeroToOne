"use client";

import type { GameState } from "@/lib/types";

interface GameStatusProps {
  gameState: GameState;
  connected: boolean;
  aiThinking: boolean;
}

export default function GameStatus({ gameState, connected, aiThinking }: GameStatusProps) {
  const { current_turn, status, captured, komi, score, ai_type, player_color, move_count, resigned_by } = gameState;

  const statusText = () => {
    if (status === "black_win") {
      if (score) {
        return `Black wins by ${(score.black - score.white).toFixed(1)} points`;
      }
      if (resigned_by === "white") {
        return resigned_by !== player_color
          ? "AI resigned — You win!"
          : "Black wins (resignation)";
      }
      return "Black wins (resignation)";
    }
    if (status === "white_win") {
      if (score) {
        return `White wins by ${(score.white - score.black).toFixed(1)} points`;
      }
      if (resigned_by === "black") {
        return resigned_by !== player_color
          ? "AI resigned — You win!"
          : "White wins (resignation)";
      }
      return "White wins (resignation)";
    }
    if (aiThinking) return "AI is thinking...";
    return `${current_turn === "black" ? "Black" : "White"} to play`;
  };

  return (
    <div className="space-y-3">
      {/* Connection status */}
      <div className="flex items-center gap-2 text-sm">
        <span
          className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
        />
        <span className="text-gray-500">
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>

      {/* Game status */}
      <div data-testid="game-status" className={`text-lg font-bold ${
        status !== "playing" ? "text-amber-600" :
        aiThinking ? "text-blue-500 animate-pulse" : ""
      }`}>
        {statusText()}
      </div>

      {/* Game info */}
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">You play</div>
          <div className="font-medium flex items-center gap-1.5">
            <span className={`inline-block w-3 h-3 rounded-full border ${
              player_color === "black" ? "bg-gray-900 border-gray-700" : "bg-white border-gray-400"
            }`} />
            {player_color === "black" ? "Black" : "White"}
          </div>
        </div>
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-500">AI</div>
          <div className="font-medium">{ai_type}</div>
        </div>
      </div>

      {/* Captures */}
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="bg-gray-50 rounded p-2 flex items-center gap-2">
          <span className="w-4 h-4 rounded-full bg-gray-900 border border-gray-700 inline-block" />
          <div>
            <div className="text-gray-500">Captures</div>
            <div className="font-bold">{captured.black}</div>
          </div>
        </div>
        <div className="bg-gray-50 rounded p-2 flex items-center gap-2">
          <span className="w-4 h-4 rounded-full bg-white border border-gray-400 inline-block" />
          <div>
            <div className="text-gray-500">Captures</div>
            <div className="font-bold">{captured.white}</div>
          </div>
        </div>
      </div>

      {/* Score (when game is over) */}
      {score && (
        <div className="bg-amber-50 border border-amber-200 rounded p-3 text-sm">
          <div className="font-bold mb-1">Final Score</div>
          <div className="flex justify-between">
            <span>Black: {score.black}</span>
            <span>White: {score.white} (komi {komi})</span>
          </div>
        </div>
      )}

      {/* Move count */}
      <div className="text-sm text-gray-500">
        Move {move_count} | Komi: {komi}
      </div>
    </div>
  );
}
