"use client";

import { useCallback, useState } from "react";
import Board from "@/components/Board";
import GameControls from "@/components/GameControls";
import GameStatus from "@/components/GameStatus";
import MoveHistory from "@/components/MoveHistory";
import { useGameSocket } from "@/hooks/useGameSocket";
import { createGame } from "@/lib/api";
import { CreateGameOptions, GameState } from "@/lib/types";

export default function Home() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [selectedPos, setSelectedPos] = useState<[number, number] | null>(null);
  const [aiType, setAiType] = useState<string>("greedy");
  const [playerColor, setPlayerColor] = useState<"red" | "black">("red");
  const [aiDepth, setAiDepth] = useState(3);

  const { gameState, connected, aiThinking, error, sendMove, sendUndo, sendResign } =
    useGameSocket(gameId);

  const handleNewGame = useCallback(async () => {
    try {
      const options: CreateGameOptions = {
        player_color: playerColor,
        ai_type: aiType as CreateGameOptions["ai_type"],
        ai_depth: aiDepth,
      };
      const result = await createGame(options);
      setGameId(result.game_id);
      setSelectedPos(null);
    } catch (e) {
      console.error("Failed to create game:", e);
    }
  }, [playerColor, aiType, aiDepth]);

  const handleCellClick = useCallback(
    (row: number, col: number) => {
      if (!gameState || gameState.status !== "playing") return;

      const piece = gameState.board[row][col];
      const isPlayerTurn =
        (gameState.current_turn === "red" && gameState.player_color === "red") ||
        (gameState.current_turn === "black" && gameState.player_color === "black");

      if (!isPlayerTurn) return;

      if (selectedPos) {
        // Check if clicking on valid destination
        const key = `${selectedPos[0]},${selectedPos[1]}`;
        const validDests = gameState.valid_moves[key] || [];
        const dest = `${row},${col}`;

        if (validDests.includes(dest)) {
          sendMove(selectedPos[0], selectedPos[1], row, col);
          setSelectedPos(null);
          return;
        }

        // Check if clicking on own piece to reselect
        const isOwnPiece =
          (gameState.current_turn === "red" && piece > 0) ||
          (gameState.current_turn === "black" && piece < 0);

        if (isOwnPiece) {
          setSelectedPos([row, col]);
          return;
        }

        // Deselect
        setSelectedPos(null);
      } else {
        // Select own piece
        const isOwnPiece =
          (gameState.current_turn === "red" && piece > 0) ||
          (gameState.current_turn === "black" && piece < 0);

        if (isOwnPiece) {
          setSelectedPos([row, col]);
        }
      }
    },
    [gameState, selectedPos, sendMove]
  );

  const handleUndo = useCallback(() => {
    sendUndo();
    setSelectedPos(null);
  }, [sendUndo]);

  const handleResign = useCallback(() => {
    if (confirm("Are you sure you want to resign?")) {
      sendResign();
    }
  }, [sendResign]);

  // Landing / setup screen
  if (!gameId || !gameState) {
    return (
      <div className="min-h-screen bg-stone-900 flex items-center justify-center">
        <div className="bg-stone-800 p-8 rounded-xl shadow-2xl max-w-md w-full">
          <h1 className="text-3xl font-bold text-stone-100 mb-2 text-center">
            Chinese Chess
          </h1>
          <p className="text-stone-400 text-center mb-6">象棋</p>

          <div className="space-y-4">
            <div>
              <label className="block text-stone-300 text-sm font-medium mb-1">
                Play as
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => setPlayerColor("red")}
                  className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                    playerColor === "red"
                      ? "bg-red-700 text-white"
                      : "bg-stone-700 text-stone-300 hover:bg-stone-600"
                  }`}
                >
                  Red (先手)
                </button>
                <button
                  onClick={() => setPlayerColor("black")}
                  className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                    playerColor === "black"
                      ? "bg-blue-700 text-white"
                      : "bg-stone-700 text-stone-300 hover:bg-stone-600"
                  }`}
                >
                  Black (后手)
                </button>
              </div>
            </div>

            <div>
              <label className="block text-stone-300 text-sm font-medium mb-1">
                AI Opponent
              </label>
              <select
                value={aiType}
                onChange={(e) => setAiType(e.target.value)}
                className="w-full bg-stone-700 text-stone-100 rounded-lg px-3 py-2 border border-stone-600 focus:outline-none focus:border-amber-600"
              >
                <option value="random">Random</option>
                <option value="greedy">Greedy (Material)</option>
                <option value="minimax">Minimax (Alpha-Beta)</option>
                <option value="human">Human (2-player)</option>
              </select>
            </div>

            {aiType === "minimax" && (
              <div>
                <label className="block text-stone-300 text-sm font-medium mb-1">
                  Search Depth: {aiDepth}
                </label>
                <input
                  type="range"
                  min={1}
                  max={4}
                  value={aiDepth}
                  onChange={(e) => setAiDepth(Number(e.target.value))}
                  className="w-full accent-amber-600"
                />
                <div className="flex justify-between text-xs text-stone-500">
                  <span>Easy (1)</span>
                  <span>Hard (4)</span>
                </div>
              </div>
            )}

            <button
              onClick={handleNewGame}
              className="w-full py-3 bg-amber-700 hover:bg-amber-600 text-white rounded-lg font-bold text-lg transition-colors"
            >
              Start Game
            </button>
          </div>
        </div>
      </div>
    );
  }

  const gameOver = gameState.status !== "playing";

  return (
    <div className="min-h-screen bg-stone-900 flex items-center justify-center p-4">
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Board */}
        <div className="bg-stone-800 p-4 rounded-xl shadow-2xl">
          <Board
            gameState={gameState}
            selectedPos={selectedPos}
            onCellClick={handleCellClick}
          />
        </div>

        {/* Side panel */}
        <div className="bg-stone-800 p-6 rounded-xl shadow-2xl w-full lg:w-72 space-y-6">
          <GameStatus gameState={gameState} connected={connected} aiThinking={aiThinking} />
          <GameControls
            onNewGame={() => {
              setGameId(null);
              setSelectedPos(null);
            }}
            onUndo={handleUndo}
            onResign={handleResign}
            gameOver={gameOver}
          />
          <div>
            <h3 className="text-stone-300 font-medium mb-2">Move History</h3>
            <MoveHistory moves={gameState.move_history} board={gameState.board} />
          </div>
          {error && (
            <div className="text-red-400 text-sm bg-red-900/30 p-2 rounded">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
