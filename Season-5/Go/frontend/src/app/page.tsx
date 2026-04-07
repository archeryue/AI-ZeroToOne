"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Board from "@/components/Board";
import GameStatus from "@/components/GameStatus";
import GameControls from "@/components/GameControls";
import MoveHistory from "@/components/MoveHistory";
import { createGame } from "@/lib/api";
import { useGameSocket } from "@/hooks/useGameSocket";
import type { GameState } from "@/lib/types";

export default function Home() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [playerColor, setPlayerColor] = useState<"black" | "white">("black");
  const [aiType, setAiType] = useState("greedy");
  const [boardSize, setBoardSize] = useState(9);
  const [mctsSims, setMctsSims] = useState(500);
  const [lastMove, setLastMove] = useState<{ row: number; col: number } | null>(null);

  const {
    gameState, setGameState, connected, aiThinking, error,
    sendMove, sendPass, sendUndo, sendResign,
  } = useGameSocket(gameId);

  // Track last move from move history
  useEffect(() => {
    if (gameState && gameState.move_history.length > 0) {
      const last = gameState.move_history[gameState.move_history.length - 1];
      if (last.position !== "pass") {
        const [r, c] = last.position.split(",").map(Number);
        setLastMove({ row: r, col: c });
      } else {
        setLastMove(null);
      }
    } else {
      setLastMove(null);
    }
  }, [gameState?.move_count]);

  const handleNewGame = useCallback(async () => {
    try {
      const res = await createGame({
        player_color: playerColor,
        ai_type: aiType,
        board_size: boardSize,
        mcts_sims: mctsSims,
      });
      setGameId(res.game_id);
      setGameState(res.game_state);
      setLastMove(null);
    } catch (e) {
      console.error("Failed to create game:", e);
    }
  }, [playerColor, aiType, boardSize, mctsSims, setGameState]);

  const handleCellClick = useCallback(
    (row: number, col: number) => {
      if (!gameState || gameState.status !== "playing" || aiThinking) return;
      // Check it's player's turn
      const isPlayerTurn = gameState.current_turn === gameState.player_color;
      if (!isPlayerTurn) return;
      sendMove(row, col);
    },
    [gameState, aiThinking, sendMove]
  );

  const handlePass = useCallback(() => {
    if (!gameState || gameState.status !== "playing" || aiThinking) return;
    sendPass();
  }, [gameState, aiThinking, sendPass]);

  // ---------- Setup Screen ----------
  if (!gameId || !gameState) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-amber-50 to-orange-50">
        <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md space-y-6">
          <h1 className="text-3xl font-bold text-center">
            围棋 <span className="text-lg text-gray-500">Go</span>
          </h1>

          {/* Board Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Board Size</label>
            <div className="flex gap-2">
              {[9, 13, 19].map((s) => (
                <button
                  key={s}
                  className={`flex-1 py-2 rounded-lg font-medium text-sm border-2 transition-colors ${
                    boardSize === s
                      ? "border-amber-500 bg-amber-50 text-amber-700"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => setBoardSize(s)}
                >
                  {s}x{s}
                </button>
              ))}
            </div>
          </div>

          {/* Player Color */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Play as</label>
            <div className="flex gap-2">
              {(["black", "white"] as const).map((c) => (
                <button
                  key={c}
                  className={`flex-1 py-2 rounded-lg font-medium text-sm border-2 transition-colors flex items-center justify-center gap-2 ${
                    playerColor === c
                      ? "border-amber-500 bg-amber-50 text-amber-700"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => setPlayerColor(c)}
                >
                  <span
                    className={`w-4 h-4 rounded-full border ${
                      c === "black" ? "bg-gray-900 border-gray-700" : "bg-white border-gray-400"
                    }`}
                  />
                  {c === "black" ? "Black (first)" : "White (second)"}
                </button>
              ))}
            </div>
          </div>

          {/* AI Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Opponent</label>
            <div className="space-y-1">
              {[
                { value: "random", label: "Random", desc: "Picks random legal moves" },
                { value: "greedy", label: "Greedy", desc: "Captures & territory heuristic" },
                { value: "mcts", label: "MCTS", desc: "Monte Carlo Tree Search" },
                { value: "human", label: "Human", desc: "Two-player mode" },
              ].map(({ value, label, desc }) => (
                <button
                  key={value}
                  className={`w-full text-left px-3 py-2 rounded-lg border-2 transition-colors ${
                    aiType === value
                      ? "border-amber-500 bg-amber-50"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => setAiType(value)}
                >
                  <div className="font-medium text-sm">{label}</div>
                  <div className="text-xs text-gray-500">{desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* MCTS simulations */}
          {aiType === "mcts" && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                MCTS Simulations: {mctsSims}
              </label>
              <input
                type="range"
                min={100}
                max={2000}
                step={100}
                value={mctsSims}
                onChange={(e) => setMctsSims(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400">
                <span>100 (fast)</span>
                <span>2000 (strong)</span>
              </div>
            </div>
          )}

          {/* Start button */}
          <button
            className="w-full py-3 bg-gray-900 text-white rounded-lg font-bold text-lg hover:bg-gray-800 transition-colors"
            onClick={handleNewGame}
          >
            Start Game
          </button>
        </div>
      </div>
    );
  }

  // ---------- Game Screen ----------
  const isPlayerTurn = gameState.current_turn === gameState.player_color;
  const isGameOver = gameState.status !== "playing";

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 to-orange-50 p-4">
      <div className="max-w-5xl mx-auto flex flex-col lg:flex-row gap-6 items-start">
        {/* Board */}
        <div className="flex-shrink-0">
          <Board
            gameState={gameState}
            onCellClick={handleCellClick}
            disabled={!isPlayerTurn || aiThinking || isGameOver}
            lastMove={lastMove}
          />
        </div>

        {/* Side Panel */}
        <div className="w-full lg:w-72 space-y-4">
          <div className="bg-white rounded-xl shadow p-4">
            <h2 className="font-bold text-lg mb-3">
              围棋 <span className="text-gray-400 text-sm">Go {gameState.board_size}x{gameState.board_size}</span>
            </h2>
            <GameStatus
              gameState={gameState}
              connected={connected}
              aiThinking={aiThinking}
            />
          </div>

          <div className="bg-white rounded-xl shadow p-4">
            <GameControls
              onPass={handlePass}
              onUndo={sendUndo}
              onResign={sendResign}
              onNewGame={() => { setGameId(null); setLastMove(null); }}
              disabled={!isPlayerTurn || aiThinking}
              gameOver={isGameOver}
            />
          </div>

          <div className="bg-white rounded-xl shadow p-4">
            <h3 className="font-bold text-sm mb-2">Move History</h3>
            <MoveHistory moves={gameState.move_history} boardSize={gameState.board_size} />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-red-700 text-sm">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
