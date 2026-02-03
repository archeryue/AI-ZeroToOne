"use client";

interface GameControlsProps {
  onNewGame: () => void;
  onUndo: () => void;
  onResign: () => void;
  gameOver: boolean;
}

export default function GameControls({
  onNewGame,
  onUndo,
  onResign,
  gameOver,
}: GameControlsProps) {
  return (
    <div className="flex gap-3">
      <button
        onClick={onNewGame}
        className="px-4 py-2 bg-amber-700 hover:bg-amber-800 text-white rounded-lg font-medium transition-colors"
      >
        New Game
      </button>
      <button
        onClick={onUndo}
        disabled={gameOver}
        className="px-4 py-2 bg-stone-600 hover:bg-stone-700 text-white rounded-lg font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Undo
      </button>
      <button
        onClick={onResign}
        disabled={gameOver}
        className="px-4 py-2 bg-red-700 hover:bg-red-800 text-white rounded-lg font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Resign
      </button>
    </div>
  );
}
