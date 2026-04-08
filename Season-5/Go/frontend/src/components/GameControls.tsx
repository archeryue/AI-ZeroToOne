"use client";

interface GameControlsProps {
  onPass: () => void;
  onUndo: () => void;
  onResign: () => void;
  onNewGame: () => void;
  disabled: boolean;
  gameOver: boolean;
}

export default function GameControls({
  onPass, onUndo, onResign, onNewGame, disabled, gameOver,
}: GameControlsProps) {
  const btnBase = "px-4 py-2 rounded font-medium text-sm transition-colors";

  return (
    <div className="flex flex-wrap gap-2">
      <button
        data-testid="btn-pass"
        className={`${btnBase} bg-gray-800 text-white hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed`}
        onClick={onPass}
        disabled={disabled || gameOver}
      >
        Pass
      </button>
      <button
        data-testid="btn-undo"
        className={`${btnBase} bg-gray-200 text-gray-800 hover:bg-gray-300 disabled:opacity-40 disabled:cursor-not-allowed`}
        onClick={onUndo}
        disabled={disabled || gameOver}
      >
        Undo
      </button>
      <button
        data-testid="btn-resign"
        className={`${btnBase} bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-40 disabled:cursor-not-allowed`}
        onClick={onResign}
        disabled={disabled || gameOver}
      >
        Resign
      </button>
      <button
        data-testid="btn-new-game"
        className={`${btnBase} bg-blue-600 text-white hover:bg-blue-500`}
        onClick={onNewGame}
      >
        New Game
      </button>
    </div>
  );
}
