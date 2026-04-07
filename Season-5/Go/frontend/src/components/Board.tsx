"use client";

import { BOARD_PADDING, getCellSize, getStarPoints } from "@/lib/constants";
import type { GameState } from "@/lib/types";

interface BoardProps {
  gameState: GameState;
  onCellClick: (row: number, col: number) => void;
  disabled: boolean;
  lastMove?: { row: number; col: number } | null;
}

export default function Board({ gameState, onCellClick, disabled, lastMove }: BoardProps) {
  const { board, board_size, legal_moves, territory, status } = gameState;
  const cellSize = getCellSize(board_size);
  const boardPixel = BOARD_PADDING * 2 + cellSize * (board_size - 1);
  const stoneRadius = cellSize * 0.45;
  const starPoints = getStarPoints(board_size);

  // Build legal move set for quick lookup
  const legalSet = new Set(legal_moves.map(([r, c]) => `${r},${c}`));

  const toX = (col: number) => BOARD_PADDING + col * cellSize;
  const toY = (row: number) => BOARD_PADDING + row * cellSize;

  return (
    <svg
      width={boardPixel}
      height={boardPixel}
      viewBox={`0 0 ${boardPixel} ${boardPixel}`}
      className="select-none"
    >
      {/* Board background */}
      <rect width={boardPixel} height={boardPixel} fill="#DCB35C" rx={4} />

      {/* Grid lines */}
      {Array.from({ length: board_size }, (_, i) => (
        <g key={`grid-${i}`}>
          {/* Horizontal */}
          <line
            x1={toX(0)} y1={toY(i)}
            x2={toX(board_size - 1)} y2={toY(i)}
            stroke="#4A3728" strokeWidth={i === 0 || i === board_size - 1 ? 1.5 : 0.8}
          />
          {/* Vertical */}
          <line
            x1={toX(i)} y1={toY(0)}
            x2={toX(i)} y2={toY(board_size - 1)}
            stroke="#4A3728" strokeWidth={i === 0 || i === board_size - 1 ? 1.5 : 0.8}
          />
        </g>
      ))}

      {/* Star points */}
      {starPoints.map(([r, c]) => (
        <circle
          key={`star-${r}-${c}`}
          cx={toX(c)} cy={toY(r)}
          r={cellSize * 0.1}
          fill="#4A3728"
        />
      ))}

      {/* Coordinate labels */}
      {Array.from({ length: board_size }, (_, i) => {
        const colLabel = String.fromCharCode(65 + (i >= 8 ? i + 1 : i)); // Skip 'I'
        return (
          <g key={`label-${i}`}>
            <text
              x={toX(i)} y={BOARD_PADDING - 14}
              textAnchor="middle" fontSize={10} fill="#4A3728" fontFamily="sans-serif"
            >
              {colLabel}
            </text>
            <text
              x={BOARD_PADDING - 14} y={toY(i) + 4}
              textAnchor="middle" fontSize={10} fill="#4A3728" fontFamily="sans-serif"
            >
              {board_size - i}
            </text>
          </g>
        );
      })}

      {/* Territory markers (shown when game is over) */}
      {territory && status !== "playing" && territory.map((row, r) =>
        row.map((t, c) => {
          if (t === 0 || board[r][c] !== 0) return null;
          return (
            <rect
              key={`territory-${r}-${c}`}
              x={toX(c) - cellSize * 0.15}
              y={toY(r) - cellSize * 0.15}
              width={cellSize * 0.3}
              height={cellSize * 0.3}
              fill={t === 1 ? "rgba(0,0,0,0.4)" : "rgba(255,255,255,0.6)"}
              stroke={t === 1 ? "#000" : "#888"}
              strokeWidth={0.5}
            />
          );
        })
      )}

      {/* Stones */}
      {board.map((row, r) =>
        row.map((cell, c) => {
          if (cell === 0) return null;
          const isBlack = cell === 1;
          const isLastMove = lastMove && lastMove.row === r && lastMove.col === c;
          return (
            <g key={`stone-${r}-${c}`}>
              {/* Shadow */}
              <circle
                cx={toX(c) + 1.5} cy={toY(r) + 1.5}
                r={stoneRadius}
                fill="rgba(0,0,0,0.25)"
              />
              {/* Stone */}
              <circle
                cx={toX(c)} cy={toY(r)}
                r={stoneRadius}
                fill={isBlack ? "#1a1a1a" : "#f5f5f0"}
                stroke={isBlack ? "#000" : "#aaa"}
                strokeWidth={0.8}
              />
              {/* Gradient for 3D effect */}
              {isBlack ? (
                <circle
                  cx={toX(c) - stoneRadius * 0.25}
                  cy={toY(r) - stoneRadius * 0.25}
                  r={stoneRadius * 0.35}
                  fill="rgba(255,255,255,0.12)"
                />
              ) : (
                <circle
                  cx={toX(c) - stoneRadius * 0.25}
                  cy={toY(r) - stoneRadius * 0.25}
                  r={stoneRadius * 0.4}
                  fill="rgba(255,255,255,0.5)"
                />
              )}
              {/* Last move marker */}
              {isLastMove && (
                <circle
                  cx={toX(c)} cy={toY(r)}
                  r={stoneRadius * 0.3}
                  fill="none"
                  stroke={isBlack ? "#e74c3c" : "#e74c3c"}
                  strokeWidth={2}
                />
              )}
              {/* Dead stone marker (in territory) */}
              {territory && status !== "playing" && territory[r][c] !== 0 && territory[r][c] !== cell && (
                <text
                  x={toX(c)} y={toY(r) + 4}
                  textAnchor="middle" fontSize={stoneRadius}
                  fill={isBlack ? "#fff" : "#000"}
                  fontWeight="bold"
                >
                  x
                </text>
              )}
            </g>
          );
        })
      )}

      {/* Clickable areas for empty intersections */}
      {!disabled && board.map((row, r) =>
        row.map((cell, c) => {
          if (cell !== 0) return null;
          const isLegal = legalSet.has(`${r},${c}`);
          return (
            <circle
              key={`click-${r}-${c}`}
              cx={toX(c)} cy={toY(r)}
              r={cellSize * 0.45}
              fill="transparent"
              className={isLegal ? "cursor-pointer" : "cursor-not-allowed"}
              onClick={() => isLegal && onCellClick(r, c)}
              onMouseEnter={(e) => {
                if (isLegal) {
                  const target = e.currentTarget;
                  target.setAttribute("fill",
                    gameState.current_turn === "black"
                      ? "rgba(0,0,0,0.2)"
                      : "rgba(255,255,255,0.4)"
                  );
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.setAttribute("fill", "transparent");
              }}
            />
          );
        })
      )}
    </svg>
  );
}
