"use client";

import React from "react";
import { ROWS, COLS, CELL_SIZE, BOARD_PADDING, BOARD_WIDTH, BOARD_HEIGHT } from "@/lib/constants";
import { GameState } from "@/lib/types";
import Piece from "./Piece";

interface BoardProps {
  gameState: GameState;
  selectedPos: [number, number] | null;
  onCellClick: (row: number, col: number) => void;
}

function toX(col: number) {
  return BOARD_PADDING + col * CELL_SIZE;
}

function toY(row: number) {
  return BOARD_PADDING + row * CELL_SIZE;
}

function GridLines() {
  const lines: React.ReactElement[] = [];

  // Horizontal lines
  for (let r = 0; r < ROWS; r++) {
    lines.push(
      <line
        key={`h-${r}`}
        x1={toX(0)}
        y1={toY(r)}
        x2={toX(COLS - 1)}
        y2={toY(r)}
        stroke="#5D4037"
        strokeWidth={1}
      />
    );
  }

  // Vertical lines (top half)
  for (let c = 0; c < COLS; c++) {
    lines.push(
      <line
        key={`vt-${c}`}
        x1={toX(c)}
        y1={toY(0)}
        x2={toX(c)}
        y2={toY(4)}
        stroke="#5D4037"
        strokeWidth={1}
      />
    );
  }

  // Vertical lines (bottom half)
  for (let c = 0; c < COLS; c++) {
    lines.push(
      <line
        key={`vb-${c}`}
        x1={toX(c)}
        y1={toY(5)}
        x2={toX(c)}
        y2={toY(9)}
        stroke="#5D4037"
        strokeWidth={1}
      />
    );
  }

  // Left and right border through the river
  lines.push(
    <line
      key="border-l"
      x1={toX(0)}
      y1={toY(4)}
      x2={toX(0)}
      y2={toY(5)}
      stroke="#5D4037"
      strokeWidth={1}
    />
  );
  lines.push(
    <line
      key="border-r"
      x1={toX(8)}
      y1={toY(4)}
      x2={toX(8)}
      y2={toY(5)}
      stroke="#5D4037"
      strokeWidth={1}
    />
  );

  // Palace diagonal lines (Black palace: rows 0-2, cols 3-5)
  lines.push(
    <line key="pd-b1" x1={toX(3)} y1={toY(0)} x2={toX(5)} y2={toY(2)} stroke="#5D4037" strokeWidth={1} />
  );
  lines.push(
    <line key="pd-b2" x1={toX(5)} y1={toY(0)} x2={toX(3)} y2={toY(2)} stroke="#5D4037" strokeWidth={1} />
  );

  // Palace diagonal lines (Red palace: rows 7-9, cols 3-5)
  lines.push(
    <line key="pd-r1" x1={toX(3)} y1={toY(7)} x2={toX(5)} y2={toY(9)} stroke="#5D4037" strokeWidth={1} />
  );
  lines.push(
    <line key="pd-r2" x1={toX(5)} y1={toY(7)} x2={toX(3)} y2={toY(9)} stroke="#5D4037" strokeWidth={1} />
  );

  return <>{lines}</>;
}

function RiverText() {
  const y = (toY(4) + toY(5)) / 2;
  return (
    <>
      <text
        x={toX(1.5)}
        y={y + 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#5D4037"
        fontSize="28"
        fontFamily="KaiTi, STKaiti, serif"
        fontWeight="bold"
        style={{ userSelect: "none" }}
      >
        楚河
      </text>
      <text
        x={toX(6.5)}
        y={y + 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#5D4037"
        fontSize="28"
        fontFamily="KaiTi, STKaiti, serif"
        fontWeight="bold"
        style={{ userSelect: "none" }}
      >
        汉界
      </text>
    </>
  );
}

export default function Board({ gameState, selectedPos, onCellClick }: BoardProps) {
  const { board, valid_moves } = gameState;

  // Get valid destinations for selected piece
  const validDestinations = new Set<string>();
  if (selectedPos) {
    const key = `${selectedPos[0]},${selectedPos[1]}`;
    const dests = valid_moves[key] || [];
    dests.forEach((d) => validDestinations.add(d));
  }

  // Find last move for highlighting
  const lastMove = gameState.move_history.length > 0
    ? gameState.move_history[gameState.move_history.length - 1]
    : null;

  return (
    <svg
      width={BOARD_WIDTH}
      height={BOARD_HEIGHT}
      viewBox={`0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`}
      className="select-none"
    >
      {/* Board background */}
      <rect
        x={0}
        y={0}
        width={BOARD_WIDTH}
        height={BOARD_HEIGHT}
        fill="#F5DEB3"
        rx={8}
      />
      <rect
        x={BOARD_PADDING - 10}
        y={BOARD_PADDING - 10}
        width={(COLS - 1) * CELL_SIZE + 20}
        height={(ROWS - 1) * CELL_SIZE + 20}
        fill="#DEB887"
        rx={4}
      />

      {/* Grid lines */}
      <GridLines />

      {/* River text */}
      <RiverText />

      {/* Last move highlight */}
      {lastMove && (
        <>
          <rect
            x={toX(lastMove.from[1]) - CELL_SIZE / 2 + 4}
            y={toY(lastMove.from[0]) - CELL_SIZE / 2 + 4}
            width={CELL_SIZE - 8}
            height={CELL_SIZE - 8}
            fill="rgba(255, 235, 59, 0.3)"
            rx={4}
          />
          <rect
            x={toX(lastMove.to[1]) - CELL_SIZE / 2 + 4}
            y={toY(lastMove.to[0]) - CELL_SIZE / 2 + 4}
            width={CELL_SIZE - 8}
            height={CELL_SIZE - 8}
            fill="rgba(255, 235, 59, 0.3)"
            rx={4}
          />
        </>
      )}

      {/* Valid move indicators */}
      {Array.from(validDestinations).map((dest) => {
        const [r, c] = dest.split(",").map(Number);
        const isCapture = board[r][c] !== 0;
        return isCapture ? (
          <circle
            key={`valid-${dest}`}
            cx={toX(c)}
            cy={toY(r)}
            r={28}
            fill="none"
            stroke="rgba(76, 175, 80, 0.6)"
            strokeWidth={3}
          />
        ) : (
          <circle
            key={`valid-${dest}`}
            cx={toX(c)}
            cy={toY(r)}
            r={10}
            fill="rgba(76, 175, 80, 0.5)"
          />
        );
      })}

      {/* Clickable cells (invisible) */}
      {Array.from({ length: ROWS }, (_, r) =>
        Array.from({ length: COLS }, (_, c) => (
          <rect
            key={`cell-${r}-${c}`}
            x={toX(c) - CELL_SIZE / 2}
            y={toY(r) - CELL_SIZE / 2}
            width={CELL_SIZE}
            height={CELL_SIZE}
            fill="transparent"
            onClick={() => onCellClick(r, c)}
            style={{ cursor: "pointer" }}
          />
        ))
      )}

      {/* Pieces */}
      {board.map((row, r) =>
        row.map((piece, c) => {
          if (piece === 0) return null;
          const isSelected =
            selectedPos !== null &&
            selectedPos[0] === r &&
            selectedPos[1] === c;
          return (
            <Piece
              key={`piece-${r}-${c}`}
              piece={piece}
              x={toX(c)}
              y={toY(r)}
              selected={isSelected}
              onClick={() => onCellClick(r, c)}
            />
          );
        })
      )}

      {/* "杀" overlay on checkmate */}
      {gameState.status !== "playing" && gameState.status !== "draw" && (
        <>
          <rect
            x={0}
            y={0}
            width={BOARD_WIDTH}
            height={BOARD_HEIGHT}
            fill="rgba(0,0,0,0.35)"
            rx={8}
          />
          <text
            x={BOARD_WIDTH / 2}
            y={BOARD_HEIGHT / 2}
            textAnchor="middle"
            dominantBaseline="central"
            fill="#C62828"
            fontSize="160"
            fontWeight="bold"
            fontFamily="KaiTi, STKaiti, serif"
            stroke="#FFD54F"
            strokeWidth={3}
            style={{ userSelect: "none" }}
          >
            杀
          </text>
        </>
      )}
    </svg>
  );
}
