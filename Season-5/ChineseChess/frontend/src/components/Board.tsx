"use client";

import React from "react";
import { ROWS, COLS, CELL_SIZE, BOARD_PADDING, BOARD_WIDTH, BOARD_HEIGHT } from "@/lib/constants";
import { GameState } from "@/lib/types";
import Piece from "./Piece";

interface BoardProps {
  gameState: GameState;
  selectedPos: [number, number] | null;
  onCellClick: (row: number, col: number) => void;
  flipped?: boolean;
}

function toX(col: number, flipped: boolean) {
  const c = flipped ? (COLS - 1 - col) : col;
  return BOARD_PADDING + c * CELL_SIZE;
}

function toY(row: number, flipped: boolean) {
  const r = flipped ? (ROWS - 1 - row) : row;
  return BOARD_PADDING + r * CELL_SIZE;
}

function vx(col: number) {
  return BOARD_PADDING + col * CELL_SIZE;
}

function vy(row: number) {
  return BOARD_PADDING + row * CELL_SIZE;
}

function GridLines() {
  const lines: React.ReactElement[] = [];

  for (let r = 0; r < ROWS; r++) {
    lines.push(
      <line key={`h-${r}`} x1={vx(0)} y1={vy(r)} x2={vx(COLS - 1)} y2={vy(r)} stroke="#5D4037" strokeWidth={1} />
    );
  }

  for (let c = 0; c < COLS; c++) {
    lines.push(
      <line key={`vt-${c}`} x1={vx(c)} y1={vy(0)} x2={vx(c)} y2={vy(4)} stroke="#5D4037" strokeWidth={1} />
    );
  }

  for (let c = 0; c < COLS; c++) {
    lines.push(
      <line key={`vb-${c}`} x1={vx(c)} y1={vy(5)} x2={vx(c)} y2={vy(9)} stroke="#5D4037" strokeWidth={1} />
    );
  }

  lines.push(
    <line key="border-l" x1={vx(0)} y1={vy(4)} x2={vx(0)} y2={vy(5)} stroke="#5D4037" strokeWidth={1} />
  );
  lines.push(
    <line key="border-r" x1={vx(8)} y1={vy(4)} x2={vx(8)} y2={vy(5)} stroke="#5D4037" strokeWidth={1} />
  );

  lines.push(
    <line key="pd-b1" x1={vx(3)} y1={vy(0)} x2={vx(5)} y2={vy(2)} stroke="#5D4037" strokeWidth={1} />
  );
  lines.push(
    <line key="pd-b2" x1={vx(5)} y1={vy(0)} x2={vx(3)} y2={vy(2)} stroke="#5D4037" strokeWidth={1} />
  );

  lines.push(
    <line key="pd-r1" x1={vx(3)} y1={vy(7)} x2={vx(5)} y2={vy(9)} stroke="#5D4037" strokeWidth={1} />
  );
  lines.push(
    <line key="pd-r2" x1={vx(5)} y1={vy(7)} x2={vx(3)} y2={vy(9)} stroke="#5D4037" strokeWidth={1} />
  );

  return <>{lines}</>;
}

function RiverText({ flipped }: { flipped: boolean }) {
  const visualY = BOARD_PADDING + 4 * CELL_SIZE + CELL_SIZE / 2;
  const leftX = BOARD_PADDING + 1.5 * CELL_SIZE;
  const rightX = BOARD_PADDING + 6.5 * CELL_SIZE;
  const leftText = flipped ? "汉界" : "楚河";
  const rightText = flipped ? "楚河" : "汉界";
  return (
    <>
      <text
        x={leftX}
        y={visualY + 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#5D4037"
        fontSize="28"
        fontFamily="KaiTi, STKaiti, serif"
        fontWeight="bold"
        style={{ userSelect: "none" }}
      >
        {leftText}
      </text>
      <text
        x={rightX}
        y={visualY + 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#5D4037"
        fontSize="28"
        fontFamily="KaiTi, STKaiti, serif"
        fontWeight="bold"
        style={{ userSelect: "none" }}
      >
        {rightText}
      </text>
    </>
  );
}

export default function Board({ gameState, selectedPos, onCellClick, flipped = false }: BoardProps) {
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
      <RiverText flipped={flipped} />

      {/* Last move highlight */}
      {lastMove && (
        <>
          <rect
            x={toX(lastMove.from[1], flipped) - CELL_SIZE / 2 + 4}
            y={toY(lastMove.from[0], flipped) - CELL_SIZE / 2 + 4}
            width={CELL_SIZE - 8}
            height={CELL_SIZE - 8}
            fill="rgba(255, 235, 59, 0.3)"
            rx={4}
          />
          <rect
            x={toX(lastMove.to[1], flipped) - CELL_SIZE / 2 + 4}
            y={toY(lastMove.to[0], flipped) - CELL_SIZE / 2 + 4}
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
            cx={toX(c, flipped)}
            cy={toY(r, flipped)}
            r={28}
            fill="none"
            stroke="rgba(76, 175, 80, 0.6)"
            strokeWidth={3}
          />
        ) : (
          <circle
            key={`valid-${dest}`}
            cx={toX(c, flipped)}
            cy={toY(r, flipped)}
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
            x={toX(c, flipped) - CELL_SIZE / 2}
            y={toY(r, flipped) - CELL_SIZE / 2}
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
              x={toX(c, flipped)}
              y={toY(r, flipped)}
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
