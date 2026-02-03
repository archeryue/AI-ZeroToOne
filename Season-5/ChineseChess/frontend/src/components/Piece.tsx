"use client";

import { PIECE_NAMES_RED, PIECE_NAMES_BLACK, PIECE_RADIUS } from "@/lib/constants";

interface PieceProps {
  piece: number;
  x: number;
  y: number;
  selected: boolean;
  onClick: () => void;
}

export default function Piece({ piece, x, y, selected, onClick }: PieceProps) {
  const isRed = piece > 0;
  const pieceType = Math.abs(piece);
  const name = isRed
    ? PIECE_NAMES_RED[pieceType]
    : PIECE_NAMES_BLACK[pieceType];

  const fillColor = selected ? "#FFFDE7" : "#FFF8E1";
  const strokeColor = selected ? "#FF6F00" : isRed ? "#C62828" : "#1A237E";
  const textColor = isRed ? "#C62828" : "#1A237E";

  return (
    <g
      onClick={onClick}
      style={{ cursor: "pointer" }}
      className="piece-group"
    >
      {/* Shadow */}
      <circle
        cx={x + 2}
        cy={y + 2}
        r={PIECE_RADIUS}
        fill="rgba(0,0,0,0.15)"
      />
      {/* Outer ring */}
      <circle
        cx={x}
        cy={y}
        r={PIECE_RADIUS}
        fill={fillColor}
        stroke={strokeColor}
        strokeWidth={2.5}
      />
      {/* Inner ring */}
      <circle
        cx={x}
        cy={y}
        r={PIECE_RADIUS - 4}
        fill="none"
        stroke={strokeColor}
        strokeWidth={1}
      />
      {/* Chinese character */}
      <text
        x={x}
        y={y + 1}
        textAnchor="middle"
        dominantBaseline="central"
        fill={textColor}
        fontSize="22"
        fontWeight="bold"
        fontFamily="KaiTi, STKaiti, serif"
        style={{ userSelect: "none" }}
      >
        {name}
      </text>
    </g>
  );
}
