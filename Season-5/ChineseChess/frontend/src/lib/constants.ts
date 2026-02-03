export const ROWS = 10;
export const COLS = 9;

export const CELL_SIZE = 64;
export const BOARD_PADDING = 40;
export const PIECE_RADIUS = 26;

export const BOARD_WIDTH = (COLS - 1) * CELL_SIZE + BOARD_PADDING * 2;
export const BOARD_HEIGHT = (ROWS - 1) * CELL_SIZE + BOARD_PADDING * 2;

// Piece type codes (absolute value)
export const GENERAL = 1;
export const ADVISOR = 2;
export const ELEPHANT = 3;
export const HORSE = 4;
export const CHARIOT = 5;
export const CANNON = 6;
export const SOLDIER = 7;

// Chinese characters for pieces
export const PIECE_NAMES_RED: Record<number, string> = {
  1: "帅",
  2: "仕",
  3: "相",
  4: "马",
  5: "车",
  6: "炮",
  7: "兵",
};

export const PIECE_NAMES_BLACK: Record<number, string> = {
  1: "将",
  2: "士",
  3: "象",
  4: "马",
  5: "车",
  6: "炮",
  7: "卒",
};

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
export const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE || "ws://localhost:8000";
