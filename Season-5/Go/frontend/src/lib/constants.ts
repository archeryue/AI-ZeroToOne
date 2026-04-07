export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8001";
export const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE || "ws://localhost:8001";

// Board rendering
export const BOARD_PADDING = 40;
export const MIN_CELL_SIZE = 28;
export const MAX_CELL_SIZE = 48;

export function getCellSize(boardSize: number): number {
  if (boardSize <= 9) return MAX_CELL_SIZE;
  if (boardSize <= 13) return 38;
  return MIN_CELL_SIZE;
}

export function getBoardPixelSize(boardSize: number): number {
  return BOARD_PADDING * 2 + getCellSize(boardSize) * (boardSize - 1);
}

// Star points (hoshi)
export function getStarPoints(size: number): [number, number][] {
  if (size === 9) {
    return [
      [2, 2], [2, 6], [6, 2], [6, 6], [4, 4],
    ];
  }
  if (size === 13) {
    return [
      [3, 3], [3, 9], [9, 3], [9, 9], [6, 6],
      [3, 6], [6, 3], [6, 9], [9, 6],
    ];
  }
  // 19x19
  return [
    [3, 3], [3, 9], [3, 15],
    [9, 3], [9, 9], [9, 15],
    [15, 3], [15, 9], [15, 15],
  ];
}
