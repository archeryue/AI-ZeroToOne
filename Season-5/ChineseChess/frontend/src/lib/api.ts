import { API_BASE } from "./constants";
import { CreateGameOptions, GameState } from "./types";

export async function createGame(options: CreateGameOptions): Promise<{
  game_id: string;
  game_state: GameState;
  ai_move: { from: number[]; to: number[] } | null;
}> {
  const res = await fetch(`${API_BASE}/api/games`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  if (!res.ok) throw new Error("Failed to create game");
  return res.json();
}

export async function getGameState(gameId: string): Promise<GameState> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}`);
  if (!res.ok) throw new Error("Game not found");
  return res.json();
}

export async function makeMove(
  gameId: string,
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number
) {
  const res = await fetch(`${API_BASE}/api/games/${gameId}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      from_row: fromRow,
      from_col: fromCol,
      to_row: toRow,
      to_col: toCol,
    }),
  });
  return res.json();
}

export async function undoMove(gameId: string): Promise<GameState> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}/undo`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Cannot undo");
  return res.json();
}

export async function resignGame(gameId: string): Promise<GameState> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}/resign`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Cannot resign");
  return res.json();
}
