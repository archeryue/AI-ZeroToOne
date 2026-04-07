import { API_BASE } from "./constants";
import type { CreateGameOptions, GameState } from "./types";

export async function createGame(options: CreateGameOptions): Promise<{
  game_id: string;
  game_state: GameState;
  ai_move?: { row: number; col: number };
}> {
  const res = await fetch(`${API_BASE}/api/games`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getGameState(gameId: string): Promise<{ game_state: GameState }> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function makeMove(
  gameId: string,
  row: number,
  col: number
): Promise<{ game_state: GameState; ai_move?: { row: number; col: number } }> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function passMove(
  gameId: string
): Promise<{ game_state: GameState; ai_move?: { row: number; col: number } }> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}/pass`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function undoMove(gameId: string): Promise<{ game_state: GameState }> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}/undo`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resignGame(gameId: string): Promise<{ game_state: GameState }> {
  const res = await fetch(`${API_BASE}/api/games/${gameId}/resign`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
