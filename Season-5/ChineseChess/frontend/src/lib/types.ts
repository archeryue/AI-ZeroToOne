export interface GameState {
  game_id: string;
  board: number[][];
  current_turn: "red" | "black";
  status: "playing" | "red_win" | "black_win" | "draw";
  in_check: boolean;
  valid_moves: Record<string, string[]>;
  move_history: MoveRecord[];
  fen: string;
  player_color: "red" | "black";
  ai_type: string;
}

export interface MoveRecord {
  from: [number, number];
  to: [number, number];
  captured: number;
  ucci: string;
}

export interface WSMessage {
  type: "game_state" | "ai_move" | "ai_thinking" | "error";
  data: GameState | AIMoveData | ErrorData;
}

export interface AIMoveData {
  from: [number, number];
  to: [number, number];
}

export interface ErrorData {
  message: string;
}

export interface CreateGameOptions {
  player_color: "red" | "black";
  ai_type: "random" | "greedy" | "minimax" | "human";
  ai_depth?: number;
}
