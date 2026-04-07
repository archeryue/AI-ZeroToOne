export interface GameState {
  game_id: string;
  board: number[][];
  board_size: number;
  current_turn: "black" | "white";
  status: "playing" | "black_win" | "white_win";
  legal_moves: [number, number][];
  move_history: MoveRecord[];
  move_count: number;
  captured: { black: number; white: number };
  komi: number;
  score?: { black: number; white: number };
  territory?: number[][];
  player_color: "black" | "white";
  ai_type: string;
}

export interface MoveRecord {
  color: "black" | "white";
  position: string;
  captured: number;
}

export interface CreateGameOptions {
  player_color: string;
  ai_type: string;
  board_size: number;
  mcts_sims: number;
}

export interface WSMessage {
  type: "game_state" | "ai_thinking" | "ai_move" | "error";
  data?: GameState | { row: number; col: number; is_pass?: boolean };
  message?: string;
}
