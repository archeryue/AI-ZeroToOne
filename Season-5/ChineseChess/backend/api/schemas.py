"""Pydantic models for the API."""

from pydantic import BaseModel
from typing import Optional


class CreateGameRequest(BaseModel):
    player_color: str = "red"  # "red" or "black"
    ai_type: str = "greedy"  # "random", "greedy", "minimax", "human"
    ai_depth: int = 3  # depth for minimax AI


class MoveRequest(BaseModel):
    from_row: int
    from_col: int
    to_row: int
    to_col: int


class GameStateResponse(BaseModel):
    game_id: str
    board: list[list[int]]
    current_turn: str
    status: str
    in_check: bool
    valid_moves: dict[str, list[str]]
    move_history: list[dict]
    fen: str
    player_color: str
    ai_type: str


class MoveResponse(BaseModel):
    success: bool
    game_state: Optional[GameStateResponse] = None
    ai_move: Optional[dict] = None
    error: Optional[str] = None


# WebSocket message types
class WSMessage(BaseModel):
    type: str
    data: Optional[dict] = None
