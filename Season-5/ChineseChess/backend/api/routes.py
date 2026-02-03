"""REST API endpoints."""

from fastapi import APIRouter, HTTPException
from api.schemas import CreateGameRequest, MoveRequest, GameStateResponse, MoveResponse
from api.game_manager import game_manager
from engine.board import RED, BLACK

router = APIRouter(prefix="/api")


@router.post("/games")
def create_game(req: CreateGameRequest) -> dict:
    session = game_manager.create_game(
        player_color=req.player_color,
        ai_type=req.ai_type,
        ai_depth=req.ai_depth,
    )

    # If AI plays first (player is black, AI is red), make AI move
    state = session.to_state_dict()
    ai_move = None
    if session.ai and session.ai_color == session.game.current_turn:
        move = session.ai.choose_move(session.game)
        session.game.make_move(move.from_row, move.from_col, move.to_row, move.to_col)
        ai_move = {"from": [move.from_row, move.from_col], "to": [move.to_row, move.to_col]}
        state = session.to_state_dict()

    return {"game_id": session.game_id, "game_state": state, "ai_move": ai_move}


@router.get("/games/{game_id}")
def get_game(game_id: str) -> dict:
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
    return session.to_state_dict()


@router.post("/games/{game_id}/move")
def make_move(game_id: str, req: MoveRequest) -> MoveResponse:
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")

    move = session.game.make_move(req.from_row, req.from_col, req.to_row, req.to_col)
    if move is None:
        return MoveResponse(success=False, error="Illegal move")

    # AI response
    ai_move_data = None
    if session.ai and session.game.status.value == "playing" and session.game.current_turn == session.ai_color:
        ai_move = session.ai.choose_move(session.game)
        session.game.make_move(ai_move.from_row, ai_move.from_col, ai_move.to_row, ai_move.to_col)
        ai_move_data = {"from": [ai_move.from_row, ai_move.from_col], "to": [ai_move.to_row, ai_move.to_col]}

    return MoveResponse(
        success=True,
        game_state=session.to_state_dict(),
        ai_move=ai_move_data,
    )


@router.post("/games/{game_id}/undo")
def undo_move(game_id: str) -> dict:
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")

    # Undo AI move first if applicable
    if session.ai:
        session.game.undo()  # undo AI move

    undone = session.game.undo()  # undo player move
    if undone is None:
        raise HTTPException(status_code=400, detail="No move to undo")

    return session.to_state_dict()


@router.post("/games/{game_id}/resign")
def resign(game_id: str) -> dict:
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")

    session.game.resign(session.player_color)
    return session.to_state_dict()
