"""REST API routes for Go game."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from engine.board import Stone, RESIGN
from engine.game import GameStatus
from .game_manager import game_manager

router = APIRouter(prefix="/api")


class CreateGameRequest(BaseModel):
    player_color: str = "black"
    ai_type: str = "random"
    board_size: int = 19
    mcts_sims: int = 500


class MoveRequest(BaseModel):
    row: int
    col: int


@router.post("/games")
def create_game(req: CreateGameRequest):
    if req.board_size not in (9, 13, 19):
        raise HTTPException(400, "Board size must be 9, 13, or 19")

    session = game_manager.create_game(
        player_color=req.player_color,
        ai_type=req.ai_type,
        board_size=req.board_size,
        mcts_sims=req.mcts_sims,
    )

    result = {"game_id": session.game_id, "game_state": session.to_state_dict()}

    # If AI plays first (player is white, AI is black)
    if session.ai and session.ai_color == Stone.BLACK:
        ai_move = session.ai.choose_move(session.game)
        if ai_move == RESIGN:
            session.game.resign(session.ai_color)
            result["game_state"] = session.to_state_dict()
            result["ai_resigned"] = True
        else:
            session.game.make_move(ai_move[0], ai_move[1])
            result["game_state"] = session.to_state_dict()
            result["ai_move"] = {"row": ai_move[0], "col": ai_move[1]}

    return result


@router.get("/games/{game_id}")
def get_game(game_id: str):
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    return {"game_state": session.to_state_dict()}


@router.post("/games/{game_id}/move")
def make_move(game_id: str, req: MoveRequest):
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(404, "Game not found")

    game = session.game
    if game.status != GameStatus.PLAYING:
        raise HTTPException(400, "Game is already over")

    try:
        move_info = game.make_move(req.row, req.col)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result = {"game_state": session.to_state_dict(), "player_move": move_info}

    # AI responds
    if session.ai and game.status == GameStatus.PLAYING and game.current_turn == session.ai_color:
        ai_move = session.ai.choose_move(game)
        if ai_move == RESIGN:
            game.resign(session.ai_color)
            result["game_state"] = session.to_state_dict()
            result["ai_resigned"] = True
        else:
            game.make_move(ai_move[0], ai_move[1])
            result["game_state"] = session.to_state_dict()
            result["ai_move"] = {"row": ai_move[0], "col": ai_move[1], "is_pass": ai_move == (-1, -1)}

    return result


@router.post("/games/{game_id}/pass")
def pass_move(game_id: str):
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(404, "Game not found")

    game = session.game
    if game.status != GameStatus.PLAYING:
        raise HTTPException(400, "Game is already over")

    game.make_move(-1, -1)
    result = {"game_state": session.to_state_dict()}

    # AI responds
    if session.ai and game.status == GameStatus.PLAYING and game.current_turn == session.ai_color:
        ai_move = session.ai.choose_move(game)
        if ai_move == RESIGN:
            game.resign(session.ai_color)
            result["game_state"] = session.to_state_dict()
            result["ai_resigned"] = True
        else:
            game.make_move(ai_move[0], ai_move[1])
            result["game_state"] = session.to_state_dict()
            result["ai_move"] = {"row": ai_move[0], "col": ai_move[1], "is_pass": ai_move == (-1, -1)}

    return result


@router.post("/games/{game_id}/undo")
def undo_move(game_id: str):
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(404, "Game not found")

    # Undo AI move + player move (2 undos)
    if session.ai:
        session.game.undo()
        session.game.undo()
    else:
        session.game.undo()

    return {"game_state": session.to_state_dict()}


@router.post("/games/{game_id}/resign")
def resign_game(game_id: str):
    session = game_manager.get_game(game_id)
    if not session:
        raise HTTPException(404, "Game not found")

    player_stone = Stone.BLACK if session.player_color == "black" else Stone.WHITE
    session.game.resign(player_stone)

    return {"game_state": session.to_state_dict()}
