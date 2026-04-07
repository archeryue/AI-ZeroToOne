"""WebSocket handler for real-time Go gameplay."""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engine.board import Stone
from engine.game import GameStatus
from .game_manager import game_manager

router = APIRouter()


@router.websocket("/ws/{game_id}")
async def game_websocket(websocket: WebSocket, game_id: str):
    session = game_manager.get_game(game_id)
    if not session:
        await websocket.close(code=4004, reason="Game not found")
        return

    await websocket.accept()

    # Send initial state
    await websocket.send_json({
        "type": "game_state",
        "data": session.to_state_dict(),
    })

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "move":
                await _handle_move(websocket, session, msg)
            elif msg_type == "pass":
                await _handle_pass(websocket, session)
            elif msg_type == "undo":
                await _handle_undo(websocket, session)
            elif msg_type == "resign":
                await _handle_resign(websocket, session)
            elif msg_type == "get_state":
                await websocket.send_json({
                    "type": "game_state",
                    "data": session.to_state_dict(),
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _handle_move(websocket, session, msg):
    game = session.game
    row = msg.get("row", -1)
    col = msg.get("col", -1)

    if game.status != GameStatus.PLAYING:
        await websocket.send_json({"type": "error", "message": "Game is over"})
        return

    try:
        game.make_move(row, col)
    except ValueError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        return

    await websocket.send_json({
        "type": "game_state",
        "data": session.to_state_dict(),
    })

    # AI turn
    if session.ai and game.status == GameStatus.PLAYING and game.current_turn == session.ai_color:
        await websocket.send_json({"type": "ai_thinking"})

        loop = asyncio.get_event_loop()
        ai_move = await loop.run_in_executor(None, session.ai.choose_move, game)

        game.make_move(ai_move[0], ai_move[1])

        await websocket.send_json({
            "type": "ai_move",
            "data": {"row": ai_move[0], "col": ai_move[1], "is_pass": ai_move == (-1, -1)},
        })
        await websocket.send_json({
            "type": "game_state",
            "data": session.to_state_dict(),
        })


async def _handle_pass(websocket, session):
    game = session.game

    if game.status != GameStatus.PLAYING:
        await websocket.send_json({"type": "error", "message": "Game is over"})
        return

    game.make_move(-1, -1)
    await websocket.send_json({
        "type": "game_state",
        "data": session.to_state_dict(),
    })

    # AI turn
    if session.ai and game.status == GameStatus.PLAYING and game.current_turn == session.ai_color:
        await websocket.send_json({"type": "ai_thinking"})

        loop = asyncio.get_event_loop()
        ai_move = await loop.run_in_executor(None, session.ai.choose_move, game)

        game.make_move(ai_move[0], ai_move[1])

        await websocket.send_json({
            "type": "ai_move",
            "data": {"row": ai_move[0], "col": ai_move[1], "is_pass": ai_move == (-1, -1)},
        })
        await websocket.send_json({
            "type": "game_state",
            "data": session.to_state_dict(),
        })


async def _handle_undo(websocket, session):
    if session.ai:
        session.game.undo()
        session.game.undo()
    else:
        session.game.undo()

    await websocket.send_json({
        "type": "game_state",
        "data": session.to_state_dict(),
    })


async def _handle_resign(websocket, session):
    player_stone = Stone.BLACK if session.player_color == "black" else Stone.WHITE
    session.game.resign(player_stone)

    await websocket.send_json({
        "type": "game_state",
        "data": session.to_state_dict(),
    })
