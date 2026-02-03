"""WebSocket handler for real-time game communication."""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.game_manager import game_manager

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
                data = msg.get("data", {})
                from_row = data.get("from_row")
                from_col = data.get("from_col")
                to_row = data.get("to_row")
                to_col = data.get("to_col")

                move = session.game.make_move(from_row, from_col, to_row, to_col)
                if move is None:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "Illegal move"},
                    })
                    continue

                # Send updated state after player move
                await websocket.send_json({
                    "type": "game_state",
                    "data": session.to_state_dict(),
                })

                # AI response — run in thread so the player move flushes first
                if (
                    session.ai
                    and session.game.status.value == "playing"
                    and session.game.current_turn == session.ai_color
                ):
                    await websocket.send_json({
                        "type": "ai_thinking",
                        "data": {},
                    })
                    loop = asyncio.get_event_loop()
                    ai_move = await loop.run_in_executor(
                        None, session.ai.choose_move, session.game
                    )
                    session.game.make_move(
                        ai_move.from_row, ai_move.from_col,
                        ai_move.to_row, ai_move.to_col,
                    )
                    await websocket.send_json({
                        "type": "ai_move",
                        "data": {
                            "from": [ai_move.from_row, ai_move.from_col],
                            "to": [ai_move.to_row, ai_move.to_col],
                        },
                    })
                    await websocket.send_json({
                        "type": "game_state",
                        "data": session.to_state_dict(),
                    })

            elif msg_type == "undo":
                if session.ai:
                    session.game.undo()
                undone = session.game.undo()
                if undone:
                    await websocket.send_json({
                        "type": "game_state",
                        "data": session.to_state_dict(),
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "No move to undo"},
                    })

            elif msg_type == "resign":
                session.game.resign(session.player_color)
                await websocket.send_json({
                    "type": "game_state",
                    "data": session.to_state_dict(),
                })

            elif msg_type == "get_state":
                await websocket.send_json({
                    "type": "game_state",
                    "data": session.to_state_dict(),
                })

    except WebSocketDisconnect:
        pass
