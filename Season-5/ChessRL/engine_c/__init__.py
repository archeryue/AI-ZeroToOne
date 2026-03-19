"""Fast C++ Chinese Chess engine with Python bindings."""

from engine_c._xiangqi import (
    # Constants
    RED, BLACK, ROWS, COLS, NUM_ACTIONS,
    GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER,
    STATUS_PLAYING, STATUS_RED_WIN, STATUS_BLACK_WIN, STATUS_DRAW,
    # Classes
    Board, Move, Game,
    # Functions
    get_legal_moves, is_in_check, is_flying_general,
    generate_all_moves, is_checkmate, is_stalemate,
    board_to_observation, get_action_mask, get_legal_action_indices,
    encode_move, decode_action,
)

__all__ = [
    'RED', 'BLACK', 'ROWS', 'COLS', 'NUM_ACTIONS',
    'GENERAL', 'ADVISOR', 'ELEPHANT', 'HORSE', 'CHARIOT', 'CANNON', 'SOLDIER',
    'STATUS_PLAYING', 'STATUS_RED_WIN', 'STATUS_BLACK_WIN', 'STATUS_DRAW',
    'Board', 'Move', 'Game',
    'get_legal_moves', 'is_in_check', 'is_flying_general',
    'generate_all_moves', 'is_checkmate', 'is_stalemate',
    'board_to_observation', 'get_action_mask', 'get_legal_action_indices',
    'encode_move', 'decode_action',
]
