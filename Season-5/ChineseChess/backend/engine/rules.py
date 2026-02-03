"""Check detection, legal move filtering, and game-over conditions."""

from engine.board import Board, ROWS, COLS, RED, BLACK, GENERAL
from engine.move import Move
from engine.pieces import get_piece_moves


def is_flying_general(board: Board) -> bool:
    """Check if the two generals face each other on the same column with no
    pieces in between (飞将). This is an illegal position.
    """
    try:
        r_red, c_red = board.find_general(RED)
        r_black, c_black = board.find_general(BLACK)
    except ValueError:
        return False

    if c_red != c_black:
        return False

    # Check if any piece stands between them
    col = c_red
    min_row = min(r_red, r_black)
    max_row = max(r_red, r_black)
    for r in range(min_row + 1, max_row):
        if board.get(r, col) != 0:
            return False
    return True  # generals face each other with nothing between


def is_in_check(board: Board, color: int) -> bool:
    """Check if the given color's general is under attack."""
    try:
        gen_r, gen_c = board.find_general(color)
    except ValueError:
        return True  # general captured = in check

    opponent = -color

    # Check if any opponent piece can reach the general
    for r in range(ROWS):
        for c in range(COLS):
            if board.color_of(r, c) != opponent:
                continue
            targets = get_piece_moves(board, r, c)
            if (gen_r, gen_c) in targets:
                return True

    # Flying general rule: if generals face each other, both are "in check"
    if is_flying_general(board):
        return True

    return False


def _apply_move_temp(board: Board, move: Move) -> Board:
    """Apply a move to a copy of the board and return the new board."""
    new_board = board.copy()
    piece = new_board.get(move.from_row, move.from_col)
    new_board.set(move.from_row, move.from_col, 0)
    new_board.set(move.to_row, move.to_col, piece)
    return new_board


def generate_all_moves(board: Board, color: int) -> list[Move]:
    """Generate all pseudo-legal moves for the given color.

    This does NOT filter out moves that leave the king in check.
    """
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            if board.color_of(r, c) != color:
                continue
            targets = get_piece_moves(board, r, c)
            for tr, tc in targets:
                # Cannot capture own piece
                if board.color_of(tr, tc) == color:
                    continue
                captured = board.get(tr, tc)
                moves.append(Move(r, c, tr, tc, captured))
    return moves


def get_legal_moves(board: Board, color: int) -> list[Move]:
    """Generate all legal moves for the given color.

    Filters out moves that would leave own general in check or
    create a flying general situation.
    """
    legal = []
    for move in generate_all_moves(board, color):
        new_board = _apply_move_temp(board, move)
        if not is_in_check(new_board, color):
            legal.append(move)
    return legal


def is_checkmate(board: Board, color: int) -> bool:
    """Check if the given color is in checkmate (in check with no legal moves)."""
    if not is_in_check(board, color):
        return False
    return len(get_legal_moves(board, color)) == 0


def is_stalemate(board: Board, color: int) -> bool:
    """Check if the given color is stalemated (not in check but no legal moves).

    In Chinese Chess, stalemate is a loss for the stalemated player.
    """
    if is_in_check(board, color):
        return False
    return len(get_legal_moves(board, color)) == 0
