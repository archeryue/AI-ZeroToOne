"""Movement rules for all 7 Chinese Chess piece types.

Each function returns a list of (row, col) squares a piece at (r, c) can
potentially move to, considering board geometry and blocking pieces.
Capturing/friendly-fire filtering is handled in rules.py.
"""

from engine.board import Board, ROWS, COLS, RED, BLACK
from engine.board import GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER


def _in_palace(row: int, col: int, color: int) -> bool:
    """Check if position is within the palace (九宫)."""
    if col < 3 or col > 5:
        return False
    if color == RED:
        return 7 <= row <= 9
    else:
        return 0 <= row <= 2


def general_moves(board: Board, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """General (帅/将): moves one step orthogonally within the palace."""
    moves = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if board.in_bounds(nr, nc) and _in_palace(nr, nc, color):
            moves.append((nr, nc))
    return moves


def advisor_moves(board: Board, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """Advisor (仕/士): moves one step diagonally within the palace."""
    moves = []
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if board.in_bounds(nr, nc) and _in_palace(nr, nc, color):
            moves.append((nr, nc))
    return moves


def elephant_moves(board: Board, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """Elephant (相/象): moves two steps diagonally, blocked by piece at midpoint (塞象眼).
    Cannot cross the river.
    """
    moves = []
    for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
        nr, nc = r + dr, c + dc
        mr, mc = r + dr // 2, c + dc // 2  # blocking midpoint
        if not board.in_bounds(nr, nc):
            continue
        # Cannot cross the river
        if color == RED and nr < 5:
            continue
        if color == BLACK and nr > 4:
            continue
        # Check blocking piece (塞象眼)
        if board.get(mr, mc) != 0:
            continue
        moves.append((nr, nc))
    return moves


def horse_moves(board: Board, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """Horse (马): L-shaped move, blocked by adjacent orthogonal piece (蹩马腿).

    The horse first moves one step orthogonally, then one step diagonally.
    If the orthogonal square is occupied, the move is blocked.
    """
    moves = []
    # (orthogonal step, then diagonal completions)
    for (br, bc), targets in [
        ((-1, 0), [(-2, -1), (-2, 1)]),
        ((1, 0), [(2, -1), (2, 1)]),
        ((0, -1), [(-1, -2), (1, -2)]),
        ((0, 1), [(-1, 2), (1, 2)]),
    ]:
        block_r, block_c = r + br, c + bc
        if not board.in_bounds(block_r, block_c):
            continue
        if board.get(block_r, block_c) != 0:
            continue  # 蹩马腿
        for dr, dc in targets:
            nr, nc = r + dr, c + dc
            if board.in_bounds(nr, nc):
                moves.append((nr, nc))
    return moves


def chariot_moves(board: Board, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """Chariot (车): moves any distance orthogonally, blocked by first piece."""
    moves = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        while board.in_bounds(nr, nc):
            if board.get(nr, nc) == 0:
                moves.append((nr, nc))
            else:
                moves.append((nr, nc))  # can capture
                break
            nr += dr
            nc += dc
    return moves


def cannon_moves(board: Board, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """Cannon (炮): moves like chariot but captures by jumping over exactly one piece (炮台)."""
    moves = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        jumped = False
        while board.in_bounds(nr, nc):
            piece = board.get(nr, nc)
            if not jumped:
                if piece == 0:
                    moves.append((nr, nc))  # non-capture move
                else:
                    jumped = True  # found the cannon platform
            else:
                if piece != 0:
                    moves.append((nr, nc))  # capture over the platform
                    break
            nr += dr
            nc += dc
    return moves


def soldier_moves(board: Board, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """Soldier (兵/卒): moves forward one step. After crossing river, can also move sideways."""
    moves = []
    if color == RED:
        # Red soldiers move upward (decreasing row)
        forward = (-1, 0)
        crossed_river = r <= 4
    else:
        # Black soldiers move downward (increasing row)
        forward = (1, 0)
        crossed_river = r >= 5

    # Forward move
    nr, nc = r + forward[0], c + forward[1]
    if board.in_bounds(nr, nc):
        moves.append((nr, nc))

    # Sideways moves (only after crossing river)
    if crossed_river:
        for dc in [-1, 1]:
            nc = c + dc
            if board.in_bounds(r, nc):
                moves.append((r, nc))

    return moves


# Dispatch table mapping piece type to move generator
MOVE_GENERATORS = {
    GENERAL: general_moves,
    ADVISOR: advisor_moves,
    ELEPHANT: elephant_moves,
    HORSE: horse_moves,
    CHARIOT: chariot_moves,
    CANNON: cannon_moves,
    SOLDIER: soldier_moves,
}


def get_piece_moves(board: Board, r: int, c: int) -> list[tuple[int, int]]:
    """Get potential target squares for the piece at (r, c)."""
    piece = board.get(r, c)
    if piece == 0:
        return []
    piece_type = abs(piece)
    color = RED if piece > 0 else BLACK
    generator = MOVE_GENERATORS.get(piece_type)
    if generator is None:
        return []
    return generator(board, r, c, color)
