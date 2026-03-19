/*
 * xiangqi.cpp — Chinese Chess engine implementation.
 */

#include "xiangqi.h"
#include <cstdlib>
#include <sstream>

namespace xiangqi {

// Initial board layout (row 0 = Black back rank, row 9 = Red back rank)
static const int8_t INITIAL_BOARD[BOARD_SIZE] = {
    -5, -4, -3, -2, -1, -2, -3, -4, -5,  // row 0
     0,  0,  0,  0,  0,  0,  0,  0,  0,  // row 1
     0, -6,  0,  0,  0,  0,  0, -6,  0,  // row 2
    -7,  0, -7,  0, -7,  0, -7,  0, -7,  // row 3
     0,  0,  0,  0,  0,  0,  0,  0,  0,  // row 4
     0,  0,  0,  0,  0,  0,  0,  0,  0,  // row 5
     7,  0,  7,  0,  7,  0,  7,  0,  7,  // row 6
     0,  6,  0,  0,  0,  0,  0,  6,  0,  // row 7
     0,  0,  0,  0,  0,  0,  0,  0,  0,  // row 8
     5,  4,  3,  2,  1,  2,  3,  4,  5,  // row 9
};

// -----------------------------------------------------------------------
// Board
// -----------------------------------------------------------------------

Board::Board() {
    std::memcpy(grid, INITIAL_BOARD, BOARD_SIZE);
    update_general_cache();
}

void Board::init_from_grid(const int8_t src[BOARD_SIZE]) {
    std::memcpy(grid, src, BOARD_SIZE);
    update_general_cache();
}

void Board::update_general_cache() {
    red_general = -1;
    black_general = -1;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (grid[i] == GENERAL)       red_general = i;
        else if (grid[i] == -GENERAL) black_general = i;
    }
}

std::string Board::to_fen() const {
    static const char* piece_chars = "?KABNRCP?kabnrcp";
    // Map: piece code -> char
    // Red (positive): 1=K, 2=A, 3=B, 4=N, 5=R, 6=C, 7=P
    // Black (negative, stored as 8+abs): same lowercase
    auto pc = [](int8_t p) -> char {
        if (p > 0 && p <= 7) {
            const char* red = "?KABNRCP";
            return red[p];
        } else if (p < 0 && p >= -7) {
            const char* blk = "?kabnrcp";
            return blk[-p];
        }
        return '?';
    };

    std::string fen;
    for (int r = 0; r < ROWS; ++r) {
        if (r > 0) fen += '/';
        int empty = 0;
        for (int c = 0; c < COLS; ++c) {
            int8_t p = grid[sq(r, c)];
            if (p == 0) {
                ++empty;
            } else {
                if (empty > 0) { fen += ('0' + empty); empty = 0; }
                fen += pc(p);
            }
        }
        if (empty > 0) fen += ('0' + empty);
    }
    return fen;
}

Board Board::from_fen(const std::string& fen) {
    Board b;
    std::memset(b.grid, 0, BOARD_SIZE);

    auto char_to_piece = [](char ch) -> int8_t {
        switch (ch) {
            case 'K': return  1; case 'A': return  2; case 'B': return  3;
            case 'N': return  4; case 'R': return  5; case 'C': return  6;
            case 'P': return  7;
            case 'k': return -1; case 'a': return -2; case 'b': return -3;
            case 'n': return -4; case 'r': return -5; case 'c': return -6;
            case 'p': return -7;
            default:  return  0;
        }
    };

    int r = 0, c = 0;
    for (char ch : fen) {
        if (ch == '/') { ++r; c = 0; }
        else if (ch >= '1' && ch <= '9') { c += (ch - '0'); }
        else { b.grid[sq(r, c)] = char_to_piece(ch); ++c; }
    }
    b.update_general_cache();
    return b;
}

// -----------------------------------------------------------------------
// Move generation — piece-specific generators
// -----------------------------------------------------------------------

static inline bool in_palace(int r, int c, int8_t color) {
    if (c < 3 || c > 5) return false;
    return color == RED ? (r >= 7 && r <= 9) : (r >= 0 && r <= 2);
}

static void gen_general(const Board& b, int r, int c, int8_t color, MoveList& out) {
    static const int dr[] = {-1, 1, 0, 0};
    static const int dc[] = {0, 0, -1, 1};
    int from = sq(r, c);
    for (int i = 0; i < 4; ++i) {
        int nr = r + dr[i], nc = c + dc[i];
        if (in_bounds(nr, nc) && in_palace(nr, nc, color)) {
            int8_t target = b.grid[sq(nr, nc)];
            if (color_of(target) != color)
                out.add(from, sq(nr, nc), target);
        }
    }
}

static void gen_advisor(const Board& b, int r, int c, int8_t color, MoveList& out) {
    static const int dr[] = {-1, -1, 1, 1};
    static const int dc[] = {-1, 1, -1, 1};
    int from = sq(r, c);
    for (int i = 0; i < 4; ++i) {
        int nr = r + dr[i], nc = c + dc[i];
        if (in_bounds(nr, nc) && in_palace(nr, nc, color)) {
            int8_t target = b.grid[sq(nr, nc)];
            if (color_of(target) != color)
                out.add(from, sq(nr, nc), target);
        }
    }
}

static void gen_elephant(const Board& b, int r, int c, int8_t color, MoveList& out) {
    static const int dr[] = {-2, -2, 2, 2};
    static const int dc[] = {-2, 2, -2, 2};
    int from = sq(r, c);
    for (int i = 0; i < 4; ++i) {
        int nr = r + dr[i], nc = c + dc[i];
        if (!in_bounds(nr, nc)) continue;
        // Cannot cross river
        if (color == RED && nr < 5) continue;
        if (color == BLACK && nr > 4) continue;
        // Check blocking piece at midpoint
        int mr = r + dr[i] / 2, mc = c + dc[i] / 2;
        if (b.grid[sq(mr, mc)] != 0) continue;
        int8_t target = b.grid[sq(nr, nc)];
        if (color_of(target) != color)
            out.add(from, sq(nr, nc), target);
    }
}

static void gen_horse(const Board& b, int r, int c, int8_t color, MoveList& out) {
    // (blocking step dr,dc) -> (target offsets)
    static const int block_dr[] = {-1, 1, 0, 0};
    static const int block_dc[] = {0, 0, -1, 1};
    static const int target_dr[][2] = {{-2, -2}, {2, 2}, {-1, 1}, {-1, 1}};
    static const int target_dc[][2] = {{-1, 1}, {-1, 1}, {-2, -2}, {2, 2}};

    int from = sq(r, c);
    for (int i = 0; i < 4; ++i) {
        int br = r + block_dr[i], bc = c + block_dc[i];
        if (!in_bounds(br, bc)) continue;
        if (b.grid[sq(br, bc)] != 0) continue;  // blocked
        for (int j = 0; j < 2; ++j) {
            int nr = r + target_dr[i][j], nc = c + target_dc[i][j];
            if (!in_bounds(nr, nc)) continue;
            int8_t target = b.grid[sq(nr, nc)];
            if (color_of(target) != color)
                out.add(from, sq(nr, nc), target);
        }
    }
}

static void gen_chariot(const Board& b, int r, int c, int8_t color, MoveList& out) {
    static const int dr[] = {-1, 1, 0, 0};
    static const int dc[] = {0, 0, -1, 1};
    int from = sq(r, c);
    for (int i = 0; i < 4; ++i) {
        int nr = r + dr[i], nc = c + dc[i];
        while (in_bounds(nr, nc)) {
            int8_t target = b.grid[sq(nr, nc)];
            if (target == 0) {
                out.add(from, sq(nr, nc), 0);
            } else {
                if (color_of(target) != color)
                    out.add(from, sq(nr, nc), target);
                break;
            }
            nr += dr[i];
            nc += dc[i];
        }
    }
}

static void gen_cannon(const Board& b, int r, int c, int8_t color, MoveList& out) {
    static const int dr[] = {-1, 1, 0, 0};
    static const int dc[] = {0, 0, -1, 1};
    int from = sq(r, c);
    for (int i = 0; i < 4; ++i) {
        int nr = r + dr[i], nc = c + dc[i];
        bool jumped = false;
        while (in_bounds(nr, nc)) {
            int8_t target = b.grid[sq(nr, nc)];
            if (!jumped) {
                if (target == 0) {
                    out.add(from, sq(nr, nc), 0);
                } else {
                    jumped = true;  // cannon platform
                }
            } else {
                if (target != 0) {
                    if (color_of(target) != color)
                        out.add(from, sq(nr, nc), target);
                    break;
                }
            }
            nr += dr[i];
            nc += dc[i];
        }
    }
}

static void gen_soldier(const Board& b, int r, int c, int8_t color, MoveList& out) {
    int from = sq(r, c);
    if (color == RED) {
        // Forward = up (row-1)
        if (r - 1 >= 0) {
            int8_t t = b.grid[sq(r - 1, c)];
            if (color_of(t) != color) out.add(from, sq(r - 1, c), t);
        }
        // Sideways after crossing river (r <= 4)
        if (r <= 4) {
            if (c - 1 >= 0) {
                int8_t t = b.grid[sq(r, c - 1)];
                if (color_of(t) != color) out.add(from, sq(r, c - 1), t);
            }
            if (c + 1 < COLS) {
                int8_t t = b.grid[sq(r, c + 1)];
                if (color_of(t) != color) out.add(from, sq(r, c + 1), t);
            }
        }
    } else {
        // Forward = down (row+1)
        if (r + 1 < ROWS) {
            int8_t t = b.grid[sq(r + 1, c)];
            if (color_of(t) != color) out.add(from, sq(r + 1, c), t);
        }
        // Sideways after crossing river (r >= 5)
        if (r >= 5) {
            if (c - 1 >= 0) {
                int8_t t = b.grid[sq(r, c - 1)];
                if (color_of(t) != color) out.add(from, sq(r, c - 1), t);
            }
            if (c + 1 < COLS) {
                int8_t t = b.grid[sq(r, c + 1)];
                if (color_of(t) != color) out.add(from, sq(r, c + 1), t);
            }
        }
    }
}

void generate_pseudo_moves(const Board& b, int8_t color, MoveList& out) {
    out.count = 0;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        int8_t p = b.grid[i];
        if (color_of(p) != color) continue;
        int r = sq_row(i), c = sq_col(i);
        int8_t pt = piece_type(p);
        switch (pt) {
            case GENERAL:  gen_general(b, r, c, color, out);  break;
            case ADVISOR:  gen_advisor(b, r, c, color, out);  break;
            case ELEPHANT: gen_elephant(b, r, c, color, out); break;
            case HORSE:    gen_horse(b, r, c, color, out);    break;
            case CHARIOT:  gen_chariot(b, r, c, color, out);  break;
            case CANNON:   gen_cannon(b, r, c, color, out);   break;
            case SOLDIER:  gen_soldier(b, r, c, color, out);  break;
        }
    }
}

// -----------------------------------------------------------------------
// Make / Unmake
// -----------------------------------------------------------------------

// Saved state for unmake: just the general positions before the move
struct MoveUndo {
    int8_t prev_red_gen;
    int8_t prev_black_gen;
};

static MoveUndo s_undo;  // thread-local would be safer, but we're single-threaded per worker

void make_move(Board& b, Move m) {
    s_undo.prev_red_gen = b.red_general;
    s_undo.prev_black_gen = b.black_general;

    int8_t piece = b.grid[m.from_sq];
    b.grid[m.to_sq] = piece;
    b.grid[m.from_sq] = 0;

    // Update general cache
    if (piece == GENERAL)       b.red_general = m.to_sq;
    else if (piece == -GENERAL) b.black_general = m.to_sq;
    // If a general was captured
    if (m.captured == GENERAL)       b.red_general = -1;
    else if (m.captured == -GENERAL) b.black_general = -1;
}

void unmake_move(Board& b, Move m) {
    int8_t piece = b.grid[m.to_sq];
    b.grid[m.from_sq] = piece;
    b.grid[m.to_sq] = m.captured;

    b.red_general = s_undo.prev_red_gen;
    b.black_general = s_undo.prev_black_gen;
}

// -----------------------------------------------------------------------
// Check detection — targeted attack checking from general's position
// -----------------------------------------------------------------------

bool is_flying_general(const Board& b) {
    if (b.red_general < 0 || b.black_general < 0) return false;
    int rc = sq_col(b.red_general);
    int bc = sq_col(b.black_general);
    if (rc != bc) return false;

    int rr = sq_row(b.red_general);
    int br = sq_row(b.black_general);
    int min_r = (br < rr) ? br : rr;
    int max_r = (br > rr) ? br : rr;

    for (int r = min_r + 1; r < max_r; ++r) {
        if (b.grid[sq(r, rc)] != 0) return false;
    }
    return true;
}

bool is_in_check(const Board& b, int8_t color) {
    int8_t gpos = b.general_pos(color);
    if (gpos < 0) return true;  // general captured

    int gr = sq_row(gpos), gc = sq_col(gpos);
    int8_t opp = -color;

    // Check orthogonal rays for Chariot and face-to-face General
    static const int dr[] = {-1, 1, 0, 0};
    static const int dc[] = {0, 0, -1, 1};
    for (int d = 0; d < 4; ++d) {
        int nr = gr + dr[d], nc = gc + dc[d];
        while (in_bounds(nr, nc)) {
            int8_t p = b.grid[sq(nr, nc)];
            if (p != 0) {
                if (color_of(p) == opp) {
                    int8_t pt = piece_type(p);
                    if (pt == CHARIOT) return true;
                    if (pt == GENERAL) return true;  // flying general
                }
                break;
            }
            nr += dr[d];
            nc += dc[d];
        }
    }

    // Check orthogonal rays for Cannon (needs exactly one piece between)
    for (int d = 0; d < 4; ++d) {
        int nr = gr + dr[d], nc = gc + dc[d];
        bool jumped = false;
        while (in_bounds(nr, nc)) {
            int8_t p = b.grid[sq(nr, nc)];
            if (!jumped) {
                if (p != 0) jumped = true;
            } else {
                if (p != 0) {
                    if (color_of(p) == opp && piece_type(p) == CANNON)
                        return true;
                    break;
                }
            }
            nr += dr[d];
            nc += dc[d];
        }
    }

    // Check Horse attacks (reverse horse movement)
    // A horse at (hr,hc) can attack general if: orthogonal step from horse is
    // unblocked, then diagonal step reaches general.
    // Reverse: from general, check 8 possible horse positions.
    static const int h_dr[] = {-2, -2, -1, -1,  1,  1,  2,  2};
    static const int h_dc[] = {-1,  1, -2,  2, -2,  2, -1,  1};
    // blocking square for each horse position (the orthogonal step FROM the horse)
    static const int hb_dr[] = { 1,  1,  0,  0,  0,  0, -1, -1};
    static const int hb_dc[] = { 0,  0,  1, -1,  1, -1,  0,  0};

    for (int i = 0; i < 8; ++i) {
        int hr = gr + h_dr[i], hc = gc + h_dc[i];
        if (!in_bounds(hr, hc)) continue;
        int8_t p = b.grid[sq(hr, hc)];
        if (color_of(p) != opp || piece_type(p) != HORSE) continue;
        // Check blocking square (from the horse's perspective)
        int blk_r = hr + hb_dr[i], blk_c = hc + hb_dc[i];
        if (b.grid[sq(blk_r, blk_c)] == 0)
            return true;
    }

    // Check Soldier attacks
    // A soldier can attack if it's one step away in the right direction
    if (color == RED) {
        // Red general attacked by black soldier: soldier above (r-1) or sideways
        // Black soldier moves down (row+1), so it attacks from row gr-1
        // After crossing river (r>=5), also sideways
        int sr = gr + 1;  // soldier one step below would have moved down to gr
        // Wait — black soldier attacks by being AT position that can reach general
        // Black soldier at (gr-1, gc) can move down to (gr, gc)? No.
        // Black soldier moves down. So a black soldier at (gr-1, gc) CAN reach gr,gc
        // by moving forward (down). Also at (gr, gc±1) if crossed river.
        if (in_bounds(gr - 1, gc) && b.grid[sq(gr - 1, gc)] == -SOLDIER)
            return true;
        // Sideways: soldier at same row, adj col, but only if crossed river (r>=5)
        if (gr >= 5) {
            if (gc - 1 >= 0 && b.grid[sq(gr, gc - 1)] == -SOLDIER) return true;
            if (gc + 1 < COLS && b.grid[sq(gr, gc + 1)] == -SOLDIER) return true;
        }
    } else {
        // Black general attacked by red soldier
        // Red soldier moves up (row-1). Red soldier at (gr+1, gc) attacks general.
        if (in_bounds(gr + 1, gc) && b.grid[sq(gr + 1, gc)] == SOLDIER)
            return true;
        if (gr <= 4) {
            if (gc - 1 >= 0 && b.grid[sq(gr, gc - 1)] == SOLDIER) return true;
            if (gc + 1 < COLS && b.grid[sq(gr, gc + 1)] == SOLDIER) return true;
        }
    }

    // Advisors and Elephants cannot attack from distance — no need to check.
    // (They only move within palace/own side, and we check from general's POV.)

    return false;
}

// -----------------------------------------------------------------------
// Legal move generation (make/unmake pattern)
// -----------------------------------------------------------------------

void get_legal_moves(const Board& b, int8_t color, MoveList& out) {
    out.count = 0;
    MoveList pseudo;
    generate_pseudo_moves(b, color, pseudo);

    // Cast away const — we restore the board via unmake
    Board& mut_b = const_cast<Board&>(b);

    for (int i = 0; i < pseudo.count; ++i) {
        Move m = pseudo.moves[i];
        make_move(mut_b, m);
        bool legal = !is_in_check(mut_b, color);
        unmake_move(mut_b, m);
        if (legal) {
            out.moves[out.count++] = m;
        }
    }
}

bool is_checkmate(const Board& b, int8_t color) {
    if (!is_in_check(b, color)) return false;
    MoveList legal;
    get_legal_moves(b, color, legal);
    return legal.count == 0;
}

bool is_stalemate(const Board& b, int8_t color) {
    if (is_in_check(b, color)) return false;
    MoveList legal;
    get_legal_moves(b, color, legal);
    return legal.count == 0;
}

// -----------------------------------------------------------------------
// Game
// -----------------------------------------------------------------------

Game::Game() : current_turn(RED), status(STATUS_PLAYING) {}

bool Game::make_move(int fr, int fc, int tr, int tc) {
    if (status != STATUS_PLAYING) return false;

    // Verify legality
    MoveList legal;
    get_legal_moves(board, current_turn, legal);

    int from = sq(fr, fc);
    int to = sq(tr, tc);
    Move found_move{};
    bool found = false;

    for (int i = 0; i < legal.count; ++i) {
        if (legal.moves[i].from_sq == from && legal.moves[i].to_sq == to) {
            found_move = legal.moves[i];
            found = true;
            break;
        }
    }
    if (!found) return false;

    // Apply
    int8_t piece = board.grid[from];
    board.grid[from] = 0;
    board.grid[to] = piece;

    // Update general cache
    if (piece == GENERAL)       board.red_general = to;
    else if (piece == -GENERAL) board.black_general = to;
    if (found_move.captured == GENERAL)       board.red_general = -1;
    else if (found_move.captured == -GENERAL) board.black_general = -1;

    current_turn = -current_turn;
    check_game_over();
    return true;
}

Game Game::simulate_action(int action) const {
    Game g = *this;  // copy

    int from_sq = action / BOARD_SIZE;
    int to_sq = action % BOARD_SIZE;

    int8_t piece = g.board.grid[from_sq];
    int8_t captured = g.board.grid[to_sq];
    g.board.grid[from_sq] = 0;
    g.board.grid[to_sq] = piece;

    // Update general cache
    if (piece == GENERAL)       g.board.red_general = to_sq;
    else if (piece == -GENERAL) g.board.black_general = to_sq;
    if (captured == GENERAL)       g.board.red_general = -1;
    else if (captured == -GENERAL) g.board.black_general = -1;

    g.current_turn = -g.current_turn;
    g.check_game_over();
    return g;
}

void Game::check_game_over() {
    if (is_checkmate(board, current_turn)) {
        status = (current_turn == BLACK) ? STATUS_RED_WIN : STATUS_BLACK_WIN;
    } else if (is_stalemate(board, current_turn)) {
        status = (current_turn == BLACK) ? STATUS_RED_WIN : STATUS_BLACK_WIN;
    }
}

// -----------------------------------------------------------------------
// Combined functions
// -----------------------------------------------------------------------

void get_legal_action_indices(const Board& b, int8_t color, std::vector<int>& out) {
    out.clear();
    MoveList legal;
    get_legal_moves(b, color, legal);
    for (int i = 0; i < legal.count; ++i) {
        Move& m = legal.moves[i];
        int fr = sq_row(m.from_sq), fc = sq_col(m.from_sq);
        int tr = sq_row(m.to_sq),   tc = sq_col(m.to_sq);
        out.push_back(encode_action(fr, fc, tr, tc));
    }
}

}  // namespace xiangqi
