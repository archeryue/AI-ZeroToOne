/*
 * xiangqi.h — Chinese Chess (Xiangqi) engine in C++.
 *
 * Board: flat int8_t[90] array (10 rows × 9 cols).
 * Pieces: ±1=General, ±2=Advisor, ±3=Elephant, ±4=Horse,
 *         ±5=Chariot, ±6=Cannon, ±7=Soldier. Positive=Red, Negative=Black.
 * Row 0 = Black back rank, Row 9 = Red back rank.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace xiangqi {

// Constants
constexpr int ROWS = 10;
constexpr int COLS = 9;
constexpr int BOARD_SIZE = ROWS * COLS;  // 90
constexpr int NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE;  // 8100

constexpr int8_t GENERAL = 1;
constexpr int8_t ADVISOR = 2;
constexpr int8_t ELEPHANT = 3;
constexpr int8_t HORSE = 4;
constexpr int8_t CHARIOT = 5;
constexpr int8_t CANNON = 6;
constexpr int8_t SOLDIER = 7;

constexpr int8_t RED = 1;
constexpr int8_t BLACK = -1;

constexpr int8_t STATUS_PLAYING = 0;
constexpr int8_t STATUS_RED_WIN = 1;
constexpr int8_t STATUS_BLACK_WIN = 2;
constexpr int8_t STATUS_DRAW = 3;

// Inline helpers
inline int sq(int r, int c) { return r * COLS + c; }
inline int sq_row(int s) { return s / COLS; }
inline int sq_col(int s) { return s % COLS; }
inline bool in_bounds(int r, int c) { return r >= 0 && r < ROWS && c >= 0 && c < COLS; }
inline int8_t color_of(int8_t piece) { return piece > 0 ? RED : (piece < 0 ? BLACK : 0); }
inline int8_t piece_type(int8_t piece) { return piece < 0 ? -piece : piece; }

inline int encode_action(int fr, int fc, int tr, int tc) {
    return (fr * COLS + fc) * BOARD_SIZE + (tr * COLS + tc);
}

// -----------------------------------------------------------------------
// Move
// -----------------------------------------------------------------------
struct Move {
    uint8_t from_sq;   // row*9+col
    uint8_t to_sq;
    int8_t captured;   // piece at dest before move (0 = no capture)
};

// Stack-allocated move list (max ~120 pseudo-legal moves)
struct MoveList {
    Move moves[128];
    int count = 0;

    void add(uint8_t from, uint8_t to, int8_t captured) {
        moves[count++] = {from, to, captured};
    }
};

// -----------------------------------------------------------------------
// Board
// -----------------------------------------------------------------------
struct Board {
    int8_t grid[BOARD_SIZE];
    int8_t red_general;    // flat index of red general, -1 if missing
    int8_t black_general;  // flat index of black general, -1 if missing

    Board();                        // initial position
    Board(const Board& o) = default;
    Board& operator=(const Board& o) = default;

    void init_from_grid(const int8_t src[BOARD_SIZE]);
    void update_general_cache();

    inline int8_t get(int r, int c) const { return grid[sq(r, c)]; }
    inline void set(int r, int c, int8_t piece) { grid[sq(r, c)] = piece; }
    Board copy() const { return *this; }

    int8_t general_pos(int8_t color) const {
        return color == RED ? red_general : black_general;
    }

    std::string to_fen() const;
    static Board from_fen(const std::string& fen);
};

// -----------------------------------------------------------------------
// Move generation
// -----------------------------------------------------------------------
void generate_pseudo_moves(const Board& b, int8_t color, MoveList& out);

// -----------------------------------------------------------------------
// Rules
// -----------------------------------------------------------------------
bool is_flying_general(const Board& b);
bool is_in_check(const Board& b, int8_t color);
void get_legal_moves(const Board& b, int8_t color, MoveList& out);
bool is_checkmate(const Board& b, int8_t color);
bool is_stalemate(const Board& b, int8_t color);

// Make/unmake (modifies board in-place, for legal move checking)
void make_move(Board& b, Move m);
void unmake_move(Board& b, Move m);

// -----------------------------------------------------------------------
// Game
// -----------------------------------------------------------------------
struct Game {
    Board board;
    int8_t current_turn;  // RED or BLACK
    int8_t status;        // STATUS_PLAYING, etc.

    Game();
    Game(const Game& o) = default;

    // Apply a legal move (from_row, from_col, to_row, to_col).
    // Returns true if the move was legal and applied.
    bool make_move(int fr, int fc, int tr, int tc);

    // Lightweight copy + apply action for MCTS. Does NOT validate legality.
    // Applies the move and checks game over.
    Game simulate_action(int action) const;

    void check_game_over();
};

// -----------------------------------------------------------------------
// Combined functions (avoid Python round-trips)
// -----------------------------------------------------------------------

// Returns encoded action indices for all legal moves
void get_legal_action_indices(const Board& b, int8_t color, std::vector<int>& out);

}  // namespace xiangqi
