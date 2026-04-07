// C++ unit tests for Go engine.
// Compile: g++ -std=c++17 -O2 -I.. test_go.cpp ../go.cpp -o test_go && ./test_go

#include "go.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <set>
#include <chrono>
#include <random>

using namespace go;

// Helper: place stone by (row, col)
template<int N>
int place(Board<N>& b, int r, int c, uint8_t clr) {
    return b.place_stone(Board<N>::pos(r, c), clr);
}

template<int N>
bool legal(const Board<N>& b, int r, int c, uint8_t clr) {
    return b.is_legal_checked(Board<N>::pos(r, c), clr);
}

// ─── Test 1: Basic placement ────────────────────────────────────

void test_basic_placement() {
    Board<9> b;
    assert(b.color[Board<9>::pos(2, 3)] == EMPTY);

    int cap = place(b, 2, 3, BLACK);
    assert(cap == 0);
    assert(b.color[Board<9>::pos(2, 3)] == BLACK);

    cap = place(b, 4, 5, WHITE);
    assert(cap == 0);
    assert(b.color[Board<9>::pos(4, 5)] == WHITE);

    // Can't place on occupied
    cap = place(b, 2, 3, WHITE);
    assert(cap == -1);

    printf("  PASS: basic placement\n");
}

// ─── Test 2: Simple capture ─────────────────────────────────────

void test_simple_capture() {
    Board<9> b;
    // Surround a single white stone
    //     . B .
    //     B W B
    //     . B .
    place(b, 1, 1, WHITE);  // the victim
    place(b, 0, 1, BLACK);
    place(b, 2, 1, BLACK);
    place(b, 1, 0, BLACK);
    int cap = place(b, 1, 2, BLACK);  // captures white
    assert(cap == 1);
    assert(b.color[Board<9>::pos(1, 1)] == EMPTY);

    printf("  PASS: simple capture\n");
}

// ─── Test 3: Group capture ──────────────────────────────────────

void test_group_capture() {
    Board<9> b;
    // Two white stones in a row, surrounded
    //   . B B .
    //   B W W B
    //   . B B .
    place(b, 1, 1, WHITE);
    place(b, 1, 2, WHITE);
    // Surround
    place(b, 0, 1, BLACK);
    place(b, 0, 2, BLACK);
    place(b, 2, 1, BLACK);
    place(b, 2, 2, BLACK);
    place(b, 1, 0, BLACK);
    int cap = place(b, 1, 3, BLACK);
    assert(cap == 2);
    assert(b.color[Board<9>::pos(1, 1)] == EMPTY);
    assert(b.color[Board<9>::pos(1, 2)] == EMPTY);

    printf("  PASS: group capture\n");
}

// ─── Test 4: Corner capture ─────────────────────────────────────

void test_corner_capture() {
    Board<9> b;
    // White in corner (0,0), surrounded by black
    place(b, 0, 0, WHITE);
    place(b, 0, 1, BLACK);
    int cap = place(b, 1, 0, BLACK);
    assert(cap == 1);
    assert(b.color[Board<9>::pos(0, 0)] == EMPTY);

    printf("  PASS: corner capture\n");
}

// ─── Test 5: Suicide prevention ─────────────────────────────────

void test_suicide_prevention() {
    Board<9> b;
    // Black stones surrounding an intersection, white tries to play inside
    //   B .
    //   . B
    // Actually let's do a clearer case:
    //   . B .
    //   B . B    ← white plays center = suicide (no capture)
    //   . B .
    place(b, 0, 1, BLACK);
    place(b, 2, 1, BLACK);
    place(b, 1, 0, BLACK);
    place(b, 1, 2, BLACK);

    // White playing at (1,1) would be suicide
    assert(!legal(b, 1, 1, WHITE));
    int cap = place(b, 1, 1, WHITE);
    assert(cap == -1);

    printf("  PASS: suicide prevention\n");
}

// ─── Test 6: Suicide that captures is legal ─────────────────────

void test_capture_not_suicide() {
    Board<9> b;
    // White stone at center with 1 liberty, black fills last liberty but CAPTURES
    //   . B .
    //   B W B    ← black at (1,1) would capture white  [wait, white is there]
    // Let's set up properly:
    //
    //   B W .
    //   W . .     ← white at (0,1) has liberties at (0,2)
    //
    // Actually simpler: black plays into a spot that looks like suicide but captures
    //
    //   W B        ← black group has 1 liberty
    //   B .        ← white plays at (1,1) — is it suicide or capture?
    //
    // More explicit:
    //   . W .
    //   W B W     ← black at (1,1) with 1 liberty at (0,1)
    //   . W .
    // Wait, (0,1) is WHITE. Let me think more carefully.
    //
    // Let's do: black has surrounded white stone, and playing captures it
    // The key test is: playing into an "enclosed" area that CAPTURES is legal.
    //
    //   B W B
    //   W . W     ← black plays (1,1), looks surrounded but captures both whites at (0,1),(1,0),(1,2),(2,1)?
    //   B W B
    //
    // Hmm, let me just verify the simple capture test already covers this.
    // The real test: a multi-stone "snap-back" capture

    // Simpler test: black plays into a spot with no empty neighbors but captures opponent
    //   . B .
    //   B W B    ← white has 1 liberty at (2,1)
    //   . * .    ← black plays * = captures white, so it's legal
    //
    // Actually we already tested this in test_simple_capture. Let me test something different:
    // Playing in a corner where it looks like suicide but captures.

    // Corner: (0,0)
    //   * W B     ← black plays *, captures white at (0,1)
    //   B B .     ← (1,1) also blocked so white has only 1 liberty
    Board<9> b2;
    place(b2, 0, 1, WHITE);
    place(b2, 1, 0, BLACK);
    place(b2, 0, 2, BLACK);
    place(b2, 1, 1, BLACK);  // block white's last escape at (1,1)
    // Now white at (0,1) has 1 liberty: (0,0)
    // Black plays (0,0): corner with 2 sides walled, but captures white → legal
    assert(legal(b2, 0, 0, BLACK));
    int cap = place(b2, 0, 0, BLACK);
    assert(cap == 1);
    assert(b2.color[Board<9>::pos(0, 1)] == EMPTY);

    printf("  PASS: capture not suicide\n");
}

// ─── Test 7: Ko detection ───────────────────────────────────────

void test_ko() {
    Board<9> b;
    // Classic ko pattern:
    //   . B W .
    //   B . B W
    //   . B W .
    // Black captures at (1,1), then white can't recapture immediately

    // Set up:
    //   row 0: . B W .
    //   row 1: B * B W    ← * is where action happens
    //   row 2: . B W .
    place(b, 0, 1, BLACK);
    place(b, 0, 2, WHITE);
    place(b, 1, 0, BLACK);
    place(b, 1, 2, BLACK);
    place(b, 1, 3, WHITE);
    place(b, 2, 1, BLACK);
    place(b, 2, 2, WHITE);

    // Place white at (1,1) first
    place(b, 1, 1, WHITE);
    // White at (1,1) has no liberties? Let's check...
    // Neighbors: (0,1)=B, (2,1)=B, (1,0)=B, (1,2)=B → all black, so white is captured
    // Hmm, that means black already captured it. Not the right setup.

    // Let me redo this. Classic ko:
    //   . B W .
    //   B W . W     ← white at (1,1), black captures by playing (1,2)
    //   . B W .
    // Wait, that's wrong too. Let me think about ko carefully.
    //
    // Ko setup on 9x9:
    //   col: 0 1 2 3
    // r0:   . B W .
    // r1:   B W . W
    // r2:   . B W .
    //
    // Black plays (1,2): captures white at (1,1) (white had 1 liberty at (1,2))
    // This creates ko: white can't recapture at (1,1) immediately.
    Board<9> b2;
    place(b2, 0, 1, BLACK);
    place(b2, 0, 2, WHITE);
    place(b2, 1, 0, BLACK);
    place(b2, 1, 1, WHITE);  // white stone to be captured
    place(b2, 1, 3, WHITE);
    place(b2, 2, 1, BLACK);
    place(b2, 2, 2, WHITE);

    // White at (1,1) liberties: (1,2) only (others are black or white)
    // Black plays (1,2): captures white at (1,1)
    int cap = place(b2, 1, 2, BLACK);
    assert(cap == 1);
    assert(b2.color[Board<9>::pos(1, 1)] == EMPTY);

    // Ko point should be set to (1,1)
    assert(b2.ko_point == Board<9>::pos(1, 1));

    // White can't recapture at (1,1)
    assert(!legal(b2, 1, 1, WHITE));

    // But white can play elsewhere
    assert(legal(b2, 4, 4, WHITE));

    printf("  PASS: ko detection\n");
}

// ─── Test 8: Ko resets after other move ─────────────────────────

void test_ko_reset() {
    Board<9> b;
    // Same ko setup
    place(b, 0, 1, BLACK);
    place(b, 0, 2, WHITE);
    place(b, 1, 0, BLACK);
    place(b, 1, 1, WHITE);
    place(b, 1, 3, WHITE);
    place(b, 2, 1, BLACK);
    place(b, 2, 2, WHITE);

    int cap = place(b, 1, 2, BLACK);
    assert(cap == 1);
    assert(b.ko_point == Board<9>::pos(1, 1));

    // White plays elsewhere → ko resets
    place(b, 5, 5, WHITE);
    // After any move, ko_point should be NO_KO (since no new ko was created)
    assert(b.ko_point == NO_KO);

    printf("  PASS: ko reset after other move\n");
}

// ─── Test 9: Zobrist hashing ────────────────────────────────────

void test_zobrist() {
    Board<9> b1;
    Board<9> b2;

    // Same moves, same order → same hash
    place(b1, 2, 3, BLACK);
    place(b1, 4, 5, WHITE);

    place(b2, 2, 3, BLACK);
    place(b2, 4, 5, WHITE);

    assert(b1.hash == b2.hash);

    // Different board → different hash (very likely)
    Board<9> b3;
    place(b3, 2, 3, BLACK);
    place(b3, 4, 4, WHITE);  // different position
    assert(b1.hash != b3.hash);

    printf("  PASS: Zobrist hashing\n");
}

// ─── Test 10: Scoring ───────────────────────────────────────────

void test_scoring() {
    Board<9> b;

    // Empty board: all territory is neutral (no stones to claim)
    auto [bs, ws] = b.score(7.5f);
    // All empty, no stones → all 81 points are dame (no border)
    // Actually on empty board, flood fill finds one big region with no borders
    // → neutral. So black=0, white=7.5
    assert(bs == 0.0f);
    assert(ws == 7.5f);

    // Simple test: black fills entire first row on 9x9
    // This creates a black wall — territory below is contested, above is nothing
    // Let's do a simpler test: one corner
    Board<9> b2;
    // Black occupies (0,0), (0,1), (1,0) on a 9x9
    // The corner territory isn't enclosed, so not really territory
    // Let's just verify stones are counted
    place(b2, 0, 0, BLACK);
    place(b2, 0, 1, BLACK);
    place(b2, 0, 2, BLACK);
    auto [bs2, ws2] = b2.score(0.0f);
    // 3 black stones, rest is one big empty region touching no white → all territory?
    // No — the empty region doesn't touch any white stones, so it borders only black
    // → all 78 empty cells = black territory
    assert(bs2 == 81.0f);  // 3 stones + 78 territory
    assert(ws2 == 0.0f);

    printf("  PASS: scoring\n");
}

// ─── Test 11: Legal moves list ──────────────────────────────────

void test_legal_moves() {
    Board<9> b;

    // Empty board: all 81 positions are legal for black
    auto moves = b.get_legal_moves(BLACK);
    assert(static_cast<int>(moves.size()) == 81);

    // Place one stone, 80 legal
    place(b, 0, 0, BLACK);
    moves = b.get_legal_moves(WHITE);
    assert(static_cast<int>(moves.size()) == 80);

    printf("  PASS: legal moves\n");
}

// ─── Test 12: Game with pass and scoring ────────────────────────

void test_game_pass() {
    Game<9> g(5.5f);
    assert(g.status == PLAYING);
    assert(g.current_turn == BLACK);

    // Both pass → game ends
    g.make_move(81);  // black pass (action = N*N = 81)
    assert(g.current_turn == WHITE);
    assert(g.status == PLAYING);

    g.make_move(81);  // white pass
    assert(g.status != PLAYING);  // game over

    printf("  PASS: game pass and end\n");
}

// ─── Test 13: Game move and capture ─────────────────────────────

void test_game_move() {
    Game<9> g(5.5f);

    // Black plays (2,3)
    int cap = g.make_move(Board<9>::pos(2, 3));
    assert(cap == 0);
    assert(g.current_turn == WHITE);
    assert(g.move_count == 1);
    assert(g.board.color[Board<9>::pos(2, 3)] == BLACK);

    printf("  PASS: game move\n");
}

// ─── Test 14: Observation encoding ──────────────────────────────

void test_observation() {
    Game<9> g(5.5f);

    // Place a few moves
    g.make_move(Board<9>::pos(2, 3));  // black
    g.make_move(Board<9>::pos(4, 5));  // white
    g.make_move(Board<9>::pos(6, 7));  // black

    // Now it's white's turn
    float obs[17 * 81];
    g.to_observation(obs);

    // Plane 0: current player (WHITE) stones
    // White stone at (4,5) → obs[0*81 + 4*9+5] should be 1.0
    assert(obs[0 * 81 + 4 * 9 + 5] == 1.0f);

    // Plane 1: opponent (BLACK) stones
    // Black stone at (2,3) → obs[1*81 + 2*9+3] should be 1.0
    assert(obs[1 * 81 + 2 * 9 + 3] == 1.0f);
    assert(obs[1 * 81 + 6 * 9 + 7] == 1.0f);  // black at (6,7)

    // Plane 16: color to play (WHITE → all 0.0)
    assert(obs[16 * 81] == 0.0f);

    printf("  PASS: observation encoding\n");
}

// ─── Test 15: Large board (19x19) ───────────────────────────────

void test_19x19() {
    Board<19> b;
    place(b, 3, 3, BLACK);
    place(b, 15, 15, WHITE);
    assert(b.color[Board<19>::pos(3, 3)] == BLACK);
    assert(b.color[Board<19>::pos(15, 15)] == WHITE);

    auto moves = b.get_legal_moves(BLACK);
    assert(static_cast<int>(moves.size()) == 359);  // 361 - 2

    printf("  PASS: 19x19 board\n");
}

// ─── Test 16: Random game simulation ────────────────────────────

void test_random_game() {
    std::mt19937 rng(42);
    int completed = 0;

    for (int trial = 0; trial < 100; ++trial) {
        Game<9> g(5.5f);
        for (int move = 0; move < 200 && g.status == PLAYING; ++move) {
            auto moves = g.board.get_legal_moves(g.current_turn);
            if (moves.empty() || (rng() % 100 < 5)) {
                g.make_move(81);  // pass
            } else {
                int action = moves[rng() % moves.size()];
                int cap = g.make_move(action);
                assert(cap >= 0);  // should always be legal since we got it from get_legal_moves
            }
        }
        if (g.status != PLAYING) completed++;
    }

    printf("  PASS: random game (%d/100 completed)\n", completed);
}

// ─── Test 17: Edge case — large group capture ───────────────────

void test_large_group_capture() {
    Board<9> b;
    // Build a white group in a line: (1,0), (1,1), (1,2), (1,3), (1,4)
    for (int c = 0; c < 5; ++c) {
        place(b, 1, c, WHITE);
    }
    // Surround with black
    for (int c = 0; c < 5; ++c) {
        place(b, 0, c, BLACK);
        place(b, 2, c, BLACK);
    }
    place(b, 1, 5, BLACK);
    // White group has 1 liberty at (1,-1)... wait, (1,0) neighbors are (0,0)=B, (2,0)=B, (1,1)=W.
    // The leftmost stone (1,0) has no liberty on the left (edge). Check chain liberties.
    // Liberties of the white chain: only empty neighbors. Let me check...
    // (1,0): neighbors (0,0)=B, (2,0)=B, (1,1)=W → no liberty
    // (1,1): neighbors (0,1)=B, (2,1)=B, (1,0)=W, (1,2)=W → no liberty
    // ... (1,4): neighbors (0,4)=B, (2,4)=B, (1,3)=W, (1,5)=B → no liberty
    // So the white group should already be captured when we place (1,5)!
    // Actually the captures happen as we place the surrounding black stones.
    // Let me check: after placing all, is white still there?

    // Let me verify by checking if white group was captured
    bool any_white = false;
    for (int c = 0; c < 5; ++c) {
        if (b.color[Board<9>::pos(1, c)] == WHITE) any_white = true;
    }

    // The white group should have been captured when its last liberty was filled
    assert(!any_white);

    printf("  PASS: large group capture\n");
}

// ─── Test 18: Chain merging ─────────────────────────────────────

void test_chain_merge() {
    Board<9> b;
    // Place two separate black stones
    place(b, 2, 2, BLACK);
    place(b, 2, 4, BLACK);
    // They should be in different chains
    assert(b.chain_id[Board<9>::pos(2, 2)] != b.chain_id[Board<9>::pos(2, 4)]);

    // Connect them
    place(b, 2, 3, BLACK);
    // Now all three should be in the same chain
    int16_t cid = b.chain_id[Board<9>::pos(2, 2)];
    assert(b.chain_id[Board<9>::pos(2, 3)] == cid);
    assert(b.chain_id[Board<9>::pos(2, 4)] == cid);

    printf("  PASS: chain merge\n");
}

// ─── Test 19: Action mask ───────────────────────────────────────

void test_action_mask() {
    Board<9> b;
    bool mask[82];
    b.get_action_mask(BLACK, mask);

    // All positions legal on empty board
    for (int i = 0; i < 81; ++i) assert(mask[i] == true);
    assert(mask[81] == true);  // pass

    place(b, 4, 4, BLACK);
    b.get_action_mask(WHITE, mask);
    assert(mask[Board<9>::pos(4, 4)] == false);  // occupied
    assert(mask[81] == true);  // pass still legal

    printf("  PASS: action mask\n");
}

// ─── Performance benchmark ──────────────────────────────────────

void bench_random_games() {
    printf("\n--- Benchmarks ---\n");

    // 9x9 random games (static buffer for legal moves)
    {
        std::mt19937 rng(123);
        auto start = std::chrono::high_resolution_clock::now();
        int games = 10000;
        int total_moves = 0;
        int legal_buf[81];
        for (int i = 0; i < games; ++i) {
            Game<9> g(5.5f);
            for (int move = 0; move < 200 && g.status == PLAYING; ++move) {
                int n_legal = g.board.get_legal_moves(g.current_turn, legal_buf);
                if (n_legal == 0 || (rng() % 100 < 3)) {
                    g.make_move(81);
                } else {
                    g.make_move(legal_buf[rng() % n_legal]);
                }
                total_moves++;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  9x9:  %d games, %d moves in %.1f ms (%.0f games/sec, %.2f us/move)\n",
               games, total_moves, ms, games / (ms / 1000.0), ms * 1000.0 / total_moves);
    }

    // 19x19 random games
    {
        std::mt19937 rng(456);
        auto start = std::chrono::high_resolution_clock::now();
        int games = 2000;
        int total_moves = 0;
        int legal_buf[361];
        for (int i = 0; i < games; ++i) {
            Game<19> g(7.5f);
            for (int move = 0; move < 500 && g.status == PLAYING; ++move) {
                int n_legal = g.board.get_legal_moves(g.current_turn, legal_buf);
                if (n_legal == 0 || (rng() % 100 < 3)) {
                    g.make_move(361);
                } else {
                    g.make_move(legal_buf[rng() % n_legal]);
                }
                total_moves++;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  19x19: %d games, %d moves in %.1f ms (%.0f games/sec, %.2f us/move)\n",
               games, total_moves, ms, games / (ms / 1000.0), ms * 1000.0 / total_moves);
    }

    // place_stone only benchmark (MCTS hot path — no get_legal_moves)
    {
        std::mt19937 rng(999);
        auto start = std::chrono::high_resolution_clock::now();
        int total_moves = 0;
        int legal_buf[361];
        for (int i = 0; i < 2000; ++i) {
            Board<19> b;
            uint8_t turn = BLACK;
            for (int move = 0; move < 300; ++move) {
                // Pick random empty position, try to place
                int empty[361];
                int n_empty = 0;
                for (int p = 0; p < 361; ++p) {
                    if (b.color[p] == EMPTY) empty[n_empty++] = p;
                }
                if (n_empty == 0) break;
                int p = empty[rng() % n_empty];
                int cap = b.place_stone(p, turn);
                if (cap >= 0) {
                    total_moves++;
                    turn = turn == BLACK ? WHITE : BLACK;
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  19x19 place_stone only: %d moves in %.1f ms (%.2f us/move)\n",
               total_moves, ms, ms * 1000.0 / total_moves);
    }

    // Board clone benchmark
    {
        Board<19> b;
        place(b, 3, 3, BLACK);
        place(b, 15, 15, WHITE);
        place(b, 10, 10, BLACK);

        auto start = std::chrono::high_resolution_clock::now();
        int iters = 1000000;
        volatile uint64_t sink = 0;
        for (int i = 0; i < iters; ++i) {
            Board<19> copy = b;
            sink += copy.hash;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start).count() / iters;
        printf("  Board<19> clone: %.0f ns (%.2f us)\n", ns, ns / 1000.0);
    }

    // is_legal benchmark
    {
        Board<19> b;
        // Place some stones for a realistic board
        std::mt19937 rng(789);
        for (int i = 0; i < 50; ++i) {
            auto moves = b.get_legal_moves((i % 2 == 0) ? BLACK : WHITE);
            if (!moves.empty()) {
                b.place_stone(moves[rng() % moves.size()], (i % 2 == 0) ? BLACK : WHITE);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        int iters = 100000;
        volatile int sink = 0;
        for (int i = 0; i < iters; ++i) {
            for (int p = 0; p < 361; ++p) {
                sink += b.is_legal(p, BLACK) ? 1 : 0;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start).count() / (iters * 361.0);
        printf("  is_legal<19> per position: %.0f ns (%.2f us)\n", ns, ns / 1000.0);
    }
}

// ─── Main ───────────────────────────────────────────────────────

int main() {
    printf("=== Go Engine C++ Tests ===\n\n");

    test_basic_placement();
    test_simple_capture();
    test_group_capture();
    test_corner_capture();
    test_suicide_prevention();
    test_capture_not_suicide();
    test_ko();
    test_ko_reset();
    test_zobrist();
    test_scoring();
    test_legal_moves();
    test_game_pass();
    test_game_move();
    test_observation();
    test_19x19();
    test_random_game();
    test_large_group_capture();
    test_chain_merge();
    test_action_mask();

    bench_random_games();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
