#pragma once
// High-performance Go engine with chain-based board representation.
// Template on board size N for compile-time optimization.

#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include "zobrist.h"

namespace go {

enum : uint8_t { EMPTY = 0, BLACK = 1, WHITE = 2 };
enum : int8_t { PLAYING = 0, BLACK_WIN = 1, WHITE_WIN = 2 };
constexpr int16_t NO_CHAIN = -1;
constexpr int16_t NO_KO = -1;
constexpr int PASS_ACTION = -1;

inline uint8_t opponent(uint8_t color) { return 3 - color; }

// ─── Board<N> ────────────────────────────────────────────────────────

template<int N>
struct Board {
    static constexpr int CELLS = N * N;
    static constexpr int MAX_CHAINS = CELLS;
    static constexpr int ACTIONS = CELLS + 1;

    // Per-cell
    uint8_t  color[CELLS];
    int16_t  chain_id[CELLS];
    int16_t  chain_next[CELLS];  // linked list within chain

    // Per-chain
    int16_t  chain_head[MAX_CHAINS];
    int16_t  chain_libs[MAX_CHAINS];   // liberty count
    int16_t  chain_size[MAX_CHAINS];

    // Free list
    int16_t  free_stack[MAX_CHAINS];
    int16_t  free_count;

    int16_t  ko_point;
    uint64_t hash;

    // Generation-stamped visited array (no zeroing needed)
    mutable uint32_t visited[CELLS];
    mutable uint32_t gen;

    // ─── Init ────────────────────────────────────────────────────

    Board() { reset(); }

    void reset() {
        std::memset(color, EMPTY, sizeof(color));
        std::memset(chain_id, 0xFF, sizeof(chain_id));
        std::memset(chain_next, 0xFF, sizeof(chain_next));
        std::memset(chain_head, 0xFF, sizeof(chain_head));
        std::memset(chain_libs, 0, sizeof(chain_libs));
        std::memset(chain_size, 0, sizeof(chain_size));
        std::memset(visited, 0, sizeof(visited));
        gen = 0;
        for (int16_t i = 0; i < MAX_CHAINS; ++i)
            free_stack[i] = MAX_CHAINS - 1 - i;
        free_count = MAX_CHAINS;
        ko_point = NO_KO;
        hash = 0;
    }

    // ─── Coordinates ────────────────────────────────────────────

    static constexpr int pos(int row, int col) { return row * N + col; }
    static constexpr int row(int p) { return p / N; }
    static constexpr int col(int p) { return p % N; }

    int neighbors(int p, int out[4]) const {
        int r = p / N, c = p % N, n = 0;
        if (r > 0)     out[n++] = p - N;
        if (r < N - 1) out[n++] = p + N;
        if (c > 0)     out[n++] = p - 1;
        if (c < N - 1) out[n++] = p + 1;
        return n;
    }

    // ─── Chain helpers ──────────────────────────────────────────

    int16_t alloc_chain() { return free_stack[--free_count]; }

    void free_chain_id(int16_t id) {
        free_stack[free_count++] = id;
        chain_head[id] = -1;
        chain_size[id] = 0;
        chain_libs[id] = 0;
    }

    // Exact liberty count via chain walk + generation stamp
    int recount_liberties(int16_t cid) const {
        ++gen;
        int libs = 0;
        int16_t s = chain_head[cid];
        while (s != -1) {
            int nbrs[4];
            int nn = neighbors(s, nbrs);
            for (int i = 0; i < nn; ++i) {
                int nb = nbrs[i];
                if (color[nb] == EMPTY && visited[nb] != gen) {
                    visited[nb] = gen;
                    libs++;
                }
            }
            s = chain_next[s];
        }
        return libs;
    }

    // Merge src into dst. Does NOT recount liberties.
    void merge_chains(int16_t dst, int16_t src) {
        if (dst == src) return;
        int16_t s = chain_head[src];
        int16_t tail = -1;
        while (s != -1) {
            chain_id[s] = dst;
            tail = s;
            s = chain_next[s];
        }
        chain_next[tail] = chain_head[dst];
        chain_head[dst] = chain_head[src];
        chain_size[dst] += chain_size[src];
        free_chain_id(src);
    }

    // Remove chain from board. Updates adjacent chains' liberties.
    int remove_chain(int16_t cid) {
        const auto& ztable = zobrist<N>();

        // Pass 1: collect unique adjacent chain IDs (small array dedup)
        int16_t adj[64];
        int n_adj = 0;
        {
            int16_t s = chain_head[cid];
            while (s != -1) {
                int nbrs[4];
                int nn = neighbors(s, nbrs);
                for (int i = 0; i < nn; ++i) {
                    int nb = nbrs[i];
                    if (color[nb] != EMPTY && chain_id[nb] != cid) {
                        int16_t ac = chain_id[nb];
                        // Linear scan dedup (n_adj is typically < 10)
                        bool found = false;
                        for (int j = 0; j < n_adj; ++j) {
                            if (adj[j] == ac) { found = true; break; }
                        }
                        if (!found) adj[n_adj++] = ac;
                    }
                }
                s = chain_next[s];
            }
        }

        // Pass 2: remove stones
        int removed = 0;
        int16_t s = chain_head[cid];
        while (s != -1) {
            int16_t next = chain_next[s];
            hash ^= ztable.stone[color[s] - 1][s];
            color[s] = EMPTY;
            chain_id[s] = NO_CHAIN;
            chain_next[s] = -1;
            removed++;
            s = next;
        }
        free_chain_id(cid);

        // Recount only the small set of affected adjacent chains
        for (int i = 0; i < n_adj; ++i) {
            chain_libs[adj[i]] = recount_liberties(adj[i]);
        }

        return removed;
    }

    // ─── Place stone ────────────────────────────────────────────

    int place_stone(int p, uint8_t clr) {
        if (!is_legal(p, clr)) return -1;

        const auto& ztable = zobrist<N>();

        // Place stone
        color[p] = clr;
        hash ^= ztable.stone[clr - 1][p];

        // Create singleton chain
        int16_t my_chain = alloc_chain();
        chain_id[p] = my_chain;
        chain_head[my_chain] = p;
        chain_next[p] = -1;
        chain_size[my_chain] = 1;
        chain_libs[my_chain] = 0;  // recount after merges

        int nbrs[4];
        int nn = neighbors(p, nbrs);
        uint8_t opp = opponent(clr);

        // Merge with friendly neighbors
        for (int i = 0; i < nn; ++i) {
            int nb = nbrs[i];
            if (color[nb] == clr && chain_id[nb] != my_chain) {
                merge_chains(chain_id[nb], my_chain);
                my_chain = chain_id[p];
            }
        }

        // ONE recount for our merged chain
        chain_libs[my_chain] = recount_liberties(my_chain);

        // Opponent chains: p was a liberty, now it's filled → decrement by 1.
        // If liberty hits 0, capture. Dedup by chain ID.
        int total_captured = 0;
        int16_t captured_pos = NO_KO;
        int16_t checked[4];
        int n_checked = 0;

        for (int i = 0; i < nn; ++i) {
            int nb = nbrs[i];
            if (color[nb] != opp) continue;
            int16_t opp_cid = chain_id[nb];

            bool dup = false;
            for (int j = 0; j < n_checked; ++j) {
                if (checked[j] == opp_cid) { dup = true; break; }
            }
            if (dup) continue;
            checked[n_checked++] = opp_cid;

            // Decrement: p was their liberty, now occupied
            chain_libs[opp_cid]--;

            if (chain_libs[opp_cid] == 0) {
                if (chain_size[opp_cid] == 1) captured_pos = chain_head[opp_cid];
                total_captured += remove_chain(opp_cid);
            }
        }

        // After captures, recount our chain (freed positions = new liberties)
        if (total_captured > 0) {
            my_chain = chain_id[p];
            chain_libs[my_chain] = recount_liberties(my_chain);
        }

        // Ko: all 3 conditions
        int16_t new_ko = NO_KO;
        if (total_captured == 1 && chain_size[my_chain] == 1 && chain_libs[my_chain] == 1) {
            new_ko = captured_pos;
        }
        ko_point = new_ko;

        return total_captured;
    }

    // ─── Is Legal (O(4)) ────────────────────────────────────────

    bool is_legal(int p, uint8_t clr) const {
        if (color[p] != EMPTY) return false;
        if (p == ko_point) return false;

        uint8_t opp = opponent(clr);
        int nbrs[4];
        int nn = neighbors(p, nbrs);

        for (int i = 0; i < nn; ++i) {
            int nb = nbrs[i];
            if (color[nb] == opp && chain_libs[chain_id[nb]] == 1)
                return true;  // captures
        }

        for (int i = 0; i < nn; ++i) {
            if (color[nbrs[i]] == EMPTY) return true;  // liberty
        }

        for (int i = 0; i < nn; ++i) {
            int nb = nbrs[i];
            if (color[nb] == clr && chain_libs[chain_id[nb]] >= 2)
                return true;
        }

        return false;
    }

    // Bounds-checked version for external API
    bool is_legal_checked(int p, uint8_t clr) const {
        if (p < 0 || p >= CELLS) return false;
        return is_legal(p, clr);
    }

    // ─── Legal moves ────────────────────────────────────────────

    std::vector<int> get_legal_moves(uint8_t clr) const {
        std::vector<int> moves;
        moves.reserve(CELLS);
        for (int p = 0; p < CELLS; ++p) {
            if (is_legal(p, clr)) moves.push_back(p);
        }
        return moves;
    }

    // Fill pre-allocated array. Returns count.
    int get_legal_moves(uint8_t clr, int* out) const {
        int n = 0;
        for (int p = 0; p < CELLS; ++p) {
            if (is_legal(p, clr)) out[n++] = p;
        }
        return n;
    }

    void get_action_mask(uint8_t clr, bool* mask) const {
        for (int p = 0; p < CELLS; ++p) mask[p] = is_legal(p, clr);
        mask[CELLS] = true;
    }

    // ─── Scoring (Tromp-Taylor) ─────────────────────────────────

    std::pair<float, float> score(float komi) const {
        int black = 0, white = 0;
        for (int p = 0; p < CELLS; ++p) {
            if (color[p] == BLACK) black++;
            else if (color[p] == WHITE) white++;
        }

        ++gen;
        int stack[CELLS];
        for (int p = 0; p < CELLS; ++p) {
            if (color[p] != EMPTY || visited[p] == gen) continue;

            int region_size = 0;
            uint8_t borders = 0;
            int sp = 0;
            stack[sp++] = p;
            visited[p] = gen;

            while (sp > 0) {
                int cur = stack[--sp];
                region_size++;
                int nbrs[4];
                int nn = neighbors(cur, nbrs);
                for (int i = 0; i < nn; ++i) {
                    int nb = nbrs[i];
                    if (color[nb] == EMPTY) {
                        if (visited[nb] != gen) {
                            visited[nb] = gen;
                            stack[sp++] = nb;
                        }
                    } else {
                        borders |= (1 << (color[nb] - 1));
                    }
                }
            }

            if (borders == 1) black += region_size;
            else if (borders == 2) white += region_size;
        }

        return {static_cast<float>(black), static_cast<float>(white) + komi};
    }

    void to_grid(uint8_t* out) const {
        std::memcpy(out, color, CELLS);
    }

    // Per-cell Tromp-Taylor ownership at the current board state.
    // Writes one int8 per cell into `out`:
    //   +1 = owned by BLACK (stone or BLACK-only-bordered empty region)
    //   -1 = owned by WHITE
    //    0 = dame (empty region bordered by both colors)
    //
    // This mirrors the flood-fill loop in `score()` but records per-cell
    // labels instead of accumulating totals. Used by the KataGo-style
    // ownership auxiliary head added in run 4: each self-play position
    // gets 169 (= 13×13) per-cell ownership labels at game end as
    // dense supervision targets, vs the single scalar value label per
    // ~150-move game that is too sparse to train a 4.5M-param net on.
    // See PHASE_TWO_TRAINING.md run3 + ownership-head section.
    void compute_ownership(int8_t* out) const {
        // Stones own themselves.
        for (int p = 0; p < CELLS; ++p) {
            if (color[p] == BLACK) out[p] = 1;
            else if (color[p] == WHITE) out[p] = -1;
            else out[p] = 0;  // overwritten below if region is single-color
        }

        // Flood-fill empty regions, label by bordering color(s).
        ++gen;
        int stack[CELLS];
        int region_cells[CELLS];
        for (int p = 0; p < CELLS; ++p) {
            if (color[p] != EMPTY || visited[p] == gen) continue;

            int region_n = 0;
            uint8_t borders = 0;
            int sp = 0;
            stack[sp++] = p;
            visited[p] = gen;

            while (sp > 0) {
                int cur = stack[--sp];
                region_cells[region_n++] = cur;
                int nbrs[4];
                int nn = neighbors(cur, nbrs);
                for (int i = 0; i < nn; ++i) {
                    int nb = nbrs[i];
                    if (color[nb] == EMPTY) {
                        if (visited[nb] != gen) {
                            visited[nb] = gen;
                            stack[sp++] = nb;
                        }
                    } else {
                        borders |= (1 << (color[nb] - 1));
                    }
                }
            }

            int8_t label;
            if (borders == 1) label = 1;       // BLACK only
            else if (borders == 2) label = -1; // WHITE only
            else label = 0;                    // dame (both or neither)
            for (int i = 0; i < region_n; ++i) out[region_cells[i]] = label;
        }
    }
};

// ─── Game<N> ──────────────────────────────────────────────────────

template<int N>
struct Game {
    static constexpr int CELLS = N * N;
    static constexpr int ACTIONS = CELLS + 1;

    Board<N> board;
    uint8_t history[8][CELLS];
    int history_len;
    int8_t current_turn;
    int8_t status;
    int consecutive_passes;
    int move_count;
    int captured[3];
    float komi;

    Game(float komi_ = 7.5f) : history_len(0), current_turn(BLACK), status(PLAYING),
                                 consecutive_passes(0), move_count(0), komi(komi_) {
        std::memset(history, 0, sizeof(history));
        captured[0] = 0;
        captured[BLACK] = 0;
        captured[WHITE] = 0;
    }

    void push_history() {
        if (history_len < 8) history_len++;
        for (int i = 7; i > 0; --i)
            std::memcpy(history[i], history[i - 1], CELLS);
        board.to_grid(history[0]);
    }

    int make_move(int action) {
        if (status != PLAYING) return -1;

        if (action == CELLS) {
            push_history();
            consecutive_passes++;
            board.ko_point = NO_KO;
            if (consecutive_passes >= 2) end_game();
            current_turn = opponent(current_turn);
            move_count++;
            return 0;
        }

        push_history();
        int cap = board.place_stone(action, current_turn);
        if (cap < 0) {
            history_len = (history_len > 0) ? history_len - 1 : 0;
            for (int i = 0; i < 7; ++i)
                std::memcpy(history[i], history[i + 1], CELLS);
            return -1;
        }

        captured[current_turn] += cap;
        consecutive_passes = 0;
        current_turn = opponent(current_turn);
        move_count++;
        return cap;
    }

    void end_game() {
        auto [bs, ws] = board.score(komi);
        status = (bs > ws) ? BLACK_WIN : WHITE_WIN;
    }

    void resign(uint8_t clr) {
        status = (clr == BLACK) ? WHITE_WIN : BLACK_WIN;
    }

    void to_observation(float* obs) const {
        const int ps = CELLS;
        std::memset(obs, 0, 17 * ps * sizeof(float));

        uint8_t me = current_turn;
        uint8_t opp = opponent(me);

        for (int p = 0; p < CELLS; ++p) {
            if (board.color[p] == me)  obs[p] = 1.0f;
            if (board.color[p] == opp) obs[ps + p] = 1.0f;
        }

        for (int t = 0; t < 7 && t < history_len; ++t) {
            float* base_me  = obs + (t + 1) * 2 * ps;
            float* base_opp = base_me + ps;
            for (int p = 0; p < CELLS; ++p) {
                if (history[t][p] == me)  base_me[p] = 1.0f;
                if (history[t][p] == opp) base_opp[p] = 1.0f;
            }
        }

        if (me == BLACK) {
            float* plane16 = obs + 16 * ps;
            for (int p = 0; p < CELLS; ++p) plane16[p] = 1.0f;
        }
    }
};

extern template struct Board<9>;
extern template struct Board<13>;
extern template struct Board<19>;
extern template struct Game<9>;
extern template struct Game<13>;
extern template struct Game<19>;

}  // namespace go
