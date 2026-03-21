/*
 * nnue_search.h — NNUE evaluation + alpha-beta search for Xiangqi.
 *
 * NNUE architecture:
 *   Per-perspective features: 1260 binary inputs (2 colors × 7 types × 90 squares)
 *   Accumulator: Linear(1260, 128) shared between perspectives
 *   Output: concat(128, 128) -> Dense(256,32) -> Dense(32,32) -> Dense(32,1) -> sigmoid
 *   All hidden activations: ClippedReLU [0, 1]
 *
 * Search:
 *   Negamax alpha-beta with:
 *   - MVV-LVA move ordering
 *   - Quiescence search (captures only)
 *   - Transposition table
 */

#pragma once

#include "xiangqi.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace nnue {

// ─── Architecture constants ─────────────────────────────────────────────────
constexpr int NUM_PIECE_TYPES = 7;
constexpr int NUM_SQUARES = 90;
constexpr int NUM_COLORS = 2;
constexpr int FEATURES = NUM_COLORS * NUM_PIECE_TYPES * NUM_SQUARES;  // 1260
constexpr int ACC_SIZE = 128;
constexpr int H1_SIZE = 32;
constexpr int H2_SIZE = 32;

// ─── NNUE weights ───────────────────────────────────────────────────────────
struct NNUEWeights {
    // Accumulator: (ACC_SIZE, FEATURES) weight + (ACC_SIZE) bias
    float acc_weight[ACC_SIZE][FEATURES];
    float acc_bias[ACC_SIZE];

    // FC1: (H1_SIZE, ACC_SIZE*2) weight + (H1_SIZE) bias
    float fc1_weight[H1_SIZE][ACC_SIZE * 2];
    float fc1_bias[H1_SIZE];

    // FC2: (H2_SIZE, H1_SIZE) weight + (H2_SIZE) bias
    float fc2_weight[H2_SIZE][H1_SIZE];
    float fc2_bias[H2_SIZE];

    // FC out: (1, H2_SIZE) weight + (1) bias
    float out_weight[H2_SIZE];
    float out_bias;
};

// ─── NNUE accumulator (cached per position) ─────────────────────────────────
struct Accumulator {
    float stm[ACC_SIZE];
    float nstm[ACC_SIZE];
};

// ─── Transposition Table ────────────────────────────────────────────────────
constexpr int TT_SIZE = 1 << 20;  // ~1M entries, ~24 MB

enum TTFlag : uint8_t { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };

struct TTEntry {
    uint64_t key;
    float value;
    int16_t depth;
    TTFlag flag;
    uint8_t best_move_idx;  // index into legal move list (not reliable across positions)
};

// ─── Search statistics ──────────────────────────────────────────────────────
struct SearchStats {
    int64_t nodes;
    int64_t qnodes;
    int64_t tt_hits;
};

// ─── Piece values for MVV-LVA ───────────────────────────────────────────────
inline int piece_value(int8_t pt) {
    switch (pt) {
        case xiangqi::GENERAL:  return 10000;
        case xiangqi::CHARIOT:  return 900;
        case xiangqi::CANNON:   return 450;
        case xiangqi::HORSE:    return 400;
        case xiangqi::ADVISOR:  return 200;
        case xiangqi::ELEPHANT: return 200;
        case xiangqi::SOLDIER:  return 100;
        default: return 0;
    }
}

// ─── Zobrist hashing ────────────────────────────────────────────────────────
struct Zobrist {
    uint64_t piece_sq[15][90];  // piece+7 (range 0-14), square
    uint64_t side;

    void init() {
        // Simple PRNG seed
        uint64_t s = 0x12345678ABCDEF01ULL;
        auto next = [&s]() -> uint64_t {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            return s;
        };
        for (int p = 0; p < 15; ++p)
            for (int sq = 0; sq < 90; ++sq)
                piece_sq[p][sq] = next();
        side = next();
    }

    uint64_t hash(const xiangqi::Board& b, int8_t stm) const {
        uint64_t h = 0;
        for (int sq = 0; sq < 90; ++sq) {
            int8_t piece = b.grid[sq];
            if (piece != 0) h ^= piece_sq[piece + 7][sq];
        }
        if (stm == xiangqi::BLACK) h ^= side;
        return h;
    }
};

// ─── NNUE Search Engine ─────────────────────────────────────────────────────
class NNUESearch {
public:
    NNUEWeights weights;
    Zobrist zobrist;
    TTEntry* tt;
    SearchStats stats;

    NNUESearch() : tt(nullptr) {
        zobrist.init();
        tt = new TTEntry[TT_SIZE];
        clear_tt();
    }

    ~NNUESearch() {
        delete[] tt;
    }

    void clear_tt() {
        std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
    }

    // ─── Weight loading ─────────────────────────────────────────────────
    bool load_weights(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;

        // Read in exact order matching export script
        fread(weights.acc_weight, sizeof(float), ACC_SIZE * FEATURES, f);
        fread(weights.acc_bias, sizeof(float), ACC_SIZE, f);
        fread(weights.fc1_weight, sizeof(float), H1_SIZE * ACC_SIZE * 2, f);
        fread(weights.fc1_bias, sizeof(float), H1_SIZE, f);
        fread(weights.fc2_weight, sizeof(float), H2_SIZE * H1_SIZE, f);
        fread(weights.fc2_bias, sizeof(float), H2_SIZE, f);
        fread(weights.out_weight, sizeof(float), H2_SIZE, f);
        fread(&weights.out_bias, sizeof(float), 1, f);

        fclose(f);
        return true;
    }

    // ─── NNUE evaluation ────────────────────────────────────────────────

    // Compute accumulator from scratch for given board and side-to-move
    void compute_accumulator(const xiangqi::Board& b, int8_t stm,
                             float stm_acc[ACC_SIZE], float nstm_acc[ACC_SIZE]) const {
        // Start with bias
        std::memcpy(stm_acc, weights.acc_bias, ACC_SIZE * sizeof(float));
        std::memcpy(nstm_acc, weights.acc_bias, ACC_SIZE * sizeof(float));

        for (int sq = 0; sq < 90; ++sq) {
            int8_t piece = b.grid[sq];
            if (piece == 0) continue;

            int8_t pc = xiangqi::color_of(piece);
            int type_idx = xiangqi::piece_type(piece) - 1;  // 0-6

            // STM perspective
            int stm_color_idx = (pc == stm) ? 0 : 1;
            int stm_feat = stm_color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + sq;
            for (int i = 0; i < ACC_SIZE; ++i)
                stm_acc[i] += weights.acc_weight[i][stm_feat];

            // NSTM perspective (mirrored)
            int mirror_sq = 89 - sq;
            int nstm_color_idx = (pc == stm) ? 1 : 0;
            int nstm_feat = nstm_color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + mirror_sq;
            for (int i = 0; i < ACC_SIZE; ++i)
                nstm_acc[i] += weights.acc_weight[i][nstm_feat];
        }
    }

    // Material evaluation from STM perspective, normalized to [0, 1]
    static float material_eval(const xiangqi::Board& b, int8_t stm) {
        float my_mat = 0.0f, opp_mat = 0.0f;
        for (int sq = 0; sq < 90; ++sq) {
            int8_t piece = b.grid[sq];
            if (piece == 0) continue;
            float val = 0.0f;
            switch (xiangqi::piece_type(piece)) {
                case xiangqi::CHARIOT:  val = 9.0f; break;
                case xiangqi::CANNON:   val = 4.5f; break;
                case xiangqi::HORSE:    val = 4.0f; break;
                case xiangqi::ADVISOR:  val = 2.0f; break;
                case xiangqi::ELEPHANT: val = 2.0f; break;
                case xiangqi::SOLDIER:  val = 1.0f; break;
                default: break;
            }
            if (xiangqi::color_of(piece) == stm)
                my_mat += val;
            else
                opp_mat += val;
        }
        // Normalize: material advantage of ~45 (all pieces) maps to [0, 1]
        // sigmoid-like scaling: 0.5 + diff / (2 * max_diff)
        float diff = my_mat - opp_mat;
        return 0.5f + diff / 90.0f;  // total material ~45 per side
    }

    // Full NNUE forward pass, returns eval in [0, 1] from STM perspective
    float evaluate_nnue(const xiangqi::Board& b, int8_t stm) const {
        // Accumulator
        float stm_acc[ACC_SIZE], nstm_acc[ACC_SIZE];
        compute_accumulator(b, stm, stm_acc, nstm_acc);

        // ClippedReLU on accumulators
        for (int i = 0; i < ACC_SIZE; ++i) {
            stm_acc[i] = std::min(1.0f, std::max(0.0f, stm_acc[i]));
            nstm_acc[i] = std::min(1.0f, std::max(0.0f, nstm_acc[i]));
        }

        // FC1: input = concat(stm_acc, nstm_acc) -> (H1_SIZE)
        float h1[H1_SIZE];
        for (int i = 0; i < H1_SIZE; ++i) {
            float sum = weights.fc1_bias[i];
            for (int j = 0; j < ACC_SIZE; ++j)
                sum += weights.fc1_weight[i][j] * stm_acc[j];
            for (int j = 0; j < ACC_SIZE; ++j)
                sum += weights.fc1_weight[i][ACC_SIZE + j] * nstm_acc[j];
            h1[i] = std::min(1.0f, std::max(0.0f, sum));  // ClippedReLU
        }

        // FC2: h1 -> (H2_SIZE)
        float h2[H2_SIZE];
        for (int i = 0; i < H2_SIZE; ++i) {
            float sum = weights.fc2_bias[i];
            for (int j = 0; j < H1_SIZE; ++j)
                sum += weights.fc2_weight[i][j] * h1[j];
            h2[i] = std::min(1.0f, std::max(0.0f, sum));  // ClippedReLU
        }

        // Output: h2 -> scalar, sigmoid
        float out = weights.out_bias;
        for (int j = 0; j < H2_SIZE; ++j)
            out += weights.out_weight[j] * h2[j];

        return 1.0f / (1.0f + std::exp(-out));  // sigmoid
    }

    // Blended evaluation: NNUE (positional) + Material (concrete)
    // NNUE_WEIGHT controls the mix: 0.0 = pure material, 1.0 = pure NNUE
    static constexpr float NNUE_WEIGHT = 0.6f;

    float evaluate(const xiangqi::Board& b, int8_t stm) const {
        float nnue_val = evaluate_nnue(b, stm);
        float mat_val = material_eval(b, stm);
        return NNUE_WEIGHT * nnue_val + (1.0f - NNUE_WEIGHT) * mat_val;
    }

    // ─── Move ordering ──────────────────────────────────────────────────

    // MVV-LVA score for move ordering
    static int mvv_lva_score(xiangqi::Move m, const xiangqi::Board& b) {
        if (m.captured == 0) return 0;
        int victim = piece_value(xiangqi::piece_type(m.captured));
        int attacker = piece_value(xiangqi::piece_type(b.grid[m.from_sq]));
        return victim * 100 - attacker;  // MVV-LVA: high victim, low attacker = good
    }

    // Sort moves: captures first (MVV-LVA), then non-captures
    static void order_moves(xiangqi::MoveList& ml, const xiangqi::Board& b) {
        // Simple insertion sort (move lists are small)
        for (int i = 1; i < ml.count; ++i) {
            xiangqi::Move key = ml.moves[i];
            int key_score = mvv_lva_score(key, b);
            int j = i - 1;
            while (j >= 0 && mvv_lva_score(ml.moves[j], b) < key_score) {
                ml.moves[j + 1] = ml.moves[j];
                j--;
            }
            ml.moves[j + 1] = key;
        }
    }

    // ─── Make/Unmake with proper undo stack ─────────────────────────────

    struct UndoInfo {
        int8_t prev_red_gen;
        int8_t prev_black_gen;
    };

    static void do_move(xiangqi::Board& b, xiangqi::Move m, UndoInfo& undo) {
        undo.prev_red_gen = b.red_general;
        undo.prev_black_gen = b.black_general;

        int8_t piece = b.grid[m.from_sq];
        b.grid[m.to_sq] = piece;
        b.grid[m.from_sq] = 0;

        if (piece == xiangqi::GENERAL)       b.red_general = m.to_sq;
        else if (piece == -xiangqi::GENERAL)  b.black_general = m.to_sq;
        if (m.captured == xiangqi::GENERAL)       b.red_general = -1;
        else if (m.captured == -xiangqi::GENERAL) b.black_general = -1;
    }

    static void undo_move(xiangqi::Board& b, xiangqi::Move m, const UndoInfo& undo) {
        int8_t piece = b.grid[m.to_sq];
        b.grid[m.from_sq] = piece;
        b.grid[m.to_sq] = m.captured;
        b.red_general = undo.prev_red_gen;
        b.black_general = undo.prev_black_gen;
    }

    // ─── Quiescence search ──────────────────────────────────────────────

    float quiescence(xiangqi::Board& b, int8_t stm, float alpha, float beta, int qdepth) {
        stats.qnodes++;

        float stand_pat = evaluate(b, stm);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;

        if (qdepth <= 0) return alpha;

        // Generate legal moves, only consider captures
        xiangqi::MoveList legal;
        xiangqi::get_legal_moves(b, stm, legal);
        order_moves(legal, b);

        for (int i = 0; i < legal.count; ++i) {
            xiangqi::Move m = legal.moves[i];
            if (m.captured == 0) continue;  // Skip non-captures

            // Capture of general = instant win
            if (xiangqi::piece_type(m.captured) == xiangqi::GENERAL)
                return 1.0f;

            UndoInfo undo;
            do_move(b, m, undo);
            float val = 1.0f - quiescence(b, -stm, 1.0f - beta, 1.0f - alpha, qdepth - 1);
            undo_move(b, m, undo);

            if (val >= beta) return beta;
            if (val > alpha) alpha = val;
        }

        return alpha;
    }

    // ─── Alpha-beta (negamax) ───────────────────────────────────────────

    float negamax(xiangqi::Board& b, int8_t stm, int depth, float alpha, float beta) {
        stats.nodes++;

        // TT probe
        uint64_t key = zobrist.hash(b, stm);
        TTEntry& entry = tt[key & (TT_SIZE - 1)];
        if (entry.key == key && entry.depth >= depth) {
            stats.tt_hits++;
            if (entry.flag == TT_EXACT) return entry.value;
            if (entry.flag == TT_BETA && entry.value >= beta) return beta;
            if (entry.flag == TT_ALPHA && entry.value <= alpha) return alpha;
        }

        // Leaf node: quiescence search
        if (depth <= 0) {
            return quiescence(b, stm, alpha, beta, 6);
        }

        xiangqi::MoveList legal;
        xiangqi::get_legal_moves(b, stm, legal);

        if (legal.count == 0) {
            return 0.0f;  // No legal moves = loss
        }

        order_moves(legal, b);

        float best_val = -1.0f;
        TTFlag flag = TT_ALPHA;

        for (int i = 0; i < legal.count; ++i) {
            xiangqi::Move m = legal.moves[i];

            // Capture general = instant win
            if (xiangqi::piece_type(m.captured) == xiangqi::GENERAL)
                return 1.0f;

            UndoInfo undo;
            do_move(b, m, undo);
            float val = 1.0f - negamax(b, -stm, depth - 1, 1.0f - beta, 1.0f - alpha);
            undo_move(b, m, undo);

            if (val > best_val) best_val = val;
            if (val > alpha) {
                alpha = val;
                flag = TT_EXACT;
            }
            if (alpha >= beta) {
                flag = TT_BETA;
                break;
            }
        }

        // TT store
        if (entry.key != key || entry.depth <= depth) {
            entry.key = key;
            entry.value = best_val;
            entry.depth = depth;
            entry.flag = flag;
        }

        return best_val;
    }

    // ─── Root search ────────────────────────────────────────────────────

    struct SearchResult {
        xiangqi::Move best_move;
        float score;
        int depth;
    };

    SearchResult search(xiangqi::Board& b, int8_t stm, int max_depth) {
        stats = {0, 0, 0};
        clear_tt();

        SearchResult result;
        result.best_move = {0, 0, 0};
        result.score = 0.0f;
        result.depth = 0;

        // Iterative deepening
        for (int depth = 1; depth <= max_depth; ++depth) {
            xiangqi::MoveList legal;
            xiangqi::get_legal_moves(b, stm, legal);

            if (legal.count == 0) break;
            order_moves(legal, b);

            float alpha = 0.0f, beta = 1.0f;
            float best_val = -1.0f;
            xiangqi::Move best_move = legal.moves[0];

            for (int i = 0; i < legal.count; ++i) {
                xiangqi::Move m = legal.moves[i];

                if (xiangqi::piece_type(m.captured) == xiangqi::GENERAL) {
                    result.best_move = m;
                    result.score = 1.0f;
                    result.depth = depth;
                    return result;
                }

                UndoInfo undo;
                do_move(b, m, undo);
                float val = 1.0f - negamax(b, -stm, depth - 1, 1.0f - beta, 1.0f - alpha);
                undo_move(b, m, undo);

                if (val > best_val) {
                    best_val = val;
                    best_move = m;
                }
                if (val > alpha) alpha = val;
            }

            result.best_move = best_move;
            result.score = best_val;
            result.depth = depth;
        }

        return result;
    }
};

}  // namespace nnue
