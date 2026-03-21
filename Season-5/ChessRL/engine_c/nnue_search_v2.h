/*
 * nnue_search_v2.h — NNUE v2 evaluation + alpha-beta search for Xiangqi.
 *
 * Changes from v1:
 *   - 692 features (piece-aware square mapping) instead of 1260
 *   - Board normalized so perspective's pieces are always at bottom
 *   - Feature map table loaded from binary file
 *   - Pure NNUE eval (no material blending) with configurable weight
 *
 * Search: same negamax alpha-beta with MVV-LVA, quiescence, TT.
 */

#pragma once

#include "xiangqi.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace nnue_v2 {

// ─── Architecture constants ─────────────────────────────────────────────────
constexpr int MAX_FEATURES = 692;
constexpr int ACC_SIZE = 128;
constexpr int H1_SIZE = 32;
constexpr int H2_SIZE = 32;

// ─── Feature map: (color_idx, piece_type, square) -> feature index ──────────
// Loaded from binary file. -1 means invalid (piece can't be on that square).
struct FeatureMap {
    int32_t sq_to_feat[2][8][90];  // [color_idx][piece_type][square]
    int32_t num_features;           // should be 692
};

// ─── NNUE weights ───────────────────────────────────────────────────────────
struct NNUEWeightsV2 {
    int num_features;  // 692

    // Accumulator: (ACC_SIZE, num_features) weight + (ACC_SIZE) bias
    float acc_weight[ACC_SIZE][MAX_FEATURES];
    float acc_bias[ACC_SIZE];

    // FC1: (H1_SIZE, ACC_SIZE*2)
    float fc1_weight[H1_SIZE][ACC_SIZE * 2];
    float fc1_bias[H1_SIZE];

    // FC2: (H2_SIZE, H1_SIZE)
    float fc2_weight[H2_SIZE][H1_SIZE];
    float fc2_bias[H2_SIZE];

    // Output: (1, H2_SIZE)
    float out_weight[H2_SIZE];
    float out_bias;
};

// ─── Transposition Table ────────────────────────────────────────────────────
constexpr int TT_SIZE = 1 << 20;

enum TTFlag : uint8_t { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };

struct TTEntry {
    uint64_t key;
    float value;
    int16_t depth;
    TTFlag flag;
    uint8_t pad;
};

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
    uint64_t piece_sq[15][90];
    uint64_t side;

    void init() {
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

// ─── NNUE v2 Search Engine ──────────────────────────────────────────────────
class NNUESearchV2 {
public:
    FeatureMap feat_map;
    NNUEWeightsV2 weights;
    Zobrist zobrist;
    TTEntry* tt;
    SearchStats stats;

    // NNUE blend weight: 1.0 = pure NNUE, 0.0 = pure material
    float nnue_weight;

    NNUESearchV2() : tt(nullptr), nnue_weight(1.0f) {
        zobrist.init();
        tt = new TTEntry[TT_SIZE];
        clear_tt();
        std::memset(&feat_map, 0, sizeof(feat_map));
    }

    ~NNUESearchV2() {
        delete[] tt;
    }

    void clear_tt() {
        std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
    }

    void set_nnue_weight(float w) {
        nnue_weight = std::min(1.0f, std::max(0.0f, w));
    }

    // ─── Weight loading ─────────────────────────────────────────────────
    bool load_weights(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;

        // Feature map: int32 (2, 8, 90)
        size_t r = fread(feat_map.sq_to_feat, sizeof(int32_t), 2 * 8 * 90, f);
        if (r != 2 * 8 * 90) { fclose(f); return false; }

        // num_features: int32
        r = fread(&feat_map.num_features, sizeof(int32_t), 1, f);
        if (r != 1) { fclose(f); return false; }
        weights.num_features = feat_map.num_features;

        // Accumulator weight: (128, num_features)
        fread(weights.acc_weight, sizeof(float), ACC_SIZE * feat_map.num_features, f);
        fread(weights.acc_bias, sizeof(float), ACC_SIZE, f);

        // FC1
        fread(weights.fc1_weight, sizeof(float), H1_SIZE * ACC_SIZE * 2, f);
        fread(weights.fc1_bias, sizeof(float), H1_SIZE, f);

        // FC2
        fread(weights.fc2_weight, sizeof(float), H2_SIZE * H1_SIZE, f);
        fread(weights.fc2_bias, sizeof(float), H2_SIZE, f);

        // Output
        fread(weights.out_weight, sizeof(float), H2_SIZE, f);
        fread(&weights.out_bias, sizeof(float), 1, f);

        fclose(f);
        return true;
    }

    // ─── NNUE v2 evaluation ─────────────────────────────────────────────

    void compute_accumulator(const xiangqi::Board& b, int8_t stm,
                             float stm_acc[ACC_SIZE], float nstm_acc[ACC_SIZE]) const {
        std::memcpy(stm_acc, weights.acc_bias, ACC_SIZE * sizeof(float));
        std::memcpy(nstm_acc, weights.acc_bias, ACC_SIZE * sizeof(float));

        int8_t nstm_color = -stm;

        for (int sq = 0; sq < 90; ++sq) {
            int8_t piece = b.grid[sq];
            if (piece == 0) continue;

            int8_t pc = xiangqi::color_of(piece);
            int pt = xiangqi::piece_type(piece);

            // STM perspective: normalize so STM's pieces are at bottom
            // If STM is Red (1): no mirror. If STM is Black (-1): mirror.
            int stm_sq = (stm == xiangqi::RED) ? sq : (89 - sq);
            int stm_color_idx = (pc == stm) ? 0 : 1;
            int stm_feat = feat_map.sq_to_feat[stm_color_idx][pt][stm_sq];
            if (stm_feat >= 0) {
                for (int i = 0; i < ACC_SIZE; ++i)
                    stm_acc[i] += weights.acc_weight[i][stm_feat];
            }

            // NSTM perspective
            int nstm_sq = (nstm_color == xiangqi::RED) ? sq : (89 - sq);
            int nstm_color_idx = (pc == nstm_color) ? 0 : 1;
            int nstm_feat = feat_map.sq_to_feat[nstm_color_idx][pt][nstm_sq];
            if (nstm_feat >= 0) {
                for (int i = 0; i < ACC_SIZE; ++i)
                    nstm_acc[i] += weights.acc_weight[i][nstm_feat];
            }
        }
    }

    float evaluate_nnue(const xiangqi::Board& b, int8_t stm) const {
        float stm_acc[ACC_SIZE], nstm_acc[ACC_SIZE];
        compute_accumulator(b, stm, stm_acc, nstm_acc);

        // ClippedReLU
        for (int i = 0; i < ACC_SIZE; ++i) {
            stm_acc[i] = std::min(1.0f, std::max(0.0f, stm_acc[i]));
            nstm_acc[i] = std::min(1.0f, std::max(0.0f, nstm_acc[i]));
        }

        // FC1
        float h1[H1_SIZE];
        for (int i = 0; i < H1_SIZE; ++i) {
            float sum = weights.fc1_bias[i];
            for (int j = 0; j < ACC_SIZE; ++j)
                sum += weights.fc1_weight[i][j] * stm_acc[j];
            for (int j = 0; j < ACC_SIZE; ++j)
                sum += weights.fc1_weight[i][ACC_SIZE + j] * nstm_acc[j];
            h1[i] = std::min(1.0f, std::max(0.0f, sum));
        }

        // FC2
        float h2[H2_SIZE];
        for (int i = 0; i < H2_SIZE; ++i) {
            float sum = weights.fc2_bias[i];
            for (int j = 0; j < H1_SIZE; ++j)
                sum += weights.fc2_weight[i][j] * h1[j];
            h2[i] = std::min(1.0f, std::max(0.0f, sum));
        }

        // Output
        float out = weights.out_bias;
        for (int j = 0; j < H2_SIZE; ++j)
            out += weights.out_weight[j] * h2[j];

        return 1.0f / (1.0f + std::exp(-out));
    }

    // Material eval (fallback, same as v1)
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
        float diff = my_mat - opp_mat;
        return 0.5f + diff / 90.0f;
    }

    // Blended eval (configurable weight, default pure NNUE)
    float evaluate(const xiangqi::Board& b, int8_t stm) const {
        if (nnue_weight >= 1.0f) return evaluate_nnue(b, stm);
        if (nnue_weight <= 0.0f) return material_eval(b, stm);
        return nnue_weight * evaluate_nnue(b, stm) + (1.0f - nnue_weight) * material_eval(b, stm);
    }

    // ─── Move ordering ──────────────────────────────────────────────────

    static int mvv_lva_score(xiangqi::Move m, const xiangqi::Board& b) {
        if (m.captured == 0) return 0;
        int victim = piece_value(xiangqi::piece_type(m.captured));
        int attacker = piece_value(xiangqi::piece_type(b.grid[m.from_sq]));
        return victim * 100 - attacker;
    }

    static void order_moves(xiangqi::MoveList& ml, const xiangqi::Board& b) {
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

    // ─── Make/Unmake ────────────────────────────────────────────────────

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

        xiangqi::MoveList legal;
        xiangqi::get_legal_moves(b, stm, legal);
        order_moves(legal, b);

        for (int i = 0; i < legal.count; ++i) {
            xiangqi::Move m = legal.moves[i];
            if (m.captured == 0) continue;

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

        uint64_t key = zobrist.hash(b, stm);
        TTEntry& entry = tt[key & (TT_SIZE - 1)];
        if (entry.key == key && entry.depth >= depth) {
            stats.tt_hits++;
            if (entry.flag == TT_EXACT) return entry.value;
            if (entry.flag == TT_BETA && entry.value >= beta) return beta;
            if (entry.flag == TT_ALPHA && entry.value <= alpha) return alpha;
        }

        if (depth <= 0) {
            return quiescence(b, stm, alpha, beta, 6);
        }

        xiangqi::MoveList legal;
        xiangqi::get_legal_moves(b, stm, legal);

        if (legal.count == 0) return 0.0f;

        order_moves(legal, b);

        float best_val = -1.0f;
        TTFlag flag = TT_ALPHA;

        for (int i = 0; i < legal.count; ++i) {
            xiangqi::Move m = legal.moves[i];

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

}  // namespace nnue_v2
