#pragma once
// SelfPlayWorker: manages multiple parallel Go games for one worker thread.
// All game logic + MCTS runs in C++ with the GIL released for true parallelism.
//
// Architecture (from PLAN.md):
//   5 workers × ~52 games = 256 parallel games
//   Each tick: workers select leaves → GPU batch → workers process results
//
// The worker handles the full game lifecycle:
//   - MCTS leaf selection + observation encoding
//   - Processing NN results (expand + backup)
//   - Move completion (policy sampling, tree advance)
//   - Game completion (value assignment, data collection)
//   - Automatic game restart for continuous self-play

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include "go.h"
#include "mcts.h"

namespace alphazero {

struct SelfPlayConfig {
    float komi = 7.5f;
    float c_puct = 1.5f;
    float dirichlet_alpha = 0.03f;
    float dirichlet_epsilon = 0.25f;
    int vl_batch = 8;
    int num_sims = 400;
    int temp_moves = 15;
    float temp_high = 1.0f;
    float temp_low = 0.1f;
    // Resign v2 — loosened threshold + move floor + credible-child
    // cross-check. Iter 12→19 regression traced back to aggressive
    // resign cutting middle/endgame positions out of the replay buffer.
    float resign_threshold = -0.90f;
    int resign_consecutive = 5;
    int resign_min_move = 20;
    float resign_disabled_frac = 0.20f;
    float resign_min_child_visits_frac = 0.05f;
    int max_game_moves = 200;
};

template<int N>
class SelfPlayWorker {
public:
    static constexpr int CELLS = N * N;
    static constexpr int ACTIONS = CELLS + 1;
    static constexpr int OBS_PLANES = 17;
    static constexpr int OBS_SIZE = OBS_PLANES * CELLS;
    static constexpr int MAX_VL = 16;

    // Hard cap on per-game MCTSTree size, enforced after each move.
    // MCTSTree::advance() only re-roots; it never frees orphaned subtrees.
    // Without this cap, a 13x13 game runs ~250 moves × 600 sims × ~100
    // children/expansion ≈ 15M nodes per tree → ~1 GB per tree → 280 GB
    // across 256 parallel games. Phase 2 dryrun OOM-killed Docker on
    // exactly this. The cap is checked after advance() so the new root
    // is the current game position; reset() rebuilds a fresh tree from
    // that position, bounding RAM at the cost of losing inherited
    // visits on the reset move.
    //
    // Sizing (13x13, ~72k new nodes per move at 600 sims):
    //   cap          per-tree   × 256 trees   reset every   cold moves
    //   -----        --------   -----------   -----------   ----------
    //   200k          ~16 MB       ~4 GB        ~3 moves       ~33 %
    //   1,000k        ~75 MB      ~19 GB       ~14 moves        ~7 %
    //   2,000k       ~150 MB      ~38 GB       ~28 moves      ~3.5 %
    //
    // 1M is the sweet spot on a 35 GB budget: peak MCTS ~19 GB fits
    // alongside buffer (3.6) + savez transient (3.6) + model/compile
    // (~4) = ~30 GB total, ~5 GB headroom. Cold-move fraction drops to
    // ~7 %, so the AI convergence hit is ~3 % instead of ~13 %.
    // See PHASE_TWO_TRAINING.md "Problem 1" for the full memory math.
    static constexpr int MAX_TREE_NODES = 1000000;

private:
    // Per-position record (obs + policy + who played + per-cell
    // ownership target). Ownership is filled in by finish_game once
    // the game ends — at training time it gives the trunk dense
    // per-cell supervision (KataGo-style) instead of the single
    // scalar value label per ~150-move game. See PHASE_TWO_TRAINING.md
    // run4/ownership-head section. Stored in current-player perspective:
    //   +1 = cell ends up owned by player-to-move at this position
    //   -1 = cell ends up owned by the opponent
    //    0 = dame (no one)
    struct MoveRecord {
        float obs[OBS_SIZE];
        float policy[ACTIONS];
        int8_t ownership[CELLS];
        uint8_t turn;
    };

    struct GameSlot {
        go::Game<N> game;
        std::unique_ptr<mcts::MCTSTree<N>> tree;
        mcts::LeafInfo leaves[MAX_VL];
        int nn_count = 0;        // NN evals from last tick_select

        int sims_done = 0;
        int move_num = 0;
        bool active = false;

        std::vector<MoveRecord> records;
        int consecutive_low = 0;
        bool disable_resign = false;
    };

    SelfPlayConfig cfg_;
    int num_games_;
    std::vector<GameSlot> slots_;
    std::mt19937 rng_;

    // Completed training data (harvested by orchestrator between ticks)
    std::vector<float> done_obs_;
    std::vector<float> done_policy_;
    std::vector<float> done_value_;
    std::vector<int8_t> done_ownership_;  // length = done_count_ * CELLS
    int done_count_ = 0;
    int games_completed_ = 0;

    // ── Game lifecycle ──────────────────────────────────────────

    void init_slot(int idx) {
        auto& s = slots_[idx];
        s.game = go::Game<N>(cfg_.komi);
        s.tree = std::make_unique<mcts::MCTSTree<N>>(
            s.game, cfg_.c_puct, cfg_.dirichlet_alpha, cfg_.dirichlet_epsilon);
        s.sims_done = 0;
        s.move_num = 0;
        s.active = true;
        s.nn_count = 0;
        s.records.clear();
        s.consecutive_low = 0;
        auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        s.disable_resign = dist(rng_) < cfg_.resign_disabled_frac;
    }

    void finish_game(int idx) {
        auto& s = slots_[idx];

        // Force end if still playing
        while (s.game.status == go::PLAYING)
            s.game.make_move(CELLS);

        float result = (s.game.status == go::BLACK_WIN) ? 1.0f : -1.0f;

        // Compute per-cell ownership ONCE at game end (Tromp-Taylor on
        // the final board). Each MoveRecord then gets a copy flipped
        // into that record's current-player perspective. This is the
        // KataGo-style auxiliary supervision target — 169 dense labels
        // per position vs the single scalar value label per game.
        int8_t abs_ownership[CELLS];
        s.game.board.compute_ownership(abs_ownership);

        for (auto& rec : s.records) {
            float value = (rec.turn == go::BLACK) ? result : -result;
            int8_t persp = (rec.turn == go::BLACK) ? 1 : -1;
            for (int p = 0; p < CELLS; ++p) {
                rec.ownership[p] = static_cast<int8_t>(abs_ownership[p] * persp);
            }

            done_obs_.insert(done_obs_.end(), rec.obs, rec.obs + OBS_SIZE);
            done_policy_.insert(done_policy_.end(), rec.policy, rec.policy + ACTIONS);
            done_value_.push_back(value);
            done_ownership_.insert(done_ownership_.end(),
                                   rec.ownership, rec.ownership + CELLS);
            done_count_++;
        }

        games_completed_++;
        s.active = false;
        s.records.clear();
    }

    void complete_move(int idx) {
        auto& s = slots_[idx];
        float temp = (s.move_num < cfg_.temp_moves) ? cfg_.temp_high : cfg_.temp_low;

        // Record position
        MoveRecord rec;
        s.game.to_observation(rec.obs);
        s.tree->get_policy(rec.policy, temp);
        rec.turn = s.game.current_turn;
        s.records.push_back(rec);

        // Resign check (v2 — see SelfPlayConfig comment).
        // Gated by a hard move-count floor to keep early-game losing
        // positions in the buffer, and cross-checked against per-child
        // Q-values so a credibly-visited "recovery line" blocks resign.
        if (!s.disable_resign && s.move_num >= cfg_.resign_min_move) {
            float root_v = s.tree->root_value();
            bool losing = (root_v < cfg_.resign_threshold);

            if (losing) {
                const auto& root = s.tree->nodes[s.tree->root_idx];
                int total_visits = root.visit_count;
                int min_visits = std::max(
                    1,
                    static_cast<int>(total_visits * cfg_.resign_min_child_visits_frac));
                for (int i = 0; i < root.num_children; ++i) {
                    const auto& child =
                        s.tree->nodes[root.children_start + i];
                    if (child.visit_count >= min_visits
                        && child.q_value() > cfg_.resign_threshold) {
                        losing = false;  // credible recovery line exists
                        break;
                    }
                }
            }

            if (losing)
                s.consecutive_low++;
            else
                s.consecutive_low = 0;

            if (s.consecutive_low >= cfg_.resign_consecutive) {
                s.game.resign(s.game.current_turn);
                finish_game(idx);
                return;
            }
        }

        // Sample action from MCTS policy
        std::discrete_distribution<int> dist(rec.policy, rec.policy + ACTIONS);
        int action = dist(rng_);

        // Play move
        s.game.make_move(action);
        s.tree->advance(action);

        // Bound per-game MCTS memory. advance() only re-roots — orphaned
        // subtrees stay allocated in the nodes vector and game_pool deque
        // for the entire game, which is what blew up the Phase 2 13x13
        // dryrun. If the persistent tree has grown past MAX_TREE_NODES,
        // throw it away and rebuild from the current game state. The
        // first move after a reset starts cold (no inherited visits) but
        // still gets the full num_sims budget, so quality only marginally
        // degrades while RSS stays bounded.
        if (s.tree->num_nodes() > MAX_TREE_NODES) {
            s.tree->reset(s.game);
        }

        s.sims_done = 0;
        s.move_num++;

        // Check game end
        if (s.game.status != go::PLAYING || s.move_num >= cfg_.max_game_moves) {
            finish_game(idx);
        }
    }

public:
    SelfPlayWorker(int num_games, const SelfPlayConfig& cfg, int seed)
        : cfg_(cfg), num_games_(num_games), rng_(seed)
    {
        slots_.resize(num_games);
        for (int i = 0; i < num_games; ++i)
            init_slot(i);
    }

    // ── Tick API (called by worker thread) ──────────────────────

    // Select leaves for all active games, write observations into obs_out.
    // Returns total number of NN evaluations needed.
    // obs_out must have space for num_games * vl_batch * OBS_SIZE floats.
    int tick_select(float* obs_out) {
        int total_nn = 0;
        int batch = std::min(cfg_.vl_batch, (int)MAX_VL);

        for (int i = 0; i < num_games_; ++i) {
            auto& s = slots_[i];
            if (!s.active) { s.nn_count = 0; continue; }

            s.tree->select_leaves(s.leaves, batch);

            // Dirichlet noise at root (once per move, after first expansion)
            if (!s.tree->root_noise_applied
                && s.tree->nodes[s.tree->root_idx].is_expanded()) {
                s.tree->apply_dirichlet_noise(rng_);
            }

            s.nn_count = s.tree->fill_observations(
                s.leaves, batch, obs_out + total_nn * OBS_SIZE);
            total_nn += s.nn_count;
        }
        return total_nn;
    }

    // Process NN results and handle move/game completion.
    // policies/values have total_nn entries (from tick_select's return value).
    // Pass nullptr if total_nn was 0.
    void tick_process(const float* policies, const float* values) {
        int nn_offset = 0;
        int batch = std::min(cfg_.vl_batch, (int)MAX_VL);

        for (int i = 0; i < num_games_; ++i) {
            auto& s = slots_[i];
            if (!s.active) continue;

            if (s.nn_count > 0) {
                s.tree->process_results(s.leaves, batch,
                    policies + nn_offset * ACTIONS,
                    values + nn_offset);
                nn_offset += s.nn_count;
            } else {
                s.tree->process_results(s.leaves, batch, nullptr, nullptr);
            }

            s.sims_done += batch;

            if (s.sims_done >= cfg_.num_sims)
                complete_move(i);  // may set active=false
        }
    }

    // Restart any completed game slots (for continuous self-play).
    void restart_completed() {
        for (int i = 0; i < num_games_; ++i) {
            if (!slots_[i].active)
                init_slot(i);
        }
    }

    // ── Data harvesting (called by orchestrator) ────────────────

    int completed_count() const { return done_count_; }
    int games_done() const { return games_completed_; }

    int active_count() const {
        int c = 0;
        for (const auto& s : slots_) if (s.active) c++;
        return c;
    }

    // Max tree-node count across all game slots (diagnostic).
    int max_tree_nodes() const {
        int m = 0;
        for (const auto& s : slots_) {
            if (s.tree) {
                int n = s.tree->num_nodes();
                if (n > m) m = n;
            }
        }
        return m;
    }

    // Move completed training data out. Returns (obs, policy, value,
    // ownership, count). Ownership added in run4 for the KataGo-style
    // auxiliary head — see compute_ownership in go.h and the
    // PHASE_TWO_TRAINING.md ownership-head section.
    //   obs:        flat float, length = count * OBS_SIZE
    //   policy:     flat float, length = count * ACTIONS
    //   value:      float, length = count
    //   ownership:  int8, length = count * CELLS  (current-player perspective)
    struct HarvestResult {
        std::vector<float> obs;
        std::vector<float> policy;
        std::vector<float> value;
        std::vector<int8_t> ownership;
        int count;
    };

    HarvestResult harvest() {
        HarvestResult r;
        r.obs = std::move(done_obs_);
        r.policy = std::move(done_policy_);
        r.value = std::move(done_value_);
        r.ownership = std::move(done_ownership_);
        r.count = done_count_;
        done_obs_.clear();
        done_policy_.clear();
        done_value_.clear();
        done_ownership_.clear();
        done_count_ = 0;
        return r;
    }
};

extern template class SelfPlayWorker<9>;
extern template class SelfPlayWorker<13>;
extern template class SelfPlayWorker<19>;

}  // namespace alphazero
