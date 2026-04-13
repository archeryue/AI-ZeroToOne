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

private:
    // Per-position record (obs + policy + who played)
    struct MoveRecord {
        float obs[OBS_SIZE];
        float policy[ACTIONS];
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

        for (const auto& rec : s.records) {
            float value = (rec.turn == go::BLACK) ? result : -result;
            done_obs_.insert(done_obs_.end(), rec.obs, rec.obs + OBS_SIZE);
            done_policy_.insert(done_policy_.end(), rec.policy, rec.policy + ACTIONS);
            done_value_.push_back(value);
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

    // Move completed training data out. Returns (obs, policy, value, count).
    // obs:    flat float array, length = count * OBS_SIZE
    // policy: flat float array, length = count * ACTIONS
    // value:  float array, length = count
    struct HarvestResult {
        std::vector<float> obs;
        std::vector<float> policy;
        std::vector<float> value;
        int count;
    };

    HarvestResult harvest() {
        HarvestResult r;
        r.obs = std::move(done_obs_);
        r.policy = std::move(done_policy_);
        r.value = std::move(done_value_);
        r.count = done_count_;
        done_obs_.clear();
        done_policy_.clear();
        done_value_.clear();
        done_count_ = 0;
        return r;
    }
};

extern template class SelfPlayWorker<9>;
extern template class SelfPlayWorker<13>;
extern template class SelfPlayWorker<19>;

}  // namespace alphazero
