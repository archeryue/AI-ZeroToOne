// C++ unit tests for MCTS.
// Compile: g++ -std=c++17 -O2 -I.. test_mcts.cpp ../go.cpp ../mcts.cpp -o test_mcts && ./test_mcts

#include "mcts.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <set>
#include <algorithm>
#include <chrono>

using namespace mcts;
using namespace go;

// Helper: create a uniform policy
template<int N>
void uniform_policy(float* policy) {
    constexpr int A = N * N + 1;
    float v = 1.0f / static_cast<float>(A);
    for (int i = 0; i < A; ++i) policy[i] = v;
}

// Helper: create policy with one hot action
template<int N>
void hot_policy(float* policy, int action, float hot_val = 0.9f) {
    constexpr int A = N * N + 1;
    float rest = (1.0f - hot_val) / static_cast<float>(A - 1);
    for (int i = 0; i < A; ++i) policy[i] = rest;
    policy[action] = hot_val;
}

// ─── Test 1: Tree creation and root state ───────────────────────

void test_tree_creation() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game);

    assert(tree.root_idx == 0);
    assert(tree.num_nodes() == 1);
    assert(tree.root_visit_count() == 0);
    assert(!tree.nodes[tree.root_idx].is_expanded());

    printf("  PASS: tree creation\n");
}

// ─── Test 2: Expand root with uniform policy ────────────────────

void test_expand_root() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game);

    // Expand root with uniform policy
    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    assert(tree.nodes[tree.root_idx].is_expanded());
    // On empty 9x9 board, all 82 actions are legal (81 positions + pass)
    assert(tree.nodes[tree.root_idx].num_children == 82);

    // Check children have correct priors (approximately uniform)
    int start = tree.nodes[tree.root_idx].children_start;
    for (int i = 0; i < 82; ++i) {
        const MCTSNode& child = tree.nodes[start + i];
        assert(child.visit_count == 0);
        assert(child.prior > 0.0f);
    }

    printf("  PASS: expand root\n");
}

// ─── Test 3: PUCT selection prefers high prior ──────────────────

void test_puct_selection() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, /*c_puct=*/1.5f);

    // Expand with a policy that heavily favors action 0 (tengen area)
    float policy[82];
    hot_policy<9>(policy, 40, 0.95f);  // center of 9x9 = (4,4) = action 40
    tree.expand(tree.root_idx, policy);

    // With no visits, PUCT selects highest prior
    int best = tree.select_child(tree.root_idx);
    assert(best >= 0);
    assert(tree.nodes[best].action == 40);

    printf("  PASS: PUCT selection prefers high prior\n");
}

// ─── Test 4: Backup propagation ─────────────────────────────────

void test_backup() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    // Expand root
    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    // Select a leaf (should be a child of root)
    auto path = tree.select_leaf();
    assert(path.size() == 2);  // root → child

    int leaf = path.back();
    assert(tree.nodes[leaf].virtual_loss == 1);

    // Backup with value +0.5
    tree.backup(path, 0.5f);

    // Leaf should have visit=1, value_sum=0.5
    assert(tree.nodes[leaf].visit_count == 1);
    assert(std::abs(tree.nodes[leaf].value_sum - 0.5f) < 1e-6f);
    assert(tree.nodes[leaf].virtual_loss == 0);

    // Root should have visit=1, value_sum=-0.5 (negated)
    assert(tree.nodes[tree.root_idx].visit_count == 1);
    assert(std::abs(tree.nodes[tree.root_idx].value_sum - (-0.5f)) < 1e-6f);

    printf("  PASS: backup propagation\n");
}

// ─── Test 5: Virtual loss yields different paths ────────────────

void test_virtual_loss() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    // Expand root with uniform policy
    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    // Select multiple leaves — virtual loss should push to different children
    int num_leaves = 8;
    auto leaves = tree.select_leaves(num_leaves);
    assert(static_cast<int>(leaves.size()) == num_leaves);

    // Collect the leaf node indices
    std::set<int> leaf_nodes;
    for (const auto& l : leaves) {
        leaf_nodes.insert(l.leaf_idx);
    }

    // With uniform prior and virtual loss, we should get different children
    // (at least a few distinct ones, though not necessarily all 8 unique)
    assert(leaf_nodes.size() >= 4);  // conservative check

    // All leaves should have virtual_loss >= 1
    for (const auto& l : leaves) {
        assert(tree.nodes[l.leaf_idx].virtual_loss >= 1);
    }

    printf("  PASS: virtual loss yields different paths (%zu unique)\n", leaf_nodes.size());
}

// ─── Test 6: Full simulation cycle ──────────────────────────────
// Expand root, select leaf, fill obs, expand leaf, backup.

void test_full_cycle() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    // First: expand root (simulates first NN call)
    float root_policy[82];
    uniform_policy<9>(root_policy);
    tree.expand(tree.root_idx, root_policy);

    // Run 100 simulations with random "NN"
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

    for (int sim = 0; sim < 100; ++sim) {
        auto leaves = tree.select_leaves(1);
        assert(leaves.size() == 1);

        const auto& leaf = leaves[0];
        if (leaf.needs_nn) {
            // Create game state for this leaf
            float obs[17 * 81];
            tree.fill_observations(leaves, obs);

            // Random policy and value
            float policy[82];
            uniform_policy<9>(policy);
            float value = vdist(rng);

            tree.process_results(leaves, policy, &value);
        } else {
            tree.process_results(leaves, nullptr, nullptr);
        }
    }

    assert(tree.root_visit_count() == 100);

    // Get policy — should be non-trivial
    float policy[82];
    tree.get_policy(policy, 1.0f);
    float sum = 0.0f;
    for (int i = 0; i < 82; ++i) sum += policy[i];
    assert(std::abs(sum - 1.0f) < 1e-4f);

    printf("  PASS: full simulation cycle (100 sims, root visits=%d, nodes=%d)\n",
           tree.root_visit_count(), tree.num_nodes());
}

// ─── Test 7: Tree reuse ────────────────────────────────────────

void test_tree_reuse() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    // Expand root
    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    // Run some sims to build up the tree
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

    for (int sim = 0; sim < 50; ++sim) {
        auto leaves = tree.select_leaves(1);
        const auto& leaf = leaves[0];
        if (leaf.needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, obs);
            float p[82];
            uniform_policy<9>(p);
            float v = vdist(rng);
            tree.process_results(leaves, p, &v);
        } else {
            tree.process_results(leaves, nullptr, nullptr);
        }
    }

    int old_root = tree.root_idx;
    int action = tree.best_action();
    assert(action >= 0 && action < 82);

    // Find the child node for this action
    int child_visits = 0;
    const MCTSNode& root = tree.nodes[old_root];
    for (int i = 0; i < root.num_children; ++i) {
        if (tree.nodes[root.children_start + i].action == action) {
            child_visits = tree.nodes[root.children_start + i].visit_count;
            break;
        }
    }

    // Advance tree
    tree.advance(action);

    assert(tree.root_idx != old_root);
    assert(tree.nodes[tree.root_idx].action == action);

    // Root visits should be preserved from the child
    assert(tree.root_visit_count() == child_visits);

    printf("  PASS: tree reuse (action=%d, preserved visits=%d)\n", action, child_visits);
}

// ─── Test 8: Dirichlet noise ───────────────────────────────────

void test_dirichlet_noise() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f, /*dir_alpha=*/0.11f, /*dir_eps=*/0.25f);

    // Expand root
    float policy[82];
    // Give one action a very high prior
    hot_policy<9>(policy, 40, 0.99f);
    tree.expand(tree.root_idx, policy);

    // Record priors before noise
    const MCTSNode& root = tree.nodes[tree.root_idx];
    float before_max = 0.0f;
    for (int i = 0; i < root.num_children; ++i) {
        before_max = std::max(before_max, tree.nodes[root.children_start + i].prior);
    }
    assert(before_max > 0.9f);

    // Apply noise
    std::mt19937 rng(42);
    tree.apply_dirichlet_noise(rng);

    // After noise, the max prior should be reduced
    float after_max = 0.0f;
    for (int i = 0; i < root.num_children; ++i) {
        after_max = std::max(after_max, tree.nodes[root.children_start + i].prior);
    }
    assert(after_max < before_max);

    // Priors should still sum to ~1
    float sum = 0.0f;
    for (int i = 0; i < root.num_children; ++i) {
        sum += tree.nodes[root.children_start + i].prior;
    }
    assert(std::abs(sum - 1.0f) < 0.01f);

    // Noise should only be applied once
    tree.apply_dirichlet_noise(rng);  // should be no-op
    float after2_max = 0.0f;
    for (int i = 0; i < root.num_children; ++i) {
        after2_max = std::max(after2_max, tree.nodes[root.children_start + i].prior);
    }
    assert(std::abs(after2_max - after_max) < 1e-6f);

    printf("  PASS: Dirichlet noise (max prior %.3f -> %.3f)\n", before_max, after_max);
}

// ─── Test 9: Policy temperature ─────────────────────────────────

void test_policy_temperature() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    // Expand root
    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    // Run sims with biased value to create visit imbalance
    std::mt19937 rng(42);
    for (int sim = 0; sim < 200; ++sim) {
        auto leaves = tree.select_leaves(1);
        const auto& leaf = leaves[0];
        if (leaf.needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, obs);
            float p[82];
            uniform_policy<9>(p);
            // Give slight preference to one child
            float v = (tree.nodes[leaf.leaf_idx].action == 40) ? 0.8f : -0.3f;
            tree.process_results(leaves, p, &v);
        } else {
            tree.process_results(leaves, nullptr, nullptr);
        }
    }

    // Temperature 1.0: proportional
    float policy_t1[82];
    tree.get_policy(policy_t1, 1.0f);
    float sum_t1 = 0.0f;
    for (int i = 0; i < 82; ++i) sum_t1 += policy_t1[i];
    assert(std::abs(sum_t1 - 1.0f) < 1e-4f);

    // Temperature 0.0: argmax
    float policy_t0[82];
    tree.get_policy(policy_t0, 0.0f);
    int non_zero = 0;
    for (int i = 0; i < 82; ++i) {
        if (policy_t0[i] > 0.5f) non_zero++;
    }
    assert(non_zero == 1);  // exactly one action has all the mass

    printf("  PASS: policy temperature\n");
}

// ─── Test 10: Integration — random NN plays legal Go ────────────

void test_integration_random_game() {
    Game<9> game(5.5f);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

    int moves_played = 0;

    for (int move = 0; move < 120 && game.status == PLAYING; ++move) {
        MCTSTree<9> tree(game, 1.5f, 0.11f, 0.25f);

        // Run 50 sims (fast, using random "NN")
        // First: expand root
        float root_policy[82];
        uniform_policy<9>(root_policy);
        tree.expand(tree.root_idx, root_policy);
        tree.apply_dirichlet_noise(rng);

        // Backup root expansion
        tree.backup({tree.root_idx}, vdist(rng));

        for (int sim = 0; sim < 50; ++sim) {
            auto leaves = tree.select_leaves(1);
            const auto& leaf = leaves[0];
            if (leaf.needs_nn) {
                float obs[17 * 81];
                tree.fill_observations(leaves, obs);
                float p[82];
                uniform_policy<9>(p);
                float v = vdist(rng);
                tree.process_results(leaves, p, &v);
            } else {
                tree.process_results(leaves, nullptr, nullptr);
            }
        }

        // Select best action
        int action = tree.best_action();
        assert(action >= 0 && action <= 81);

        // Make the move
        int result = game.make_move(action);
        assert(result >= 0);  // should always be legal
        moves_played++;
    }

    printf("  PASS: integration — random NN plays %d legal moves (status=%d)\n",
           moves_played, (int)game.status);
}

// ─── Test 11: Batch leaf collection ─────────────────────────────

void test_batch_leaves() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    // Expand root
    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    // Collect 8 leaves
    auto leaves = tree.select_leaves(8);
    assert(leaves.size() == 8);

    // Fill observations
    float obs_buffer[8 * 17 * 81];
    int nn_count = tree.fill_observations(leaves, obs_buffer);
    assert(nn_count == 8);  // all should need NN (first time)

    // Create random results
    float policies[8 * 82];
    float values[8];
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);
    for (int i = 0; i < 8; ++i) {
        uniform_policy<9>(policies + i * 82);
        values[i] = vdist(rng);
    }

    // Process results
    tree.process_results(leaves, policies, values);

    // Root should have 8 visits
    assert(tree.root_visit_count() == 8);

    printf("  PASS: batch leaf collection (nn=%d, root_visits=%d)\n",
           nn_count, tree.root_visit_count());
}

// ─── Test 12: Tree reuse with advance + new sims ────────────────

void test_tree_reuse_with_sims() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

    // Expand root and run sims
    float root_policy[82];
    uniform_policy<9>(root_policy);
    tree.expand(tree.root_idx, root_policy);

    for (int sim = 0; sim < 100; ++sim) {
        auto leaves = tree.select_leaves(1);
        if (leaves[0].needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, obs);
            float p[82];
            uniform_policy<9>(p);
            float v = vdist(rng);
            tree.process_results(leaves, p, &v);
        } else {
            tree.process_results(leaves, nullptr, nullptr);
        }
    }

    // Play best action
    int action = tree.best_action();
    game.make_move(action);
    tree.advance(action);

    // Continue running sims on reused tree
    if (!tree.nodes[tree.root_idx].is_expanded()) {
        float p[82];
        uniform_policy<9>(p);
        tree.expand(tree.root_idx, p);
    }

    for (int sim = 0; sim < 50; ++sim) {
        auto leaves = tree.select_leaves(1);
        if (leaves[0].needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, obs);
            float p[82];
            uniform_policy<9>(p);
            float v = vdist(rng);
            tree.process_results(leaves, p, &v);
        } else {
            tree.process_results(leaves, nullptr, nullptr);
        }
    }

    // Should have additional visits
    assert(tree.root_visit_count() > 0);

    printf("  PASS: tree reuse with continued sims (root visits=%d, nodes=%d)\n",
           tree.root_visit_count(), tree.num_nodes());
}

// ─── Performance benchmark ──────────────────────────────────────

void bench_mcts() {
    printf("\n--- MCTS Benchmarks ---\n");

    // Benchmark: MCTS simulations per second (9x9, random NN)
    {
        Game<9> game(5.5f);
        MCTSTree<9> tree(game, 1.5f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

        // Expand root
        float root_policy[82];
        uniform_policy<9>(root_policy);
        tree.expand(tree.root_idx, root_policy);

        int total_sims = 10000;
        auto start = std::chrono::high_resolution_clock::now();

        for (int sim = 0; sim < total_sims; ++sim) {
            auto leaves = tree.select_leaves(1);
            if (leaves[0].needs_nn) {
                float obs[17 * 81];
                tree.fill_observations(leaves, obs);
                float p[82];
                uniform_policy<9>(p);
                float v = vdist(rng);
                tree.process_results(leaves, p, &v);
            } else {
                tree.process_results(leaves, nullptr, nullptr);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  9x9 MCTS: %d sims in %.1f ms (%.0f sims/sec, %.2f us/sim)\n",
               total_sims, ms, total_sims / (ms / 1000.0), ms * 1000.0 / total_sims);
        printf("    nodes=%d, root_visits=%d\n", tree.num_nodes(), tree.root_visit_count());
    }

    // Benchmark: batch leaf collection (8 leaves at a time)
    {
        Game<9> game(5.5f);
        MCTSTree<9> tree(game, 1.5f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

        float root_policy[82];
        uniform_policy<9>(root_policy);
        tree.expand(tree.root_idx, root_policy);

        int total_sims = 10000;
        int batch = 8;
        auto start = std::chrono::high_resolution_clock::now();

        for (int sim = 0; sim < total_sims; sim += batch) {
            int b = std::min(batch, total_sims - sim);
            auto leaves = tree.select_leaves(b);

            float obs[8 * 17 * 81];
            int nn = tree.fill_observations(leaves, obs);

            float policies[8 * 82];
            float values[8];
            for (int i = 0; i < nn; ++i) {
                uniform_policy<9>(policies + i * 82);
                values[i] = vdist(rng);
            }
            tree.process_results(leaves, policies, values);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  9x9 MCTS (batch=8): %d sims in %.1f ms (%.0f sims/sec, %.2f us/sim)\n",
               total_sims, ms, total_sims / (ms / 1000.0), ms * 1000.0 / total_sims);
    }

    // Benchmark: 19x19 MCTS
    {
        Game<19> game(7.5f);
        MCTSTree<19> tree(game, 1.5f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

        float root_policy[362];
        float v = 1.0f / 362.0f;
        for (int i = 0; i < 362; ++i) root_policy[i] = v;
        tree.expand(tree.root_idx, root_policy);

        int total_sims = 5000;
        auto start = std::chrono::high_resolution_clock::now();

        for (int sim = 0; sim < total_sims; ++sim) {
            auto leaves = tree.select_leaves(1);
            if (leaves[0].needs_nn) {
                float obs[17 * 361];
                tree.fill_observations(leaves, obs);
                float p[362];
                for (int i = 0; i < 362; ++i) p[i] = v;
                float val = vdist(rng);
                tree.process_results(leaves, p, &val);
            } else {
                tree.process_results(leaves, nullptr, nullptr);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  19x19 MCTS: %d sims in %.1f ms (%.0f sims/sec, %.2f us/sim)\n",
               total_sims, ms, total_sims / (ms / 1000.0), ms * 1000.0 / total_sims);
        printf("    nodes=%d, root_visits=%d\n", tree.num_nodes(), tree.root_visit_count());
    }
}

// ─── Main ───────────────────────────────────────────────────────

int main() {
    printf("=== MCTS C++ Tests ===\n\n");

    test_tree_creation();
    test_expand_root();
    test_puct_selection();
    test_backup();
    test_virtual_loss();
    test_full_cycle();
    test_tree_reuse();
    test_dirichlet_noise();
    test_policy_temperature();
    test_integration_random_game();
    test_batch_leaves();
    test_tree_reuse_with_sims();

    bench_mcts();

    printf("\n=== All MCTS tests passed! ===\n");
    return 0;
}
