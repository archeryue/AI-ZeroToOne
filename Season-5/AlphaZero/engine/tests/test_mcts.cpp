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
    assert(tree.nodes[tree.root_idx].game_idx >= 0);

    printf("  PASS: tree creation\n");
}

// ─── Test 2: Expand root with uniform policy ────────────────────

void test_expand_root() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game);

    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    assert(tree.nodes[tree.root_idx].is_expanded());
    assert(tree.nodes[tree.root_idx].num_children == 82);

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
    MCTSTree<9> tree(game, 1.5f);

    float policy[82];
    hot_policy<9>(policy, 40, 0.95f);
    tree.expand(tree.root_idx, policy);

    int best = tree.select_child(tree.root_idx);
    assert(best >= 0);
    assert(tree.nodes[best].action == 40);

    printf("  PASS: PUCT selection prefers high prior\n");
}

// ─── Test 4: Backup propagation ─────────────────────────────────

void test_backup() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    int path[MAX_PATH_DEPTH];
    int path_len = tree.select_leaf(path);
    assert(path_len == 2);  // root → child

    int leaf = path[path_len - 1];
    assert(tree.nodes[leaf].virtual_loss == 1);

    // Backup with value +0.5
    tree.backup(path, path_len, 0.5f);

    assert(tree.nodes[leaf].visit_count == 1);
    // Convention 2: leaf stores value from parent's perspective (negated)
    assert(std::abs(tree.nodes[leaf].value_sum - (-0.5f)) < 1e-6f);
    assert(tree.nodes[leaf].virtual_loss == 0);

    // Root: value flipped again → from root's hypothetical parent's perspective
    assert(tree.nodes[tree.root_idx].visit_count == 1);
    assert(std::abs(tree.nodes[tree.root_idx].value_sum - 0.5f) < 1e-6f);

    printf("  PASS: backup propagation\n");
}

// ─── Test 5: Virtual loss yields different paths ────────────────

void test_virtual_loss() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    int num_leaves = 8;
    LeafInfo leaves[8];
    tree.select_leaves(leaves, num_leaves);

    std::set<int> leaf_nodes;
    for (int i = 0; i < num_leaves; ++i) {
        leaf_nodes.insert(leaves[i].leaf_idx);
    }

    // With uniform prior and virtual loss, should get diverse children
    assert(leaf_nodes.size() >= 4);

    for (int i = 0; i < num_leaves; ++i) {
        assert(tree.nodes[leaves[i].leaf_idx].virtual_loss >= 1);
    }

    printf("  PASS: virtual loss yields different paths (%zu unique)\n", leaf_nodes.size());
}

// ─── Test 6: Full simulation cycle ──────────────────────────────

void test_full_cycle() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    float root_policy[82];
    uniform_policy<9>(root_policy);
    tree.expand(tree.root_idx, root_policy);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

    for (int sim = 0; sim < 100; ++sim) {
        LeafInfo leaves[1];
        tree.select_leaves(leaves, 1);

        if (leaves[0].needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, 1, obs);

            float policy[82];
            uniform_policy<9>(policy);
            float value = vdist(rng);
            tree.process_results(leaves, 1, policy, &value);
        } else {
            tree.process_results(leaves, 1, nullptr, nullptr);
        }
    }

    assert(tree.root_visit_count() == 100);

    float policy[82];
    tree.get_policy(policy, 1.0f);
    float sum = 0.0f;
    for (int i = 0; i < 82; ++i) sum += policy[i];
    assert(std::abs(sum - 1.0f) < 1e-4f);

    printf("  PASS: full simulation cycle (100 sims, root visits=%d, nodes=%d, game_states=%d)\n",
           tree.root_visit_count(), tree.num_nodes(), tree.num_game_states());
}

// ─── Test 7: Tree reuse ────────────────────────────────────────

void test_tree_reuse() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

    for (int sim = 0; sim < 50; ++sim) {
        LeafInfo leaves[1];
        tree.select_leaves(leaves, 1);
        if (leaves[0].needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, 1, obs);
            float p[82];
            uniform_policy<9>(p);
            float v = vdist(rng);
            tree.process_results(leaves, 1, p, &v);
        } else {
            tree.process_results(leaves, 1, nullptr, nullptr);
        }
    }

    int old_root = tree.root_idx;
    int action = tree.best_action();
    assert(action >= 0 && action < 82);

    int child_visits = 0;
    const MCTSNode& root = tree.nodes[old_root];
    for (int i = 0; i < root.num_children; ++i) {
        if (tree.nodes[root.children_start + i].action == action) {
            child_visits = tree.nodes[root.children_start + i].visit_count;
            break;
        }
    }

    tree.advance(action);

    assert(tree.root_idx != old_root);
    assert(tree.nodes[tree.root_idx].action == action);
    assert(tree.root_visit_count() == child_visits);

    printf("  PASS: tree reuse (action=%d, preserved visits=%d)\n", action, child_visits);
}

// ─── Test 8: Dirichlet noise ───────────────────────────────────

void test_dirichlet_noise() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f, 0.11f, 0.25f);

    float policy[82];
    hot_policy<9>(policy, 40, 0.99f);
    tree.expand(tree.root_idx, policy);

    const MCTSNode& root = tree.nodes[tree.root_idx];
    float before_max = 0.0f;
    for (int i = 0; i < root.num_children; ++i)
        before_max = std::max(before_max, tree.nodes[root.children_start + i].prior);
    assert(before_max > 0.9f);

    std::mt19937 rng(42);
    tree.apply_dirichlet_noise(rng);

    float after_max = 0.0f;
    for (int i = 0; i < root.num_children; ++i)
        after_max = std::max(after_max, tree.nodes[root.children_start + i].prior);
    assert(after_max < before_max);

    float sum = 0.0f;
    for (int i = 0; i < root.num_children; ++i)
        sum += tree.nodes[root.children_start + i].prior;
    assert(std::abs(sum - 1.0f) < 0.01f);

    // Should be idempotent
    tree.apply_dirichlet_noise(rng);
    float after2_max = 0.0f;
    for (int i = 0; i < root.num_children; ++i)
        after2_max = std::max(after2_max, tree.nodes[root.children_start + i].prior);
    assert(std::abs(after2_max - after_max) < 1e-6f);

    printf("  PASS: Dirichlet noise (max prior %.3f -> %.3f)\n", before_max, after_max);
}

// ─── Test 9: Policy temperature ─────────────────────────────────

void test_policy_temperature() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    std::mt19937 rng(42);
    for (int sim = 0; sim < 200; ++sim) {
        LeafInfo leaves[1];
        tree.select_leaves(leaves, 1);
        if (leaves[0].needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, 1, obs);
            float p[82];
            uniform_policy<9>(p);
            float v = (tree.nodes[leaves[0].leaf_idx].action == 40) ? 0.8f : -0.3f;
            tree.process_results(leaves, 1, p, &v);
        } else {
            tree.process_results(leaves, 1, nullptr, nullptr);
        }
    }

    float policy_t1[82];
    tree.get_policy(policy_t1, 1.0f);
    float sum_t1 = 0.0f;
    for (int i = 0; i < 82; ++i) sum_t1 += policy_t1[i];
    assert(std::abs(sum_t1 - 1.0f) < 1e-4f);

    float policy_t0[82];
    tree.get_policy(policy_t0, 0.0f);
    int non_zero = 0;
    for (int i = 0; i < 82; ++i) {
        if (policy_t0[i] > 0.5f) non_zero++;
    }
    assert(non_zero == 1);

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

        // Expand root
        float root_policy[82];
        uniform_policy<9>(root_policy);
        tree.expand(tree.root_idx, root_policy);
        tree.apply_dirichlet_noise(rng);

        // Backup root expansion
        int root_path[] = {tree.root_idx};
        tree.backup(root_path, 1, vdist(rng));

        for (int sim = 0; sim < 50; ++sim) {
            LeafInfo leaves[1];
            tree.select_leaves(leaves, 1);
            if (leaves[0].needs_nn) {
                float obs[17 * 81];
                tree.fill_observations(leaves, 1, obs);
                float p[82];
                uniform_policy<9>(p);
                float v = vdist(rng);
                tree.process_results(leaves, 1, p, &v);
            } else {
                tree.process_results(leaves, 1, nullptr, nullptr);
            }
        }

        int action = tree.best_action();
        assert(action >= 0 && action <= 81);

        int result = game.make_move(action);
        assert(result >= 0);
        moves_played++;
    }

    printf("  PASS: integration — random NN plays %d legal moves (status=%d)\n",
           moves_played, (int)game.status);
}

// ─── Test 11: Batch leaf collection ─────────────────────────────

void test_batch_leaves() {
    Game<9> game(5.5f);
    MCTSTree<9> tree(game, 1.5f);

    float policy[82];
    uniform_policy<9>(policy);
    tree.expand(tree.root_idx, policy);

    LeafInfo leaves[8];
    tree.select_leaves(leaves, 8);

    float obs_buffer[8 * 17 * 81];
    int nn_count = tree.fill_observations(leaves, 8, obs_buffer);
    assert(nn_count == 8);

    float policies[8 * 82];
    float values[8];
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);
    for (int i = 0; i < 8; ++i) {
        uniform_policy<9>(policies + i * 82);
        values[i] = vdist(rng);
    }

    tree.process_results(leaves, 8, policies, values);
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

    float root_policy[82];
    uniform_policy<9>(root_policy);
    tree.expand(tree.root_idx, root_policy);

    for (int sim = 0; sim < 100; ++sim) {
        LeafInfo leaves[1];
        tree.select_leaves(leaves, 1);
        if (leaves[0].needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, 1, obs);
            float p[82];
            uniform_policy<9>(p);
            float v = vdist(rng);
            tree.process_results(leaves, 1, p, &v);
        } else {
            tree.process_results(leaves, 1, nullptr, nullptr);
        }
    }

    int action = tree.best_action();
    game.make_move(action);
    tree.advance(action);

    if (!tree.nodes[tree.root_idx].is_expanded()) {
        float p[82];
        uniform_policy<9>(p);
        tree.expand(tree.root_idx, p);
    }

    for (int sim = 0; sim < 50; ++sim) {
        LeafInfo leaves[1];
        tree.select_leaves(leaves, 1);
        if (leaves[0].needs_nn) {
            float obs[17 * 81];
            tree.fill_observations(leaves, 1, obs);
            float p[82];
            uniform_policy<9>(p);
            float v = vdist(rng);
            tree.process_results(leaves, 1, p, &v);
        } else {
            tree.process_results(leaves, 1, nullptr, nullptr);
        }
    }

    assert(tree.root_visit_count() > 0);

    printf("  PASS: tree reuse with continued sims (root visits=%d, nodes=%d, game_states=%d)\n",
           tree.root_visit_count(), tree.num_nodes(), tree.num_game_states());
}

// ─── Test 13: Node struct size ──────────────────────────────────

void test_node_size() {
    // Should be 32 bytes or less for cache efficiency
    assert(sizeof(MCTSNode) <= 32);
    printf("  PASS: MCTSNode is %zu bytes, alignof=%zu\n", sizeof(MCTSNode), alignof(MCTSNode));
}

// ─── Performance benchmark ──────────────────────────────────────

void bench_mcts() {
    printf("\n--- MCTS Benchmarks ---\n");

    // 9x9 single-leaf
    {
        Game<9> game(5.5f);
        MCTSTree<9> tree(game, 1.5f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);

        float root_policy[82];
        uniform_policy<9>(root_policy);
        tree.expand(tree.root_idx, root_policy);

        int total_sims = 10000;
        auto start = std::chrono::high_resolution_clock::now();

        for (int sim = 0; sim < total_sims; ++sim) {
            LeafInfo leaves[1];
            tree.select_leaves(leaves, 1);
            if (leaves[0].needs_nn) {
                float obs[17 * 81];
                tree.fill_observations(leaves, 1, obs);
                float p[82];
                uniform_policy<9>(p);
                float v = vdist(rng);
                tree.process_results(leaves, 1, p, &v);
            } else {
                tree.process_results(leaves, 1, nullptr, nullptr);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  9x9 MCTS: %d sims in %.1f ms (%.0f sims/sec, %.2f us/sim)\n",
               total_sims, ms, total_sims / (ms / 1000.0), ms * 1000.0 / total_sims);
        printf("    nodes=%d, game_states=%d, root_visits=%d\n",
               tree.num_nodes(), tree.num_game_states(), tree.root_visit_count());
    }

    // 9x9 batch=8
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
            LeafInfo leaves[8];
            tree.select_leaves(leaves, b);

            float obs[8 * 17 * 81];
            int nn = tree.fill_observations(leaves, b, obs);

            float policies[8 * 82];
            float values[8];
            for (int i = 0; i < nn; ++i) {
                uniform_policy<9>(policies + i * 82);
                values[i] = vdist(rng);
            }
            tree.process_results(leaves, b, policies, values);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  9x9 MCTS (batch=8): %d sims in %.1f ms (%.0f sims/sec, %.2f us/sim)\n",
               total_sims, ms, total_sims / (ms / 1000.0), ms * 1000.0 / total_sims);
    }

    // 19x19
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
            LeafInfo leaves[1];
            tree.select_leaves(leaves, 1);
            if (leaves[0].needs_nn) {
                float obs[17 * 361];
                tree.fill_observations(leaves, 1, obs);
                float p[362];
                for (int i = 0; i < 362; ++i) p[i] = v;
                float val = vdist(rng);
                tree.process_results(leaves, 1, p, &val);
            } else {
                tree.process_results(leaves, 1, nullptr, nullptr);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  19x19 MCTS: %d sims in %.1f ms (%.0f sims/sec, %.2f us/sim)\n",
               total_sims, ms, total_sims / (ms / 1000.0), ms * 1000.0 / total_sims);
        printf("    nodes=%d, game_states=%d, root_visits=%d\n",
               tree.num_nodes(), tree.num_game_states(), tree.root_visit_count());
    }

    // 19x19 batch=8
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
        int batch = 8;
        auto start = std::chrono::high_resolution_clock::now();

        for (int sim = 0; sim < total_sims; sim += batch) {
            int b = std::min(batch, total_sims - sim);
            LeafInfo leaves[8];
            tree.select_leaves(leaves, b);

            float obs[8 * 17 * 361];
            int nn = tree.fill_observations(leaves, b, obs);

            float policies[8 * 362];
            float values[8];
            for (int i = 0; i < nn; ++i) {
                for (int j = 0; j < 362; ++j) policies[i * 362 + j] = v;
                values[i] = vdist(rng);
            }
            tree.process_results(leaves, b, policies, values);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  19x19 MCTS (batch=8): %d sims in %.1f ms (%.0f sims/sec, %.2f us/sim)\n",
               total_sims, ms, total_sims / (ms / 1000.0), ms * 1000.0 / total_sims);
    }
}

// ─── Main ───────────────────────────────────────────────────────

int main() {
    printf("=== MCTS C++ Tests ===\n\n");

    test_node_size();
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
