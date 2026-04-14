#pragma once
// MCTS (Monte Carlo Tree Search) for AlphaZero.
// Arena-allocated nodes, PUCT selection, virtual loss, batch leaf collection.
//
// Performance-critical design choices:
//   - Game states in deque pool (no reallocation of 9KB Game objects)
//   - MCTSNode ~32 bytes for cache efficiency
//   - Fixed-size path buffers (no heap alloc in hot path)
//   - Bulk child allocation in expand() (single resize)
//   - is_terminal flag avoids game-state lookup during traversal

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <deque>
#include <algorithm>
#include <random>
#include "go.h"

namespace mcts {

// ─── MCTSNode (exactly 32 bytes — 2 per 64-byte cache line) ─────

struct MCTSNode {
    // Hot fields (accessed in UCB scoring inner loop)
    float prior;            // P(s,a) from NN policy
    float value_sum;        // W(s,a) — total value
    int visit_count;        // N(s,a)
    int16_t virtual_loss;   // for parallel leaf selection (max ~8)
    int16_t num_children;   // number of children (max 362)
    int children_start;     // index into node pool (-1 if unexpanded)
    // Cooler fields
    int action;             // action that led to this node (-1 for root)
    int game_idx;           // index into game_pool (-1 if no state)
    bool is_terminal;       // true if game is over at this node

    MCTSNode()
        : prior(0.0f), value_sum(0.0f), visit_count(0),
          virtual_loss(0), num_children(0), children_start(-1),
          action(-1), game_idx(-1), is_terminal(false) {}

    bool is_expanded() const { return children_start >= 0; }

    float q_value() const {
        int n = visit_count + virtual_loss;
        if (n == 0) return 0.0f;
        return (value_sum - static_cast<float>(virtual_loss)) / static_cast<float>(n);
    }
};

// ─── Fixed-size path buffer ─────────────────────────────────────
// MCTS tree depth is usually ~10-30, but with tree reuse across many
// moves the effective depth can grow unboundedly. 256 is a safe bound
// for any realistic Go position; select_leaf also hard-caps against
// this limit to prevent out-of-bounds writes.

static constexpr int MAX_PATH_DEPTH = 256;

struct LeafInfo {
    int path[MAX_PATH_DEPTH];
    int path_len;
    int leaf_idx;
    bool needs_nn;
    float terminal_value;
};

// ─── MCTSTree<N> ─────────────────────────────────────────────────

template<int N>
class MCTSTree {
public:
    static constexpr int CELLS = N * N;
    static constexpr int ACTIONS = CELLS + 1;  // board positions + pass

    // ─── Configuration ──────────────────────────────────────────

    float c_puct;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    bool root_noise_applied;

    // ─── Node storage (arena, 32-byte aligned) ──────────────────

    std::vector<MCTSNode> nodes;
    int root_idx;

    // Game state pool — compact, indexed by MCTSNode::game_idx.
    // Only expanded nodes and leaves under evaluation get a game state.
    std::deque<go::Game<N>> game_pool;

    // ─── Constructor ────────────────────────────────────────────

    MCTSTree(const go::Game<N>& game, float c_puct_ = 1.5f,
             float dir_alpha = 0.03f, float dir_eps = 0.25f)
        : c_puct(c_puct_), dirichlet_alpha(dir_alpha),
          dirichlet_epsilon(dir_eps), root_noise_applied(false)
    {
        // Reserve a modest baseline. The per-game upper bound is enforced
        // by SelfPlayWorker::MAX_TREE_NODES (200k), which triggers a
        // reset mid-game when the tree gets too big. A bigger reserve
        // just wastes RAM across 256 parallel trees — 131072 × 32 B ×
        // 256 = 1 GB of untouched capacity, which contributed to the
        // Phase 2 dryrun OOM.
        nodes.reserve(16384);


        root_idx = alloc_node();
        nodes[root_idx].action = -1;
        nodes[root_idx].game_idx = alloc_game_state();
        game_pool[nodes[root_idx].game_idx] = game;
    }

    // ─── Allocation ─────────────────────────────────────────────

    int alloc_node() {
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back();
        return idx;
    }

    int alloc_game_state() {
        int idx = static_cast<int>(game_pool.size());
        game_pool.emplace_back();
        return idx;
    }

    // ─── PUCT selection ─────────────────────────────────────────

    float ucb_score(const MCTSNode& parent, const MCTSNode& child) const {
        int parent_visits = parent.visit_count + parent.virtual_loss;
        float sqrt_parent = std::sqrt(static_cast<float>(
            parent_visits > 0 ? parent_visits : 1));
        float pb_c = c_puct * child.prior * sqrt_parent
                     / (1.0f + child.visit_count + child.virtual_loss);
        return child.q_value() + pb_c;
    }

    int select_child(int node_idx) const {
        const MCTSNode& node = nodes[node_idx];
        const MCTSNode* children = &nodes[node.children_start];
        int best = -1;
        float best_score = -1e9f;

        for (int i = 0; i < node.num_children; ++i) {
            float score = ucb_score(node, children[i]);
            if (score > best_score) {
                best_score = score;
                best = i;
            }
        }
        return best >= 0 ? node.children_start + best : -1;
    }

    // ─── Select leaf ────────────────────────────────────────────
    // Writes path into out_path, returns path length.
    // Applies virtual loss along the path.

    int select_leaf(int* out_path) {
        int len = 0;
        int current = root_idx;
        out_path[len++] = current;

        while (nodes[current].is_expanded() && !nodes[current].is_terminal) {
            if (len >= MAX_PATH_DEPTH) break;  // hard cap — prevents OOB
            int child = select_child(current);
            if (child < 0) break;

            nodes[child].virtual_loss++;
            current = child;
            out_path[len++] = current;
        }

        return len;
    }

    // ─── Select multiple leaves ─────────────────────────────────

    int select_leaves(LeafInfo* leaves, int num_leaves) {
        for (int i = 0; i < num_leaves; ++i) {
            LeafInfo& info = leaves[i];
            info.path_len = select_leaf(info.path);
            info.leaf_idx = info.path[info.path_len - 1];

            const MCTSNode& leaf = nodes[info.leaf_idx];

            if (leaf.is_expanded() && leaf.is_terminal) {
                info.needs_nn = false;
                const auto& game = game_pool[leaf.game_idx];
                // Value from current player's perspective
                if (game.status == go::BLACK_WIN) {
                    info.terminal_value = (game.current_turn == go::BLACK) ? 1.0f : -1.0f;
                } else {
                    info.terminal_value = (game.current_turn == go::WHITE) ? 1.0f : -1.0f;
                }
            } else {
                info.needs_nn = true;
                info.terminal_value = 0.0f;
            }
        }
        return num_leaves;
    }

    // ─── Fill observations for NN evaluation ────────────────────

    int fill_observations(const LeafInfo* leaves, int num_leaves, float* obs_buffer) {
        int count = 0;
        for (int i = 0; i < num_leaves; ++i) {
            const LeafInfo& leaf = leaves[i];
            if (!leaf.needs_nn) continue;

            int node_idx = leaf.leaf_idx;
            MCTSNode& node = nodes[node_idx];

            // Create game state if not present
            if (node.game_idx < 0) {
                if (leaf.path_len >= 2) {
                    int parent_idx = leaf.path[leaf.path_len - 2];
                    int parent_gi = nodes[parent_idx].game_idx;

                    int gi = alloc_game_state();
                    node.game_idx = gi;
                    game_pool[gi] = game_pool[parent_gi];
                    game_pool[gi].make_move(node.action);
                }
            }

            game_pool[node.game_idx].to_observation(obs_buffer + count * 17 * CELLS);
            count++;
        }
        return count;
    }

    // ─── Expand node ────────────────────────────────────────────

    void expand(int node_idx, const float* policy) {
        MCTSNode& node = nodes[node_idx];
        if (node.is_expanded()) return;

        int gi = node.game_idx;
        if (gi < 0) return;

        const auto& game = game_pool[gi];
        if (game.status != go::PLAYING) {
            node.is_terminal = true;
            // Mark as "expanded" with 0 children so we don't re-enter
            node.children_start = 0;
            node.num_children = 0;
            return;
        }

        // Get legal action mask
        bool legal[ACTIONS];
        game.board.get_action_mask(game.current_turn, legal);

        // Count legal actions
        int n_legal = 0;
        for (int a = 0; a < ACTIONS; ++a) if (legal[a]) n_legal++;
        if (n_legal == 0) return;

        // Mask and renormalize policy
        float masked[ACTIONS];
        float sum = 0.0f;
        for (int a = 0; a < ACTIONS; ++a) {
            masked[a] = legal[a] ? policy[a] : 0.0f;
            sum += masked[a];
        }

        if (sum > 1e-8f) {
            float inv_sum = 1.0f / sum;
            for (int a = 0; a < ACTIONS; ++a) masked[a] *= inv_sum;
        } else {
            float uniform = 1.0f / static_cast<float>(n_legal);
            for (int a = 0; a < ACTIONS; ++a)
                masked[a] = legal[a] ? uniform : 0.0f;
        }

        // Bulk-allocate all children at once (single resize, no realloc risk)
        int first_child = static_cast<int>(nodes.size());
        nodes.resize(nodes.size() + n_legal);

        // Use node_idx (not reference) since resize may have moved memory
        nodes[node_idx].children_start = first_child;
        nodes[node_idx].num_children = static_cast<int16_t>(n_legal);

        int child_offset = 0;
        for (int a = 0; a < ACTIONS; ++a) {
            if (!legal[a]) continue;
            // nodes were already default-constructed by resize()
            nodes[first_child + child_offset].action = a;
            nodes[first_child + child_offset].prior = masked[a];
            child_offset++;
        }
    }

    // ─── Backup ─────────────────────────────────────────────────

    void backup(const int* path, int path_len, float value) {
        // Negate: NN/terminal value is from the leaf's current player perspective,
        // but node values are stored from the parent's perspective (Convention 2)
        // so that ucb_score can use child.q_value() directly without negation.
        float v = -value;
        for (int i = path_len - 1; i >= 0; --i) {
            MCTSNode& node = nodes[path[i]];
            node.value_sum += v;
            node.visit_count++;
            if (node.virtual_loss > 0) node.virtual_loss--;
            v = -v;
        }
    }

    // ─── Process NN results ─────────────────────────────────────

    void process_results(const LeafInfo* leaves, int num_leaves,
                         const float* policies, const float* values) {
        int nn_idx = 0;
        for (int i = 0; i < num_leaves; ++i) {
            const LeafInfo& leaf = leaves[i];
            if (leaf.needs_nn) {
                expand(leaf.leaf_idx, policies + nn_idx * ACTIONS);
                backup(leaf.path, leaf.path_len, values[nn_idx]);
                nn_idx++;
            } else {
                backup(leaf.path, leaf.path_len, leaf.terminal_value);
            }
        }
    }

    // ─── Dirichlet noise at root ────────────────────────────────

    void apply_dirichlet_noise(std::mt19937& rng) {
        if (root_noise_applied) return;
        MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded() || root.num_children == 0) return;

        std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);
        int nc = root.num_children;

        // Stack-allocate noise (max 362 for 19x19)
        float noise[ACTIONS];
        float noise_sum = 0.0f;
        for (int i = 0; i < nc; ++i) {
            noise[i] = gamma(rng);
            noise_sum += noise[i];
        }
        if (noise_sum > 1e-8f) {
            float inv = 1.0f / noise_sum;
            for (int i = 0; i < nc; ++i) noise[i] *= inv;
        }

        float eps = dirichlet_epsilon;
        float one_minus_eps = 1.0f - eps;
        MCTSNode* children = &nodes[root.children_start];
        for (int i = 0; i < nc; ++i) {
            children[i].prior = one_minus_eps * children[i].prior + eps * noise[i];
        }

        root_noise_applied = true;
    }

    // ─── Get action distribution ────────────────────────────────

    void get_policy(float* policy, float temperature = 1.0f) const {
        std::memset(policy, 0, ACTIONS * sizeof(float));

        const MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded()) return;

        const MCTSNode* children = &nodes[root.children_start];
        int nc = root.num_children;

        if (temperature < 1e-6f) {
            // Argmax
            int best_act = -1;
            int best_visits = -1;
            for (int i = 0; i < nc; ++i) {
                if (children[i].visit_count > best_visits) {
                    best_visits = children[i].visit_count;
                    best_act = children[i].action;
                }
            }
            if (best_act >= 0) policy[best_act] = 1.0f;
        } else {
            float inv_temp = 1.0f / temperature;
            float sum = 0.0f;
            for (int i = 0; i < nc; ++i) {
                float v = std::pow(static_cast<float>(children[i].visit_count), inv_temp);
                policy[children[i].action] = v;
                sum += v;
            }
            if (sum > 1e-8f) {
                float inv_sum = 1.0f / sum;
                for (int a = 0; a < ACTIONS; ++a) policy[a] *= inv_sum;
            }
        }
    }

    int best_action() const {
        const MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded()) return CELLS;

        const MCTSNode* children = &nodes[root.children_start];
        int best = -1;
        int best_visits = -1;
        for (int i = 0; i < root.num_children; ++i) {
            if (children[i].visit_count > best_visits) {
                best_visits = children[i].visit_count;
                best = children[i].action;
            }
        }
        return best >= 0 ? best : CELLS;
    }

    // ─── Tree reuse ─────────────────────────────────────────────

    void advance(int action) {
        const MCTSNode& root = nodes[root_idx];
        int new_root = -1;

        if (root.is_expanded()) {
            for (int i = 0; i < root.num_children; ++i) {
                int child_idx = root.children_start + i;
                if (nodes[child_idx].action == action) {
                    new_root = child_idx;
                    break;
                }
            }
        }

        if (new_root >= 0) {
            // Ensure new root has game state
            if (nodes[new_root].game_idx < 0) {
                int parent_gi = nodes[root_idx].game_idx;
                int gi = alloc_game_state();
                nodes[new_root].game_idx = gi;
                game_pool[gi] = game_pool[parent_gi];
                game_pool[gi].make_move(action);
            }
            root_idx = new_root;
        } else {
            // Action not in tree — create fresh root
            int parent_gi = nodes[root_idx].game_idx;
            go::Game<N> new_game = game_pool[parent_gi];
            new_game.make_move(action);

            root_idx = alloc_node();
            nodes[root_idx].action = action;
            int gi = alloc_game_state();
            nodes[root_idx].game_idx = gi;
            game_pool[gi] = new_game;
        }

        root_noise_applied = false;
    }

    // ─── Reset ──────────────────────────────────────────────────

    void reset(const go::Game<N>& game) {
        nodes.clear();
        game_pool.clear();
        // Reserve a modest baseline. The per-game upper bound is enforced
        // by SelfPlayWorker::MAX_TREE_NODES (200k), which triggers a
        // reset mid-game when the tree gets too big. A bigger reserve
        // just wastes RAM across 256 parallel trees — 131072 × 32 B ×
        // 256 = 1 GB of untouched capacity, which contributed to the
        // Phase 2 dryrun OOM.
        nodes.reserve(16384);


        root_idx = alloc_node();
        nodes[root_idx].action = -1;
        int gi = alloc_game_state();
        nodes[root_idx].game_idx = gi;
        game_pool[gi] = game;
        root_noise_applied = false;
    }

    // ─── Stats ──────────────────────────────────────────────────

    int root_visit_count() const { return nodes[root_idx].visit_count; }
    int num_nodes() const { return static_cast<int>(nodes.size()); }
    int num_game_states() const { return static_cast<int>(game_pool.size()); }

    float root_value() const {
        const MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded() || root.num_children == 0) return 0.0f;
        // Use the most-visited child's Q-value, which is already stored
        // from the root's (parent's) perspective — no sign flip needed.
        const MCTSNode* children = &nodes[root.children_start];
        int best = 0;
        for (int i = 1; i < root.num_children; ++i) {
            if (children[i].visit_count > children[best].visit_count)
                best = i;
        }
        return children[best].q_value();
    }

    std::vector<std::pair<int, float>> root_children_q() const {
        std::vector<std::pair<int, float>> result;
        const MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded()) return result;
        for (int i = 0; i < root.num_children; ++i) {
            const MCTSNode& child = nodes[root.children_start + i];
            result.emplace_back(child.action, child.q_value());
        }
        return result;
    }

    std::vector<std::pair<int, int>> root_children_visits() const {
        std::vector<std::pair<int, int>> result;
        const MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded()) return result;
        for (int i = 0; i < root.num_children; ++i) {
            const MCTSNode& child = nodes[root.children_start + i];
            result.emplace_back(child.action, child.visit_count);
        }
        return result;
    }
};

extern template class MCTSTree<9>;
extern template class MCTSTree<13>;
extern template class MCTSTree<19>;

}  // namespace mcts
