#pragma once
// MCTS (Monte Carlo Tree Search) for AlphaZero.
// Arena-allocated nodes, PUCT selection, virtual loss, batch leaf collection.

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include "go.h"

namespace mcts {

// ─── MCTSNode ────────────────────────────────────────────────────

struct MCTSNode {
    int action;           // action that led to this node (-1 for root)
    float prior;          // P(s,a) from NN policy
    int visit_count;      // N(s,a)
    float value_sum;      // W(s,a) — total value
    int virtual_loss;     // for parallel leaf selection
    int children_start;   // index into node pool (-1 if unexpanded)
    int num_children;     // number of children

    MCTSNode()
        : action(-1), prior(0.0f), visit_count(0), value_sum(0.0f),
          virtual_loss(0), children_start(-1), num_children(0) {}

    bool is_expanded() const { return children_start >= 0; }

    float q_value() const {
        int n = visit_count + virtual_loss;
        if (n == 0) return 0.0f;
        return (value_sum - static_cast<float>(virtual_loss)) / static_cast<float>(n);
    }
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
    float dirichlet_epsilon;  // fraction of noise to mix in (0.25)
    bool root_noise_applied;

    // ─── Node storage (arena) ───────────────────────────────────

    std::vector<MCTSNode> nodes;
    int root_idx;

    // Game state only for expanded nodes + leaves being evaluated.
    // Key = node index. Only populated when needed (not for all children).
    std::unordered_map<int, go::Game<N>> game_states;

    // ─── Constructor ────────────────────────────────────────────

    MCTSTree(const go::Game<N>& game, float c_puct_ = 1.5f,
             float dir_alpha = 0.03f, float dir_eps = 0.25f)
        : c_puct(c_puct_), dirichlet_alpha(dir_alpha),
          dirichlet_epsilon(dir_eps), root_noise_applied(false)
    {
        nodes.reserve(8192);
        game_states.reserve(4096);

        // Create root node
        root_idx = alloc_node();
        nodes[root_idx].action = -1;
        game_states[root_idx] = game;
    }

    // ─── Node allocation ────────────────────────────────────────

    int alloc_node() {
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back();
        return idx;
    }

    bool has_game_state(int idx) const {
        return game_states.count(idx) > 0;
    }

    // ─── PUCT selection ─────────────────────────────────────────

    float ucb_score(const MCTSNode& parent, const MCTSNode& child) const {
        int parent_visits = parent.visit_count + parent.virtual_loss;
        // Use max(1, N) to ensure exploration term is non-zero on first selection
        float sqrt_parent = std::sqrt(static_cast<float>(std::max(1, parent_visits)));
        float pb_c = c_puct * child.prior * sqrt_parent
                     / (1.0f + child.visit_count + child.virtual_loss);
        return child.q_value() + pb_c;
    }

    // Select best child by UCB score. Returns child index in node pool.
    int select_child(int node_idx) const {
        const MCTSNode& node = nodes[node_idx];
        int best = -1;
        float best_score = -1e9f;
        for (int i = 0; i < node.num_children; ++i) {
            int child_idx = node.children_start + i;
            float score = ucb_score(node, nodes[child_idx]);
            if (score > best_score) {
                best_score = score;
                best = child_idx;
            }
        }
        return best;
    }

    // ─── Select leaf (single path from root) ────────────────────
    // Returns path of node indices from root to leaf.
    // Applies virtual loss along the path.

    std::vector<int> select_leaf() {
        std::vector<int> path;
        int current = root_idx;
        path.push_back(current);

        while (nodes[current].is_expanded()) {
            // Terminal game state — don't descend further
            auto it = game_states.find(current);
            if (it != game_states.end() && it->second.status != go::PLAYING) break;

            int child = select_child(current);
            if (child < 0) break;

            // Apply virtual loss
            nodes[child].virtual_loss++;

            current = child;
            path.push_back(current);
        }

        return path;
    }

    // ─── Select multiple leaves (batch, using virtual loss) ─────

    struct LeafInfo {
        std::vector<int> path;
        int leaf_idx;
        bool needs_nn;  // false if terminal (value known)
        float terminal_value;
    };

    std::vector<LeafInfo> select_leaves(int num_leaves) {
        std::vector<LeafInfo> leaves;
        leaves.reserve(num_leaves);

        for (int i = 0; i < num_leaves; ++i) {
            auto path = select_leaf();
            int leaf = path.back();

            LeafInfo info;
            info.path = std::move(path);
            info.leaf_idx = leaf;

            // Check if leaf is already expanded (terminal node)
            if (nodes[leaf].is_expanded()) {
                info.needs_nn = false;
                // Terminal: evaluate from game outcome
                const auto& game = game_states.at(leaf);
                // Value from current player's perspective (who is about to move)
                // Game is over, so evaluate who won
                if (game.status == go::BLACK_WIN) {
                    info.terminal_value = (game.current_turn == go::BLACK) ? 1.0f : -1.0f;
                } else {
                    info.terminal_value = (game.current_turn == go::WHITE) ? 1.0f : -1.0f;
                }
            } else {
                info.needs_nn = true;
                info.terminal_value = 0.0f;
            }

            leaves.push_back(std::move(info));
        }

        return leaves;
    }

    // Fill observation buffer for leaves that need NN evaluation.
    // Returns number of leaves that need NN.
    int fill_observations(const std::vector<LeafInfo>& leaves, float* obs_buffer) {
        int count = 0;
        for (const auto& leaf : leaves) {
            if (!leaf.needs_nn) continue;

            int node_idx = leaf.leaf_idx;

            // Create game state for this leaf if not already present
            if (!has_game_state(node_idx)) {
                // Find parent (second-to-last in path)
                if (leaf.path.size() >= 2) {
                    int parent_idx = leaf.path[leaf.path.size() - 2];
                    game_states[node_idx] = game_states.at(parent_idx);
                    // Apply the action that led to this node
                    int action = nodes[node_idx].action;
                    game_states[node_idx].make_move(action);
                }
                // else: root and unexpanded — state already stored
            }

            game_states.at(node_idx).to_observation(obs_buffer + count * 17 * CELLS);
            count++;
        }
        return count;
    }

    // ─── Expand node with NN policy output ──────────────────────

    void expand(int node_idx, const float* policy) {
        MCTSNode& node = nodes[node_idx];
        if (node.is_expanded()) return;

        auto it = game_states.find(node_idx);
        if (it == game_states.end()) return;
        const auto& game = it->second;
        if (game.status != go::PLAYING) return;

        // Get legal action mask
        bool legal[ACTIONS];
        game.board.get_action_mask(game.current_turn, legal);

        // Mask and renormalize policy
        float masked[ACTIONS];
        float sum = 0.0f;
        for (int a = 0; a < ACTIONS; ++a) {
            masked[a] = legal[a] ? policy[a] : 0.0f;
            sum += masked[a];
        }

        // Normalize (handle degenerate case)
        if (sum > 1e-8f) {
            for (int a = 0; a < ACTIONS; ++a) masked[a] /= sum;
        } else {
            // Uniform over legal moves
            int n_legal = 0;
            for (int a = 0; a < ACTIONS; ++a) if (legal[a]) n_legal++;
            if (n_legal > 0) {
                float uniform = 1.0f / static_cast<float>(n_legal);
                for (int a = 0; a < ACTIONS; ++a)
                    masked[a] = legal[a] ? uniform : 0.0f;
            }
        }

        // Create children for legal actions
        node.children_start = static_cast<int>(nodes.size());
        node.num_children = 0;

        for (int a = 0; a < ACTIONS; ++a) {
            if (!legal[a]) continue;

            int child_idx = alloc_node();
            // Note: alloc_node may invalidate `node` reference if vector reallocates
            nodes[child_idx].action = a;
            nodes[child_idx].prior = masked[a];
            nodes[node_idx].num_children++;
        }
    }

    // ─── Backup value along path ────────────────────────────────
    // value: from the perspective of the player at the LEAF node

    void backup(const std::vector<int>& path, float value) {
        // Walk from leaf to root.
        // At each node, the value alternates sign because players alternate.
        float v = value;
        for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
            MCTSNode& node = nodes[path[i]];
            node.value_sum += v;
            node.visit_count++;
            if (node.virtual_loss > 0) node.virtual_loss--;
            v = -v;  // flip for parent
        }
    }

    // ─── Process NN results back into tree ──────────────────────
    // policies: (num_nn, ACTIONS), values: (num_nn,)
    // Expands leaves and backs up values.

    void process_results(const std::vector<LeafInfo>& leaves,
                         const float* policies, const float* values) {
        int nn_idx = 0;
        for (const auto& leaf : leaves) {
            if (leaf.needs_nn) {
                // Expand
                expand(leaf.leaf_idx, policies + nn_idx * ACTIONS);
                // Backup: value is from NN's perspective (current player at leaf)
                backup(leaf.path, values[nn_idx]);
                nn_idx++;
            } else {
                // Terminal — backup known value
                backup(leaf.path, leaf.terminal_value);
            }
        }
    }

    // ─── Dirichlet noise at root ────────────────────────────────

    void apply_dirichlet_noise(std::mt19937& rng) {
        if (root_noise_applied) return;
        MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded() || root.num_children == 0) return;

        // Generate Dirichlet noise
        std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);
        std::vector<float> noise(root.num_children);
        float noise_sum = 0.0f;
        for (int i = 0; i < root.num_children; ++i) {
            noise[i] = gamma(rng);
            noise_sum += noise[i];
        }
        if (noise_sum > 1e-8f) {
            for (int i = 0; i < root.num_children; ++i)
                noise[i] /= noise_sum;
        }

        // Mix noise into priors
        float eps = dirichlet_epsilon;
        for (int i = 0; i < root.num_children; ++i) {
            MCTSNode& child = nodes[root.children_start + i];
            child.prior = (1.0f - eps) * child.prior + eps * noise[i];
        }

        root_noise_applied = true;
    }

    // ─── Get action distribution from root ──────────────────────
    // Returns visit count distribution over all ACTIONS (size = N*N+1).
    // temperature: 1.0 = proportional to visits, 0.0 = argmax

    void get_policy(float* policy, float temperature = 1.0f) const {
        std::memset(policy, 0, ACTIONS * sizeof(float));

        const MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded()) return;

        if (temperature < 1e-6f) {
            // Argmax
            int best_act = -1;
            int best_visits = -1;
            for (int i = 0; i < root.num_children; ++i) {
                const MCTSNode& child = nodes[root.children_start + i];
                if (child.visit_count > best_visits) {
                    best_visits = child.visit_count;
                    best_act = child.action;
                }
            }
            if (best_act >= 0) policy[best_act] = 1.0f;
        } else {
            // Proportional to visit_count^(1/temperature)
            float sum = 0.0f;
            for (int i = 0; i < root.num_children; ++i) {
                const MCTSNode& child = nodes[root.children_start + i];
                float v = std::pow(static_cast<float>(child.visit_count),
                                   1.0f / temperature);
                policy[child.action] = v;
                sum += v;
            }
            if (sum > 1e-8f) {
                for (int a = 0; a < ACTIONS; ++a) policy[a] /= sum;
            }
        }
    }

    // Select best action (most visited child)
    int best_action() const {
        const MCTSNode& root = nodes[root_idx];
        if (!root.is_expanded()) return CELLS;  // pass

        int best = -1;
        int best_visits = -1;
        for (int i = 0; i < root.num_children; ++i) {
            const MCTSNode& child = nodes[root.children_start + i];
            if (child.visit_count > best_visits) {
                best_visits = child.visit_count;
                best = child.action;
            }
        }
        return best >= 0 ? best : CELLS;
    }

    // ─── Tree reuse: advance to child after playing a move ──────
    // Promotes the subtree of the chosen action as new root.

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
            if (!has_game_state(new_root)) {
                game_states[new_root] = game_states.at(root_idx);
                game_states[new_root].make_move(action);
            }
            root_idx = new_root;
        } else {
            // Action not in tree — create fresh root
            go::Game<N> new_game = game_states.at(root_idx);
            new_game.make_move(action);

            root_idx = alloc_node();
            nodes[root_idx].action = action;
            game_states[root_idx] = new_game;
        }

        root_noise_applied = false;
    }

    // ─── Reset tree (for new game) ──────────────────────────────

    void reset(const go::Game<N>& game) {
        nodes.clear();
        game_states.clear();
        nodes.reserve(8192);
        game_states.reserve(4096);

        root_idx = alloc_node();
        nodes[root_idx].action = -1;
        game_states[root_idx] = game;
        root_noise_applied = false;
    }

    // ─── Stats ──────────────────────────────────────────────────

    int root_visit_count() const { return nodes[root_idx].visit_count; }
    int num_nodes() const { return static_cast<int>(nodes.size()); }

    int num_game_states() const { return static_cast<int>(game_states.size()); }

    float root_value() const {
        const MCTSNode& root = nodes[root_idx];
        if (root.visit_count == 0) return 0.0f;
        return root.value_sum / static_cast<float>(root.visit_count);
    }

    // Get Q-values of root children for debugging
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

    // Get visit counts of root children
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
