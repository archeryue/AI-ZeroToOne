// pybind11 bindings for Go engine + MCTS + SelfPlayWorker.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "go.h"
#include "mcts.h"
#include "worker.h"

namespace py = pybind11;

template<int N>
void bind_board(py::module_& m, const char* name) {
    using B = go::Board<N>;
    py::class_<B>(m, name)
        .def(py::init<>())
        .def("reset", &B::reset)
        .def("place_stone", [](B& b, int row, int col, uint8_t clr) {
            return b.place_stone(B::pos(row, col), clr);
        }, py::arg("row"), py::arg("col"), py::arg("color"))
        .def("is_legal", [](const B& b, int row, int col, uint8_t clr) {
            return b.is_legal(B::pos(row, col), clr);
        }, py::arg("row"), py::arg("col"), py::arg("color"))
        .def("get_legal_moves", [](const B& b, uint8_t clr) {
            auto moves = b.get_legal_moves(clr);
            std::vector<std::pair<int, int>> result;
            result.reserve(moves.size());
            for (int p : moves) {
                result.emplace_back(B::row(p), B::col(p));
            }
            return result;
        }, py::arg("color"))
        .def("score", &B::score, py::arg("komi"))
        .def("get_action_mask", [](const B& b, uint8_t clr) {
            py::array_t<bool> mask(B::ACTIONS);
            b.get_action_mask(clr, mask.mutable_data());
            return mask;
        }, py::arg("color"))
        .def("board_grid", [](const B& b) {
            py::array_t<uint8_t> grid({N, N});
            auto buf = grid.mutable_unchecked<2>();
            for (int r = 0; r < N; ++r)
                for (int c = 0; c < N; ++c)
                    buf(r, c) = b.color[B::pos(r, c)];
            return grid;
        })
        .def_property_readonly("ko_point", [](const B& b) -> py::object {
            if (b.ko_point == go::NO_KO) return py::none();
            return py::make_tuple(B::row(b.ko_point), B::col(b.ko_point));
        })
        .def_property_readonly("hash", [](const B& b) { return b.hash; })
        ;
}

template<int N>
void bind_game(py::module_& m, const char* name) {
    using G = go::Game<N>;
    using B = go::Board<N>;
    py::class_<G>(m, name)
        .def(py::init<float>(), py::arg("komi") = 7.5f)
        .def("make_move", [](G& g, int row, int col) {
            return g.make_move(B::pos(row, col));
        }, py::arg("row"), py::arg("col"))
        .def("pass_move", [](G& g) {
            return g.make_move(G::CELLS);
        })
        .def("resign", &G::resign, py::arg("color"))
        .def("is_legal", [](const G& g, int row, int col) {
            return g.board.is_legal(B::pos(row, col), g.current_turn);
        }, py::arg("row"), py::arg("col"))
        .def("get_legal_moves", [](const G& g) {
            auto moves = g.board.get_legal_moves(g.current_turn);
            std::vector<std::pair<int, int>> result;
            result.reserve(moves.size());
            for (int p : moves) {
                result.emplace_back(B::row(p), B::col(p));
            }
            return result;
        })
        .def("score", [](const G& g) {
            return g.board.score(g.komi);
        })
        .def("to_observation", [](const G& g) {
            py::array_t<float> obs({17, N, N});
            g.to_observation(obs.mutable_data());
            return obs;
        })
        .def("get_action_mask", [](const G& g) {
            py::array_t<bool> mask(G::ACTIONS);
            g.board.get_action_mask(g.current_turn, mask.mutable_data());
            return mask;
        })
        .def("board_grid", [](const G& g) {
            py::array_t<uint8_t> grid({N, N});
            auto buf = grid.mutable_unchecked<2>();
            for (int r = 0; r < N; ++r)
                for (int c = 0; c < N; ++c)
                    buf(r, c) = g.board.color[B::pos(r, c)];
            return grid;
        })
        .def("compute_ownership", [](const G& g) {
            // Tromp-Taylor per-cell ownership at the current board
            // state. Returns int8 (N, N) with +1 BLACK / -1 WHITE / 0 dame.
            // Used by tests; production self-play computes this in
            // worker.h::finish_game so it never crosses the bindings.
            py::array_t<int8_t> own({N, N});
            g.board.compute_ownership(own.mutable_data());
            return own;
        })
        .def_property_readonly("current_turn", [](const G& g) { return (int)g.current_turn; })
        .def_property_readonly("status", [](const G& g) { return (int)g.status; })
        .def_property_readonly("move_count", [](const G& g) { return g.move_count; })
        .def_property_readonly("consecutive_passes", [](const G& g) { return g.consecutive_passes; })
        .def_readonly("komi", &G::komi)
        .def_property_readonly("captured", [](const G& g) {
            return py::make_tuple(g.captured[go::BLACK], g.captured[go::WHITE]);
        })
        ;
}

template<int N>
void bind_mcts(py::module_& m, const char* name) {
    using Tree = mcts::MCTSTree<N>;
    using G = go::Game<N>;
    constexpr int ACTIONS = N * N + 1;
    constexpr int OBS_PLANES = 17;

    py::class_<Tree>(m, name)
        .def(py::init<const G&, float, float, float>(),
             py::arg("game"),
             py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.03f,
             py::arg("dirichlet_epsilon") = 0.25f)

        // Run N simulations with a callable NN evaluator.
        // evaluator(obs_batch) -> (policies, values)
        .def("run_simulations", [](Tree& tree, int num_sims, int leaves_per_sim,
                                    py::function evaluator, bool add_noise, int seed) {
            std::mt19937 rng(seed);
            constexpr int MAX_BATCH = 64;
            mcts::LeafInfo leaves[MAX_BATCH];

            for (int sim = 0; sim < num_sims; sim += leaves_per_sim) {
                int batch = std::min(leaves_per_sim, num_sims - sim);
                batch = std::min(batch, MAX_BATCH);

                tree.select_leaves(leaves, batch);

                // Apply Dirichlet noise on first expansion
                if (add_noise && !tree.root_noise_applied
                    && tree.nodes[tree.root_idx].is_expanded()) {
                    tree.apply_dirichlet_noise(rng);
                }

                int nn_count = 0;
                for (int i = 0; i < batch; ++i) if (leaves[i].needs_nn) nn_count++;

                if (nn_count > 0) {
                    py::array_t<float> obs_batch({nn_count, OBS_PLANES, N, N});
                    tree.fill_observations(leaves, batch, obs_batch.mutable_data());

                    py::tuple result = evaluator(obs_batch);
                    py::array_t<float> policies = result[0].cast<py::array_t<float>>();
                    py::array_t<float> values = result[1].cast<py::array_t<float>>();

                    tree.process_results(leaves, batch, policies.data(), values.data());
                } else {
                    tree.process_results(leaves, batch, nullptr, nullptr);
                }
            }
        }, py::arg("num_sims"), py::arg("leaves_per_sim") = 8,
           py::arg("evaluator") = py::none(),
           py::arg("add_noise") = true, py::arg("seed") = 42)

        // Get policy from visit counts
        .def("get_policy", [](const Tree& tree, float temperature) {
            py::array_t<float> policy(ACTIONS);
            tree.get_policy(policy.mutable_data(), temperature);
            return policy;
        }, py::arg("temperature") = 1.0f)

        .def("best_action", &Tree::best_action)

        .def("advance", [](Tree& tree, int action) {
            tree.advance(action);
        }, py::arg("action"))

        .def("advance_rc", [](Tree& tree, int row, int col) {
            tree.advance(row * N + col);
        }, py::arg("row"), py::arg("col"))

        .def("reset", &Tree::reset, py::arg("game"))

        // Stats
        .def_property_readonly("root_visit_count", &Tree::root_visit_count)
        .def_property_readonly("root_value", &Tree::root_value)
        .def_property_readonly("num_nodes", &Tree::num_nodes)
        .def_property_readonly("num_game_states", &Tree::num_game_states)
        .def("root_children_visits", &Tree::root_children_visits)
        .def("root_children_q", &Tree::root_children_q)
        ;
}

template<int N>
void bind_worker(py::module_& m, const char* name) {
    using W = alphazero::SelfPlayWorker<N>;
    using Cfg = alphazero::SelfPlayConfig;

    py::class_<W>(m, name)
        .def(py::init<int, const Cfg&, int>(),
             py::arg("num_games"), py::arg("config"), py::arg("seed") = 42)

        // tick_select: GIL released for C++ parallelism
        .def("tick_select", [](W& w, py::array_t<float> obs_out) {
            float* ptr = obs_out.mutable_data();
            py::gil_scoped_release release;
            return w.tick_select(ptr);
        }, py::arg("obs_out"))

        // tick_process: GIL released for C++ parallelism
        .def("tick_process", [](W& w, py::object policies_obj, py::object values_obj) {
            const float* p = nullptr;
            const float* v = nullptr;
            py::array_t<float> p_arr, v_arr;  // prevent early dealloc
            if (!policies_obj.is_none()) {
                p_arr = policies_obj.cast<py::array_t<float>>();
                v_arr = values_obj.cast<py::array_t<float>>();
                p = p_arr.data();
                v = v_arr.data();
            }
            py::gil_scoped_release release;
            w.tick_process(p, v);
        }, py::arg("policies") = py::none(), py::arg("values") = py::none())

        .def("restart_completed", [](W& w) {
            py::gil_scoped_release release;
            w.restart_completed();
        })

        .def("harvest", [](W& w) {
            auto r = w.harvest();
            constexpr int OBS_SIZE = 17 * N * N;
            constexpr int ACTIONS = N * N + 1;
            constexpr int CELLS = N * N;
            int n = r.count;
            if (n == 0) {
                auto obs = py::array_t<float>(std::vector<ssize_t>{0, 17, N, N});
                auto pol = py::array_t<float>(std::vector<ssize_t>{0, ACTIONS});
                auto val = py::array_t<float>(std::vector<ssize_t>{0});
                auto own = py::array_t<int8_t>(std::vector<ssize_t>{0, N, N});
                return py::make_tuple(obs, pol, val, own, (int)0);
            }
            auto obs = py::array_t<float>(std::vector<ssize_t>{n, 17, N, N});
            std::memcpy(obs.mutable_data(), r.obs.data(), n * OBS_SIZE * sizeof(float));
            auto pol = py::array_t<float>(std::vector<ssize_t>{n, ACTIONS});
            std::memcpy(pol.mutable_data(), r.policy.data(), n * ACTIONS * sizeof(float));
            auto val = py::array_t<float>(std::vector<ssize_t>{n});
            std::memcpy(val.mutable_data(), r.value.data(), n * sizeof(float));
            auto own = py::array_t<int8_t>(std::vector<ssize_t>{n, N, N});
            std::memcpy(own.mutable_data(), r.ownership.data(), n * CELLS * sizeof(int8_t));
            return py::make_tuple(obs, pol, val, own, (int)n);
        })

        .def_property_readonly("active_count", &W::active_count)
        .def_property_readonly("games_done", &W::games_done)
        .def_property_readonly("completed_count", &W::completed_count)

        // Diagnostic: max tree-node count across all game slots. Used by
        // the tree-cap test to verify MAX_TREE_NODES is honored.
        .def("max_tree_nodes", &W::max_tree_nodes)
        .def_property_readonly_static("MAX_TREE_NODES",
            [](py::object) { return (int)W::MAX_TREE_NODES; })
        ;
}

PYBIND11_MODULE(go_engine, m) {
    m.doc() = "High-performance C++ Go engine for AlphaZero training";

    // Constants
    m.attr("EMPTY") = (int)go::EMPTY;
    m.attr("BLACK") = (int)go::BLACK;
    m.attr("WHITE") = (int)go::WHITE;
    m.attr("PLAYING") = (int)go::PLAYING;
    m.attr("BLACK_WIN") = (int)go::BLACK_WIN;
    m.attr("WHITE_WIN") = (int)go::WHITE_WIN;

    bind_board<9>(m, "Board9");
    bind_board<13>(m, "Board13");
    bind_board<19>(m, "Board19");

    bind_game<9>(m, "Game9");
    bind_game<13>(m, "Game13");
    bind_game<19>(m, "Game19");

    bind_mcts<9>(m, "MCTSTree9");
    bind_mcts<13>(m, "MCTSTree13");
    bind_mcts<19>(m, "MCTSTree19");

    // SelfPlayConfig
    using Cfg = alphazero::SelfPlayConfig;
    py::class_<Cfg>(m, "SelfPlayConfig")
        .def(py::init<>())
        .def_readwrite("komi", &Cfg::komi)
        .def_readwrite("c_puct", &Cfg::c_puct)
        .def_readwrite("dirichlet_alpha", &Cfg::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &Cfg::dirichlet_epsilon)
        .def_readwrite("vl_batch", &Cfg::vl_batch)
        .def_readwrite("num_sims", &Cfg::num_sims)
        .def_readwrite("temp_moves", &Cfg::temp_moves)
        .def_readwrite("temp_high", &Cfg::temp_high)
        .def_readwrite("temp_low", &Cfg::temp_low)
        .def_readwrite("resign_threshold", &Cfg::resign_threshold)
        .def_readwrite("resign_consecutive", &Cfg::resign_consecutive)
        .def_readwrite("resign_min_move", &Cfg::resign_min_move)
        .def_readwrite("resign_disabled_frac", &Cfg::resign_disabled_frac)
        .def_readwrite("resign_min_child_visits_frac",
                       &Cfg::resign_min_child_visits_frac)
        .def_readwrite("max_game_moves", &Cfg::max_game_moves)
        .def_readwrite("pass_min_move", &Cfg::pass_min_move)
        ;

    bind_worker<9>(m, "SelfPlayWorker9");
    bind_worker<13>(m, "SelfPlayWorker13");
    bind_worker<19>(m, "SelfPlayWorker19");
}
