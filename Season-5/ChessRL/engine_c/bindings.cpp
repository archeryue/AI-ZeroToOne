/*
 * bindings.cpp — pybind11 bindings for the xiangqi C++ engine.
 *
 * Exposes: Board, Move, Game, get_legal_moves, is_in_check,
 *          board_to_observation, get_action_mask, etc.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xiangqi.h"

namespace py = pybind11;
using namespace xiangqi;

// -----------------------------------------------------------------------
// board_to_observation: returns (15, 10, 9) float32 numpy array
// -----------------------------------------------------------------------
static py::array_t<float> py_board_to_observation(const Board& b, int8_t current_turn) {
    auto result = py::array_t<float>({15, ROWS, COLS});
    auto buf = result.mutable_unchecked<3>();

    // Zero-fill
    std::memset(result.mutable_data(), 0, 15 * ROWS * COLS * sizeof(float));

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int8_t p = b.grid[sq(r, c)];
            if (p == 0) continue;
            int pt = piece_type(p);      // 1-7
            int plane = pt - 1;          // 0-6
            if (p < 0) plane += 7;       // black: 7-13
            buf(plane, r, c) = 1.0f;
        }
    }

    // Turn plane
    if (current_turn == RED) {
        for (int r = 0; r < ROWS; ++r)
            for (int c = 0; c < COLS; ++c)
                buf(14, r, c) = 1.0f;
    }

    return result;
}

// -----------------------------------------------------------------------
// get_action_mask: returns (8100,) bool numpy array
// -----------------------------------------------------------------------
static py::array_t<bool> py_get_action_mask(const Board& b, int8_t color) {
    auto result = py::array_t<bool>(NUM_ACTIONS);
    auto buf = result.mutable_unchecked<1>();

    // Zero-fill
    std::memset(result.mutable_data(), 0, NUM_ACTIONS * sizeof(bool));

    MoveList legal;
    get_legal_moves(b, color, legal);

    for (int i = 0; i < legal.count; ++i) {
        Move& m = legal.moves[i];
        int fr = sq_row(m.from_sq), fc = sq_col(m.from_sq);
        int tr = sq_row(m.to_sq),   tc = sq_col(m.to_sq);
        int action = encode_action(fr, fc, tr, tc);
        buf(action) = true;
    }

    return result;
}

// -----------------------------------------------------------------------
// Helper: convert Python list[list[int]] to Board
// -----------------------------------------------------------------------
static Board board_from_pylist(const py::list& grid) {
    Board b;
    std::memset(b.grid, 0, BOARD_SIZE);
    for (int r = 0; r < ROWS; ++r) {
        py::list row = grid[r].cast<py::list>();
        for (int c = 0; c < COLS; ++c) {
            b.grid[sq(r, c)] = row[c].cast<int8_t>();
        }
    }
    b.update_general_cache();
    return b;
}

// -----------------------------------------------------------------------
// Module definition
// -----------------------------------------------------------------------
PYBIND11_MODULE(_xiangqi, m) {
    m.doc() = "Fast Chinese Chess engine in C++";

    // Constants
    m.attr("RED") = RED;
    m.attr("BLACK") = BLACK;
    m.attr("ROWS") = ROWS;
    m.attr("COLS") = COLS;
    m.attr("NUM_ACTIONS") = NUM_ACTIONS;
    m.attr("GENERAL") = GENERAL;
    m.attr("ADVISOR") = ADVISOR;
    m.attr("ELEPHANT") = ELEPHANT;
    m.attr("HORSE") = HORSE;
    m.attr("CHARIOT") = CHARIOT;
    m.attr("CANNON") = CANNON;
    m.attr("SOLDIER") = SOLDIER;

    // Status constants
    m.attr("STATUS_PLAYING") = STATUS_PLAYING;
    m.attr("STATUS_RED_WIN") = STATUS_RED_WIN;
    m.attr("STATUS_BLACK_WIN") = STATUS_BLACK_WIN;
    m.attr("STATUS_DRAW") = STATUS_DRAW;

    // Move
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readonly("from_sq", &Move::from_sq)
        .def_readonly("to_sq", &Move::to_sq)
        .def_readonly("captured", &Move::captured)
        .def_property_readonly("from_row", [](const Move& m) { return sq_row(m.from_sq); })
        .def_property_readonly("from_col", [](const Move& m) { return sq_col(m.from_sq); })
        .def_property_readonly("to_row", [](const Move& m) { return sq_row(m.to_sq); })
        .def_property_readonly("to_col", [](const Move& m) { return sq_col(m.to_sq); })
        .def("__repr__", [](const Move& m) {
            return "Move(" + std::to_string(sq_row(m.from_sq)) + "," +
                   std::to_string(sq_col(m.from_sq)) + "->" +
                   std::to_string(sq_row(m.to_sq)) + "," +
                   std::to_string(sq_col(m.to_sq)) + ")";
        });

    // Board
    py::class_<Board>(m, "Board")
        .def(py::init<>())  // default = initial position
        .def(py::init([](const py::list& grid) {
            return board_from_pylist(grid);
        }), py::arg("grid"))
        .def("get", &Board::get)
        .def("set", &Board::set)
        .def("copy", &Board::copy)
        .def("to_fen", &Board::to_fen)
        .def_static("from_fen", &Board::from_fen)
        .def("in_bounds", [](const Board&, int r, int c) { return in_bounds(r, c); })
        .def("color_of", [](const Board& b, int r, int c) -> int {
            return color_of(b.grid[sq(r, c)]);
        })
        .def("piece_type", [](const Board& b, int r, int c) -> int {
            return piece_type(b.grid[sq(r, c)]);
        })
        .def("find_general", [](const Board& b, int8_t color) -> py::tuple {
            int8_t pos = b.general_pos(color);
            if (pos < 0) throw py::value_error("General not found");
            return py::make_tuple(sq_row(pos), sq_col(pos));
        })
        .def_property_readonly("grid", [](const Board& b) -> py::list {
            py::list rows;
            for (int r = 0; r < ROWS; ++r) {
                py::list row;
                for (int c = 0; c < COLS; ++c) {
                    row.append((int)b.grid[sq(r, c)]);
                }
                rows.append(row);
            }
            return rows;
        });

    // Game
    py::class_<Game>(m, "Game")
        .def(py::init<>())
        .def_readwrite("current_turn", &Game::current_turn)
        .def_readwrite("status", &Game::status)
        .def_property_readonly("board", [](const Game& g) -> const Board& {
            return g.board;
        }, py::return_value_policy::reference_internal)
        .def("make_move", &Game::make_move)
        .def("simulate_action", &Game::simulate_action)
        .def("_check_game_over", &Game::check_game_over);

    // Free functions
    m.def("get_legal_moves", [](const Board& b, int8_t color) -> py::list {
        MoveList ml;
        get_legal_moves(b, color, ml);
        py::list result;
        for (int i = 0; i < ml.count; ++i) {
            result.append(ml.moves[i]);
        }
        return result;
    }, py::arg("board"), py::arg("color"));

    m.def("is_in_check", [](const Board& b, int8_t color) -> bool {
        return is_in_check(b, color);
    }, py::arg("board"), py::arg("color"));

    m.def("is_flying_general", [](const Board& b) -> bool {
        return is_flying_general(b);
    }, py::arg("board"));

    m.def("generate_all_moves", [](const Board& b, int8_t color) -> py::list {
        MoveList ml;
        generate_pseudo_moves(b, color, ml);
        py::list result;
        for (int i = 0; i < ml.count; ++i) {
            result.append(ml.moves[i]);
        }
        return result;
    }, py::arg("board"), py::arg("color"));

    m.def("is_checkmate", [](const Board& b, int8_t color) { return is_checkmate(b, color); });
    m.def("is_stalemate", [](const Board& b, int8_t color) { return is_stalemate(b, color); });

    // Combined functions
    m.def("board_to_observation", &py_board_to_observation,
          py::arg("board"), py::arg("current_turn"));
    m.def("get_action_mask", &py_get_action_mask,
          py::arg("board"), py::arg("color"));
    m.def("get_legal_action_indices", [](const Board& b, int8_t color) {
        std::vector<int> indices;
        get_legal_action_indices(b, color, indices);
        return indices;
    }, py::arg("board"), py::arg("color"));

    // Action encoding/decoding
    m.def("encode_move", &encode_action);
    m.def("decode_action", [](int action) -> py::tuple {
        int from_sq = action / BOARD_SIZE;
        int to_sq = action % BOARD_SIZE;
        return py::make_tuple(sq_row(from_sq), sq_col(from_sq),
                              sq_row(to_sq), sq_col(to_sq));
    });
}
