#pragma once
// Zobrist hashing for Go board positions.
// Deterministic PRNG seeded with a fixed value for reproducibility.

#include <cstdint>
#include <array>

namespace go {

// Simple splitmix64 PRNG (deterministic, high quality)
inline uint64_t splitmix64(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

template<int N>
struct ZobristTable {
    static constexpr int CELLS = N * N;

    // stone[color-1][pos]: color is BLACK=1, WHITE=2 → index 0 or 1
    std::array<std::array<uint64_t, CELLS>, 2> stone;
    uint64_t side_to_move;  // XOR when it's black's turn

    ZobristTable() {
        uint64_t state = 0xDEADBEEF42ULL;  // fixed seed
        for (int c = 0; c < 2; ++c)
            for (int p = 0; p < CELLS; ++p)
                stone[c][p] = splitmix64(state);
        side_to_move = splitmix64(state);
    }
};

// Global tables (one per board size, initialized at startup)
template<int N>
inline const ZobristTable<N>& zobrist() {
    static const ZobristTable<N> table;
    return table;
}

}  // namespace go
