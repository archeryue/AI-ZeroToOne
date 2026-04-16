# Playout Cap Randomization — Implementation Plan

KataGo's playout cap randomization gives ~2.7× self-play speedup
with minimal quality loss. Plan to implement after seed 400 validates
the pass-floor + bias-reg + downweighting fixes.

---

## How it works

Each move gets an independent coin flip:
- **25% chance → full search** (400 sims)
- **75% chance → reduced search** (100 sims)

No positional bias — uniform random across the game.

**Key simplification vs KataGo:** KataGo masks value/ownership losses
on reduced-search moves because their value targets come from MCTS
rollout estimates. Our targets come from **game outcomes** — score
and ownership are computed by Tromp-Taylor flood-fill at game end,
not from search. So all targets are equally valid regardless of
search depth. **No masking needed.** All positions train all losses.

This means no `full_search` flag in the buffer or trainer. The
implementation is just a coin flip in worker.h to choose sims count.

## Speedup estimate

| | Current | With cap |
|---|---|---|
| Forward passes/iter | 1024 × 184 × 400 = 75M | 1024 × 184 × 175 = 28M |
| Self-play time/iter | ~30 min | ~12 min |
| 60 iters total | ~30 hours | ~12 hours |

The effective sims/move is `0.25 × 400 + 0.75 × 100 = 175`.

## Implementation

### 1. C++ worker.h (~5 lines)

In `complete_move`, before `run_simulations`:

```cpp
// Playout cap randomization: 25% full, 75% reduced
float full_prob = cfg_.full_search_prob;
int sims = (dist(rng_) < full_prob)
    ? cfg_.num_simulations
    : cfg_.reduced_simulations;

s.tree->run_simulations(sims, cfg_.virtual_loss_batch,
                        evaluator_, cfg_.add_noise, seed);
```

Add to `SelfPlayConfig`:
```cpp
int reduced_simulations = 100;
float full_search_prob = 0.25f;
```

### 2. Python config (model/config.py)

```python
reduced_simulations: int = 100
full_search_prob: float = 0.25
```

### 3. Bindings (engine/bindings.cpp)

Expose `reduced_simulations` and `full_search_prob` on the config.

### 4. parallel_self_play.py

Pass the new config fields to the C++ worker.

That's it. No buffer changes, no trainer changes, no test changes.

## Files to change

| File | Change |
|---|---|
| `engine/worker.h` | Coin flip, choose sims count (~5 lines) |
| `engine/bindings.cpp` | Expose 2 new config fields |
| `model/config.py` | `reduced_simulations`, `full_search_prob` |
| `training/parallel_self_play.py` | Pass config to workers |

Engine rebuild required (`setup.py build_ext --inplace`).

## Risks

- **Reduced search quality**: 100 sims on 13x13 may produce noisier
  policy targets in complex positions. Monitor policy loss — if it
  rises significantly, increase reduced_simulations to 150-200.

- **Policy target noise**: with 100 sims, MCTS visit distribution is
  more spread out (less concentrated on best move). This is actually
  mild regularization — similar to higher temperature. KataGo found
  this acceptable and even beneficial for exploration.

## When to implement

After seed 400 validates the 3 current fixes (pass_min=120, bias reg,
downweighting) over at least 5 iters. If those fixes work, playout cap
is pure speedup with no architectural risk.
