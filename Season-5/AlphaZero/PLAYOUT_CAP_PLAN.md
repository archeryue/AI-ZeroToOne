# Playout Cap Randomization — Implementation Plan

KataGo's playout cap randomization gives ~2.7× self-play speedup
with minimal quality loss. Plan to implement after seed 400 validates
the pass-floor + bias-reg + downweighting fixes.

---

## How it works

Each move gets an independent coin flip:
- **25% chance → full search** (400 sims)
- **75% chance → reduced search** (100 sims)

Training target split:
- **Policy**: trained from ALL moves (both full and reduced)
- **Value/score/ownership**: trained from **full-search moves only**

No positional bias — uniform random across the game.

## Speedup estimate

| | Current | With cap |
|---|---|---|
| Forward passes/iter | 1024 × 184 × 400 = 75M | 1024 × 184 × 175 = 28M |
| Self-play time/iter | ~30 min | ~12 min |
| 60 iters total | ~30 hours | ~12 hours |

The effective sims/move is `0.25 × 400 + 0.75 × 100 = 175`.

## Implementation

### 1. C++ worker.h (~10 lines)

In `complete_move`, before `run_simulations`:

```cpp
// Playout cap randomization: 25% full, 75% reduced
bool full_search = (dist(rng_) < 0.25f);
int sims = full_search ? cfg_.num_simulations : cfg_.reduced_simulations;

s.tree->run_simulations(sims, cfg_.virtual_loss_batch,
                        evaluator_, cfg_.add_noise, seed);

// Store flag in MoveRecord
rec.full_search = full_search;
```

Add to `SelfPlayConfig`:
```cpp
int reduced_simulations = 100;
float full_search_prob = 0.25f;
```

Add to `MoveRecord`:
```cpp
bool full_search = true;  // default true for backward compat
```

### 2. Python config (model/config.py)

```python
reduced_simulations: int = 100
full_search_prob: float = 0.25
```

### 3. Replay buffer (replay_buffer.py)

Add `full_search` bool array:
```python
self.full_search = np.ones(capacity, dtype=np.bool_)
```

Push/sample/save/load need to handle the new field.

### 4. Trainer (trainer.py)

Mask value/score/ownership losses to full-search positions only:

```python
# full_search_mask is (B,) bool tensor
if full_search_mask is not None:
    # Policy loss: all positions
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
    
    # Value/score/ownership: full-search only
    if full_search_mask.any():
        fs = full_search_mask
        score_loss = F.mse_loss(score_pred[fs], target_score[fs])
        ownership_loss = F.binary_cross_entropy_with_logits(
            ownership_logits[fs], target_ownership_01[fs])
    else:
        score_loss = torch.zeros((), device=self.device)
        ownership_loss = torch.zeros((), device=self.device)
```

### 5. Bindings (engine/bindings.cpp)

Expose new config fields and MoveRecord.full_search to Python.

### 6. Tests

Add to `_test_correctness.py`:
- Verify ~25% of buffer positions have `full_search=True`
- Verify policy targets exist for all positions
- Verify score/value targets exist only for full-search positions

## Files to change

| File | Change |
|---|---|
| `engine/worker.h` | Coin flip, reduced sims, flag in MoveRecord |
| `engine/bindings.cpp` | Expose new config fields |
| `model/config.py` | `reduced_simulations`, `full_search_prob` |
| `training/replay_buffer.py` | `full_search` array, save/load |
| `training/trainer.py` | Masked losses |
| `training/parallel_self_play.py` | Pass config to workers |
| `training/_test_correctness.py` | Verification |

Engine rebuild required (`setup.py build_ext --inplace`).

## Risks

- **Reduced search quality**: 100 sims on 13x13 may be too few for
  useful policy targets in complex positions. KataGo uses ~128 reduced
  on 19x19 with a stronger net. Monitor policy loss — if it rises
  significantly, increase reduced_simulations to 150-200.

- **Value target sparsity**: only 25% of positions train value/score.
  With 1024 games × 184 moves = ~188k positions/iter, 25% is ~47k
  positions. At batch 1024 and 50 steps, each step sees 1024 positions
  of which ~256 have value targets. Should be sufficient but monitor.

## When to implement

After seed 400 validates the 3 current fixes (pass_min=120, bias reg,
downweighting) over at least 5 iters. If those fixes work, playout cap
is pure speedup with no architectural risk.
