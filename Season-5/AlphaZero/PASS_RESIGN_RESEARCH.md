# Pass & Resign in Go AI Self-Play Training

Research notes on how open-source Go AIs handle premature passing and
resignation during self-play training. Motivated by Phase 2 run5's
pass-collapse: 51% of iter 3 games ended at move 60-70, right after
the pass floor lifted.

---

## The Problem

During self-play training from scratch, the value/score head develops
biases before it can accurately evaluate territory. When both players
believe the game is "decided" (even incorrectly), they pass or resign
early. This creates a feedback loop:

1. Value/score head has bias → players think game is settled
2. Both pass early → short games
3. Short games → buffer lacks mid/late-game positions
4. Score head trains on biased data → bias persists or worsens

This is distinct from resign-collapse (value head memorizes → confidence
too high → resign at move 20). Pass-collapse happens through the game's
natural termination rule (two consecutive passes), which no safety
mechanism can block without breaking Go rules.

---

## How Open-Source Go AIs Handle It

### KataGo (most sophisticated)

**No pass floor.** KataGo solves pass-collapse structurally, not
with band-aids.

**Key mechanisms:**

1. **Ownership head as the structural fix.** Per-cell ownership
   predictions (which player owns each intersection) force the trunk
   to learn spatial territory understanding. The network must know
   *why* a position is won, not just *that* it's won. This prevents
   the "territory looks settled" confusion that causes premature
   passing. Weight: `1.5 / board_size²` per intersection.

2. **Games play to completion — no resignation.** When the losing
   side's MCTS winrate drops below 5% for 5 consecutive turns, visit
   count is reduced (saving compute) but the game continues. Training
   samples from clearly-decided positions are **downweighted to 10%**
   probability, not discarded. This ensures:
   - Late-game territory data stays in the buffer
   - The score/ownership heads see complete game outcomes
   - No resign-related data bias

3. **Tromp-Taylor scoring (modified).** After consecutive passes,
   moves in pass-alive territory are prohibited, and a tiny bias
   favors passing when it wouldn't change the score. Games end
   cleanly once territory is truly settled — not because the net
   thinks it's settled.

4. **Playout cap randomization.** 25% of moves get full search
   (600-1000 visits), 75% get fast search (100-200 visits). Only
   full-search positions train the policy head. All positions train
   value/ownership. This generates 3-4× more games per unit compute,
   which dilutes the impact of any remaining short games.

5. **Score distribution head.** KataGo predicts a full score
   probability distribution (PDF + CDF), not just a scalar. This
   gives the network calibrated uncertainty about the margin and
   prevents overconfident "game is decided" signals.

**Source:** [KataGo paper](https://ar5iv.labs.arxiv.org/html/1902.10565),
[KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)

### Leela Zero

**Had exactly our problem.** GitHub issue #152 documents pass-pass
endings (games of 1-4 moves) persisting even after 100K+ training
games. The model would play pass on move 1 because the policy head
learned "pass is a valid opening move" from MCTS visit noise.

**No special fix.** Leela Zero relied on brute-force self-play volume
to eventually overcome the problem. With millions of games, the cold
start bias gets diluted by better-quality games. This is considered
a weakness of the pure AlphaGo Zero approach — it works but wastes
enormous compute on degenerate games during cold start.

**Source:** [leela-zero/leela-zero#152](https://github.com/leela-zero/leela-zero/issues/152)

### AlphaGo Zero / AlphaZero

**Resign mechanism only (no pass handling):**
- Resign threshold `v_resign` calibrated so false-positive resignation
  rate stays below ~5%
- Periodically plays resigned games to completion to verify the
  threshold is correctly calibrated
- Paper acknowledges "early resignation can reduce diversity of games
  in the replay buffer and result in negative feedback loops"

No pass-specific mechanisms documented. The original paper trained
on 19x19 with massive compute (5000 TPUs, 4.9M games), so cold-start
pass-collapse was diluted by sheer volume.

**Source:** [AlphaGo Zero paper](https://www.nature.com/articles/nature24270)

### ELF OpenGo

Follows AlphaGo Zero closely. Resign threshold tuning but no
pass-specific fixes. Trained with 2000 GPUs — volume-based approach.

---

## Our Experience (Phase 2 Run 4-5)

### Run 4: Derived-value pass-collapse

The derived-value architecture (`value = tanh(scale * Σ(ownership))`)
caused the ownership head to learn "territory is settled" from
late-game data, which leaked into early-game policy. 51% of games
ended via pass-pass before move 80.

**Fix attempt (v1):** Pass floor `pass_min_move=60` — zero pass in
sampled action only. **Failed:** policy target still contained pass
visits from MCTS, so the net learned pass was valid.

**Fix (v2):** Zero pass in both the sampled action AND the stored
policy target. Pass probability in the opening dropped monotonically.

### Run 5: Score head pass-collapse

Score head architecture prevents value memorization but score bias
oscillation causes both players to think the game is decided. Once
the pass floor lifts at move 60, players pass immediately.

**Observations:**
- 0 games end before move 60 (pass floor works)
- 527/1025 games (51%) end at move 60-70
- 228/1025 games (22%) end at move 70-80
- Only 270/1025 games (26%) reach move 80+
- Iter 4: avg moves dropped to 74 (from 166 at iter 0)

---

## Techniques to Consider

### Tier 1: Proven (adopted by KataGo)

1. **Play games to completion, downweight decided positions.**
   Remove resignation entirely. When winrate < 5% for 5 turns,
   reduce MCTS visits to save compute but keep the game going.
   Downweight training samples from decided positions (10% weight).
   This ensures the buffer always contains full-game territory data.

2. **Strengthen ownership auxiliary loss.** The ownership head is
   the structural fix for pass-collapse — it forces spatial territory
   understanding. Our current ownership loss weight (1.5) is in
   KataGo's range, but KataGo normalizes by board size
   (`1.5 / N²`). Consider whether our weight is appropriate.

3. **Score distribution instead of scalar.** Predict a distribution
   over possible final scores, not just the mean. This gives
   calibrated uncertainty and prevents overconfident "game is decided"
   signals. More complex to implement.

### Tier 2: Pragmatic (quick fixes)

4. **Raise pass floor.** `pass_min_move=60` is too low — the model
   passes immediately at move 60. Raise to 100 or 120 (roughly 2/3
   of a typical 180-move game). This is a band-aid but buys time.

5. **Score bias regularization.** Add a loss term:
   `loss += α * (score.mean() - target.mean())²`
   This anchors the score head's mean prediction to the target mean,
   preventing the optimistic/pessimistic oscillation that makes both
   players think the game is decided.

6. **Playout cap randomization.** Cheap to implement, generates more
   games per unit compute, dilutes degenerate games.

### Tier 3: Experimental

7. **Asymmetric pass penalty.** During self-play, add a small
   negative reward for passing before move X. This discourages
   premature passing without removing it as a legal move.

8. **Pass temperature.** Apply extra temperature to the pass action
   during self-play (e.g., divide pass logit by 2). This makes pass
   less likely to be selected without zeroing it entirely.

9. **Progressive pass floor.** Start with pass_min_move=120 and
   gradually reduce it as training progresses and the ownership head
   matures. Requires tracking training progress.

---

## Recommendation for Next Run

Combine Tier 1 and Tier 2 fixes:

1. **Raise pass_min_move to 120** (immediate, one config change)
2. **Add score bias regularization** (one line in trainer)
3. **Disable resignation** and downweight decided positions (moderate
   code change in worker.h + trainer.py)

These three changes address the three problems we've observed:
- Pass-collapse (higher floor + eventually ownership head maturity)
- Score bias oscillation (regularization)
- Data quality from short games (no resign + downweight)

Longer term, consider implementing KataGo's playout cap
randomization and score distribution head.
