"""Circular replay buffer for AlphaZero self-play data.

Design note (run-2 redesign):
  Earlier the buffer stored 8 augmented copies of every position at
  push time. With ~165K positions per iter and 8-fold augmentation,
  push() ran 1.32M times per iter into a 300K-capacity buffer,
  wrapping the buffer ~4.4× per iter. Effective history was the last
  ~37.5K positions of the most-recent self-play iter — the trainer
  was always fitting on its own just-played games with no real
  diversity.

  The fix: store ONE raw position per push, apply a random symmetry
  per sample at training time. Same memory, **8× the effective
  diversity**, history goes from ~0.23 iters to ~1.8 iters. Vectorized
  in `sample()` by bucketing the batch by symmetry and rotating each
  bucket once.

Phase 2 note (uint8 obs storage):
  Every obs plane cell is exactly 0 or 1 (stones + color-to-play —
  verified in _test_correctness.py). Storing them as uint8 instead of
  float32 is a lossless 4× memory cut: the Phase 2 13x13 buffer at
  1M samples drops from 12.2 GB → 3.6 GB, which is what makes it fit
  under the 42.8 GB cgroup host alongside MCTS trees + compile cache.
  The float cast happens in `Trainer.train_step` after the H2D copy,
  so the uint8 bytes also cross PCIe instead of float32 — a free 4×
  bandwidth win as a side-effect.
"""

import os
import numpy as np


# ─── Symmetry helpers ───────────────────────────────────────────────

# 8 group elements: 4 rotations × {identity, horizontal flip}.
# Each element is a (k, flip) tuple where k is the np.rot90 count.
SYMMETRIES = [(k, f) for k in range(4) for f in (False, True)]


def augment_8fold(obs: np.ndarray, policy: np.ndarray, board_size: int):
    """Yield all 8 symmetries of a single (obs, policy) pair.

    Reference implementation kept for the smoke tests. Production
    sampling uses the vectorized path inside ReplayBuffer.sample().

    Args:
        obs: (17, N, N) observation planes
        policy: (N*N+1,) policy vector — last element is pass probability
        board_size: N
    """
    N = board_size
    pass_prob = policy[-1]
    board_policy = policy[:-1].reshape(N, N)

    for k, flip in SYMMETRIES:
        obs_t = np.rot90(obs, k, axes=(1, 2)).copy()
        pol_t = np.rot90(board_policy, k).copy()
        if flip:
            obs_t = np.flip(obs_t, axis=2).copy()
            pol_t = np.flip(pol_t, axis=1).copy()
        policy_t = np.append(pol_t.ravel(), pass_prob)
        yield obs_t, policy_t


def _apply_symmetry_batch(obs_batch: np.ndarray, policy_batch: np.ndarray,
                          k: int, flip: bool, N: int):
    """Apply one (k, flip) symmetry to a whole batch at once.

    Operates on the spatial axes of obs (last two) and the reshaped
    board portion of policy. The pass action sits at policy[..., -1]
    and is invariant under any board symmetry.
    """
    # obs: (B, 17, N, N) — rotate spatial axes
    obs_t = np.rot90(obs_batch, k, axes=(2, 3))
    if flip:
        obs_t = np.flip(obs_t, axis=3)
    # Force a contiguous copy so downstream torch.from_numpy doesn't
    # see strided/negative-strides views (np.flip returns a view).
    obs_t = np.ascontiguousarray(obs_t)

    # policy: (B, N*N + 1). Split off pass, transform the (B, N, N)
    # board half, then re-concatenate.
    board_pol = policy_batch[:, :N * N].reshape(-1, N, N)
    pass_pol = policy_batch[:, -1:]
    board_pol_t = np.rot90(board_pol, k, axes=(1, 2))
    if flip:
        board_pol_t = np.flip(board_pol_t, axis=2)
    board_pol_t = np.ascontiguousarray(board_pol_t)
    pol_t = np.concatenate(
        [board_pol_t.reshape(-1, N * N), pass_pol], axis=1)
    return obs_t, pol_t


# ─── ReplayBuffer ───────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer storing (obs, policy, value) tuples.

    Stores raw (un-augmented) positions. Augmentation is applied
    randomly per sample at training time via `sample()`.
    """

    def __init__(self, capacity: int, board_size: int, input_planes: int = 17):
        self.capacity = capacity
        self.board_size = board_size
        N = board_size
        actions = N * N + 1

        # obs is uint8 — every cell is 0/1 so float32 would waste 4×
        # RAM (and 4× PCIe bandwidth per training step). Trainer casts
        # to float32 after H2D copy.
        self.obs = np.zeros((capacity, input_planes, N, N), dtype=np.uint8)
        self.policy = np.zeros((capacity, actions), dtype=np.float32)
        self.value = np.zeros(capacity, dtype=np.float32)

        self.size = 0
        self.index = 0

    def push(self, obs: np.ndarray, policy: np.ndarray, value: float):
        """Add one raw position to the buffer.

        Augmentation is no longer done at push time — it happens at
        sample time so the buffer holds 8× more distinct positions.
        """
        idx = self.index % self.capacity
        # Incoming obs is float32 from the C++ engine / Python self-play
        # path. Values are strictly 0.0/1.0, so the uint8 cast is lossless.
        self.obs[idx] = obs.astype(np.uint8, copy=False)
        self.policy[idx] = policy
        self.value[idx] = value
        self.index += 1
        self.size = min(self.size + 1, self.capacity)

    def _store(self, obs: np.ndarray, policy: np.ndarray, value: float):
        """Direct insert without symmetry. Used by tests + load_from."""
        idx = self.index % self.capacity
        self.obs[idx] = obs.astype(np.uint8, copy=False)
        self.policy[idx] = policy
        self.value[idx] = value
        self.index += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch, applying a random symmetry per sample.

        Each sampled position is transformed by one of the 8 dihedral
        symmetries, chosen uniformly at random per sample. The batch
        is processed in 8 buckets (one per symmetry) so each rotation
        runs once on a contiguous sub-batch — much faster than a
        per-sample Python loop.
        """
        N = self.board_size
        # Sample positions uniformly with replacement (replacement is
        # fine here: 256-batch from a 300K buffer, collision odds are
        # ~1/1000 per pair and don't bias the gradient).
        indices = np.random.randint(0, self.size, size=batch_size)
        sym_choices = np.random.randint(0, 8, size=batch_size)

        raw_obs = self.obs[indices]        # (B, 17, N, N)
        raw_pol = self.policy[indices]     # (B, N*N+1)
        out_value = self.value[indices].copy()  # values are scalar, sym-invariant

        out_obs = np.empty_like(raw_obs)
        out_pol = np.empty_like(raw_pol)

        # Bucket by symmetry id and process each bucket as one rotation.
        for sym_id, (k, flip) in enumerate(SYMMETRIES):
            mask = sym_choices == sym_id
            if not mask.any():
                continue
            obs_t, pol_t = _apply_symmetry_batch(
                raw_obs[mask], raw_pol[mask], k, flip, N)
            out_obs[mask] = obs_t
            out_pol[mask] = pol_t

        return out_obs, out_pol, out_value

    def save_to(self, path: str) -> None:
        """Persist current live samples to a .npz file.

        Only the `size` live samples are written so reloads into a
        different-capacity buffer still work. Written atomically via a
        temp file + rename so a crash mid-save never leaves a partial
        file that `load_from` would silently accept.
        """
        # np.savez always appends .npz if the stem has none — pick a temp
        # path that already ends in .npz so the written file is exactly
        # `tmp_path`, then atomically rename to `path`.
        tmp_path = path + ".tmp.npz"
        with open(tmp_path, "wb") as f:
            np.savez(
                f,
                obs=self.obs[:self.size],
                policy=self.policy[:self.size],
                value=self.value[:self.size],
                index=np.int64(self.index),
                size=np.int64(self.size),
            )
        os.replace(tmp_path, path)

    def load_from(self, path: str) -> None:
        """Restore samples from a file produced by `save_to`.

        Truncates the saved payload to this buffer's capacity if the
        saved buffer was larger. `index` is advanced to `size` so new
        pushes continue to wrap circularly. Handles legacy float32
        obs saves by casting to uint8 on load (values are always 0/1).
        """
        data = np.load(path)
        n = int(data["size"])
        n = min(n, self.capacity)
        saved_obs = data["obs"][:n]
        if saved_obs.dtype != np.uint8:
            saved_obs = saved_obs.astype(np.uint8, copy=False)
        self.obs[:n] = saved_obs
        self.policy[:n] = data["policy"][:n]
        self.value[:n] = data["value"][:n]
        self.size = n
        self.index = n

    def __len__(self) -> int:
        return self.size
