"""Circular replay buffer with 8-fold symmetry augmentation for Go."""

import numpy as np


def augment_8fold(obs: np.ndarray, policy: np.ndarray, board_size: int):
    """Apply 8-fold symmetry (4 rotations x 2 reflections) to a single sample.

    Args:
        obs: (17, N, N) observation planes
        policy: (N*N+1,) policy vector — last element is pass probability
        board_size: N

    Yields:
        (obs_aug, policy_aug) for each of the 8 symmetries
    """
    N = board_size
    pass_prob = policy[-1]
    board_policy = policy[:-1].reshape(N, N)

    for k in range(4):  # 0, 90, 180, 270 degrees
        for flip in (False, True):
            obs_t = np.rot90(obs, k, axes=(1, 2)).copy()
            pol_t = np.rot90(board_policy, k).copy()
            if flip:
                obs_t = np.flip(obs_t, axis=2).copy()
                pol_t = np.flip(pol_t, axis=1).copy()
            policy_t = np.append(pol_t.ravel(), pass_prob)
            yield obs_t, policy_t


class ReplayBuffer:
    """Fixed-size circular buffer storing (obs, policy, value) tuples.

    Stores pre-allocated numpy arrays for zero-copy sampling.
    """

    def __init__(self, capacity: int, board_size: int, input_planes: int = 17):
        self.capacity = capacity
        self.board_size = board_size
        N = board_size
        actions = N * N + 1

        self.obs = np.zeros((capacity, input_planes, N, N), dtype=np.float32)
        self.policy = np.zeros((capacity, actions), dtype=np.float32)
        self.value = np.zeros(capacity, dtype=np.float32)

        self.size = 0
        self.index = 0

    def push(self, obs: np.ndarray, policy: np.ndarray, value: float,
             augment: bool = True):
        """Add a sample (with optional 8-fold augmentation)."""
        if augment:
            for obs_t, pol_t in augment_8fold(obs, policy, self.board_size):
                self._store(obs_t, pol_t, value)
        else:
            self._store(obs, policy, value)

    def _store(self, obs: np.ndarray, policy: np.ndarray, value: float):
        idx = self.index % self.capacity
        self.obs[idx] = obs
        self.policy[idx] = policy
        self.value[idx] = value
        self.index += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch. Returns (obs, policy, value) numpy arrays."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.obs[indices], self.policy[indices], self.value[indices]

    def __len__(self) -> int:
        return self.size
