"""Shared utility helpers for the nn package."""

import numpy as np


def set_seed(seed: int) -> None:
    """Seed numpy's global random state for reproducible runs.

    Args:
        seed (int): seed value passed to numpy's global RNG.
    """
    np.random.seed(seed)
