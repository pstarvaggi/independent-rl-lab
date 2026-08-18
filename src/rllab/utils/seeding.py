"""Deterministic random-number plumbing.

NumPy ``Generator`` instances are passed explicitly throughout the project.  This
module is the one boundary that optionally seeds process-global libraries.
"""

from __future__ import annotations

import importlib
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """A root seed and independently spawned child seeds."""

    root: int
    environment: int
    agent: int
    evaluation: int


def spawn_seeds(seed: int) -> SeedBundle:
    """Derive stable, statistically independent seeds from ``seed``."""

    children = np.random.SeedSequence(seed).spawn(3)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return SeedBundle(seed, *values)


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed Python, NumPy's legacy global state, and PyTorch when installed.

    The returned ``Generator`` should be preferred over NumPy's global functions.
    ``PYTHONHASHSEED`` only affects child processes created after this call.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch: Any = importlib.import_module("torch")
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)

    return np.random.default_rng(seed)
