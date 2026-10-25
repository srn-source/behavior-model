"""Persona-shuffle ablation: does a model use the person's own persona?

Each person is given another person's persona (from the same split) while everything else stays theirs:
the wave-4 items and experimental condition they saw, and the products and prices they were shown.
If a model's predictions and score barely change, it is not using person-specific information.

The same donor map is used for the tabular models (scripts/run_persona_shuffle.py) and the
fine-tuned LLM (scripts/ft_predict.py --persona-shuffle-seed), so all models are shuffled identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def donor_map(pids, seed: int = 0) -> dict[int, int]:
    """pid -> the pid whose persona it receives: a random derangement (nobody keeps their own persona).

    Shuffle the people, then give each the persona of the next one in the shuffled order (a single cycle),
    so no one is their own donor and every persona is used exactly once."""
    pids = [int(p) for p in pids]
    order = np.random.default_rng(seed).permutation(pids)
    return {int(order[i]): int(order[(i + 1) % len(order)]) for i in range(len(order))}


class DonorFeatures:
    """Wraps a fitted feature transformer: rows for `pids` are the donors' features, indexed by `pids`."""

    def __init__(self, base, donors: dict[int, int]):
        self.base, self.donors = base, donors

    def transform(self, pids) -> pd.DataFrame:
        pids = list(pids)
        X = self.base.transform([self.donors[int(p)] for p in pids])
        X.index = pd.Index(pids, name=X.index.name)
        return X
