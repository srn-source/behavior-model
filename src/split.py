"""Participant-disjoint 70/15/15 split, stratified on age x sex, frozen to data/splits.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data

SPLIT_PATH = data.REPO_ROOT / "data" / "splits.json"
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 0
STRATA_COLUMNS = ["QID13", "QID12"]  # age bracket, sex at birth
MIN_STRATUM = 20


def make_split(seed: int = SEED) -> dict:
    w13 = data.load_responses("wave1_3")
    strata = w13[STRATA_COLUMNS].astype(str).agg("|".join, axis=1)
    sizes = strata.value_counts()
    small = sizes[sizes < MIN_STRATUM]
    if len(small):
        print(f"warning: sparse strata {small.to_dict()}")
    rng = np.random.default_rng(seed)
    out = {k: [] for k in FRACTIONS}
    for _, pids in strata.groupby(strata):
        ids = rng.permutation(pids.index.to_numpy())
        n_train = int(round(len(ids) * FRACTIONS["train"]))
        n_val = int(round(len(ids) * FRACTIONS["val"]))
        out["train"] += ids[:n_train].tolist()
        out["val"] += ids[n_train:n_train + n_val].tolist()
        out["test"] += ids[n_train + n_val:].tolist()
    out = {k: sorted(int(i) for i in v) for k, v in out.items()}
    return dict(seed=seed, strata=STRATA_COLUMNS, dataset_revision=data.DATASET_REVISION, **out)


def load_split() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)


def make_folds(k: int = 5, seed: int = SEED) -> pd.Series:
    """Assign every participant to one of k folds, stratified on age x sex.

    Used for cross-validated baselines that must be scored on all 2,058 people
    (to compare with the paper's full-sample numbers) without in-sample fitting.
    """
    w13 = data.load_responses("wave1_3")
    strata = w13[STRATA_COLUMNS].astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(seed)
    fold = pd.Series(-1, index=w13.index, name="fold")
    offset = 0
    for _, pids in strata.groupby(strata):
        ids = rng.permutation(pids.index.to_numpy())
        fold.loc[ids] = (np.arange(len(ids)) + offset) % k
        offset += len(ids)  # continue the round-robin so fold sizes stay balanced
    return fold
