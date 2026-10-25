"""The paper's accuracy metric, re-implemented in catalog-column space.

Reference: evaluation/mad_accuracy_evaluation.py in
github.com/tianyipeng-lab/Digital-Twin-Simulation (commit f1eed51).

Per scored cell:   acc = 1 - |pred - truth| / (max - min)
Headline number:   for each respondent, average cells within each task, average
                   tasks, then average respondents ("task level" summary).
Eligibility:       a cell counts only if both human answers (wave 1-3 and wave 4)
                   exist, so retest and every model are scored on identical cells.

Quirks of the reference scorer that we reproduce on purpose (and flag in the report):
  * Ranges are hand-set per column, not derived from the options.
  * Anchoring free-text estimates are mapped to deciles 1-10 using thresholds
    pooled over a pair of columns. The paper fits thresholds on all participants;
    we let callers pass training-only reference values instead.
  * Sunk-cost answers are free numbers scored with a fixed 0-20 range, so a cell
    can score below zero.
  * Proportion-dominance items (QID177-179) are grouped under "less is more".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    column: str
    task: str
    lo: float
    hi: float
    decile_group: str | None = None

    @property
    def range(self) -> float:
        return self.hi - self.lo


def _spec() -> list[ColumnSpec]:
    s: list[ColumnSpec] = []

    def add(cols, task, lo, hi, group=None):
        s.extend(ColumnSpec(c, task, lo, hi, group) for c in cols)

    fc_ids = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    add([f"QID287_{i}" for i in fc_ids], "false consensus", 1, 5)
    add([f"QID290_{i}" for i in fc_ids], "false consensus", 0, 100)
    add(["QID156", "QID154"], "base rate", 0, 100)
    add(["QID157", "QID158"], "framing problem", 1, 6)
    add([f"QID{q}_{i}" for q in (159, 160) for i in (1, 2, 3)], "conjunction problem (Linda)", 1, 6)
    add(["QID161", "QID162"], "outcome bias", 1, 7)
    add(["QID164_TEXT", "QID166_TEXT"], "anchoring and adjustment", 1, 10, "african")
    add(["QID168_TEXT", "QID170_TEXT"], "anchoring and adjustment", 1, 10, "redwood")
    add([f"QID{q}" for q in range(171, 177)], "less is more", 1, 5)
    add([f"QID{q}" for q in range(177, 180)], "less is more", 1, 6)
    add(["QID181_TEXT", "QID182_TEXT"], "sunk cost fallacy", 0, 20)
    add(["QID183", "QID184"], "absolute vs. relative savings", 1, 2)
    add(["QID189", "QID190", "QID191"], "WTA/WTP-Thaler", 1, 10)
    add(["QID192", "QID193"], "Allais", 1, 2)
    add(["QID194", "QID195"], "myside", 1, 6)
    add([f"QID198_{i}" for i in range(1, 11)], "prob matching vs. max", 1, 2)
    add([f"QID203_{i}" for i in range(1, 7)], "prob matching vs. max", 1, 2)
    add([f"QID288_{i}" for i in (1, 2, 3, 4)], "non-separability of risks and benefits", 1, 7)
    add([f"QID289_{i}" for i in (1, 2, 3, 4)], "non-separability of risks and benefits", 1, 7)
    add(["QID291"], "omission", 1, 4)
    add(["QID196"], "denominator neglect", 1, 2)
    add([f"QID9_{i}" for i in range(1, 41)], "pricing", 1, 2)
    return s


SPEC: list[ColumnSpec] = _spec()
SPEC_BY_COL: dict[str, ColumnSpec] = {c.column: c for c in SPEC}
SCORED_COLUMNS: list[str] = [c.column for c in SPEC]
TASKS: list[str] = list(dict.fromkeys(c.task for c in SPEC))
DECILE_GROUPS: dict[str, list[str]] = {
    g: [c.column for c in SPEC if c.decile_group == g] for g in ("african", "redwood")
}


def decile_thresholds(reference: pd.DataFrame) -> dict[str, np.ndarray]:
    """10th..90th percentiles of the pooled reference values for each anchoring pair."""
    out = {}
    for group, cols in DECILE_GROUPS.items():
        pooled = pd.concat([pd.to_numeric(reference[c], errors="coerce") for c in cols]).dropna()
        out[group] = np.percentile(pooled, np.arange(10, 100, 10))
    return out


def to_scoring_space(df: pd.DataFrame, thresholds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Restrict to scored columns, coerce to numbers and decile-code the anchoring answers."""
    out = df.reindex(columns=SCORED_COLUMNS).apply(pd.to_numeric, errors="coerce")
    for group, cols in DECILE_GROUPS.items():
        t = thresholds[group]
        for c in cols:
            v = out[c].to_numpy(dtype=float)
            dec = np.searchsorted(t, v, side="left") + 1.0  # value <= t[i] -> decile i+1
            out[c] = np.where(np.isnan(v), np.nan, dec)
    return out


def cell_accuracy(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """1 - |pred - truth| / range per cell; NaN where either side is missing."""
    ranges = pd.Series({c.column: c.range for c in SPEC})
    pred = pred.reindex(index=truth.index, columns=SCORED_COLUMNS)
    truth = truth.reindex(columns=SCORED_COLUMNS)
    return 1 - (pred - truth).abs() / ranges


def respondent_scores(acc: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.Series:
    """Paper headline per respondent: mean over tasks of the mean cell accuracy in each task."""
    if mask is not None:
        acc = acc.where(mask.reindex_like(acc).fillna(False).astype(bool))
    task_of = pd.Series({c.column: c.task for c in SPEC})
    by_task = acc.T.groupby(task_of).mean().T  # respondents x tasks, NaN if no cells
    return by_task.mean(axis=1, skipna=True).dropna()


def task_scores(acc: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.Series:
    """Mean cell accuracy within each task, pooling respondents (paper's task-level table)."""
    if mask is not None:
        acc = acc.where(mask.reindex_like(acc).fillna(False).astype(bool))
    task_of = pd.Series({c.column: c.task for c in SPEC})
    long = acc.stack().rename("acc").reset_index(level=1).rename(columns={"level_1": "column"})
    long["task"] = long["column"].map(task_of)
    return long.groupby("task")["acc"].mean().reindex(TASKS)


def column_scores(acc: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.Series:
    if mask is not None:
        acc = acc.where(mask.reindex_like(acc).fillna(False).astype(bool))
    return acc.mean(axis=0)


def eligibility_mask(w13: pd.DataFrame, w4: pd.DataFrame) -> pd.DataFrame:
    """A cell is eligible when both human waves answered it."""
    w4 = w4.reindex(index=w13.index, columns=w13.columns)
    return w13.notna() & w4.notna()


def bootstrap_mean(scores: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling respondents."""
    x = scores.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_delta(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) over shared respondents with a 95% percentile CI."""
    common = a.index.intersection(b.index)
    return bootstrap_mean(a.loc[common] - b.loc[common], n_boot=n_boot, seed=seed)


def bootstrap_ratio_of_means(num: pd.Series, den: pd.Series, n_boot: int = 2000, seed: int = 0):
    """mean(num) / mean(den) over shared respondents, with a 95% percentile CI that resamples
    respondents jointly (so the uncertainty of the denominator is included)."""
    common = num.index.intersection(den.index)
    x, y = num.loc[common].to_numpy(dtype=float), den.loc[common].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1) / y[idx].mean(axis=1)
    return float(x.mean() / y.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
