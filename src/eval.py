"""One scorer for every experiment.

Every experiment saves predictions the same way: a wide table, one row per pid,
one column per scored catalog column, values in the *raw answer space* of
wave4_response.csv (anchoring estimates as plain numbers, pricing 1 = buy, 2 = not).
This module turns them into paper-metric scores on a split, against a target.

Protocol (configs/experiment.yaml):
  * targets: "wave4" (primary, assignment framing) and "wave1_3" (paper framing)
  * eligible cell: both human waves answered it
  * anchoring decile thresholds: fitted on TRAINING participants' wave 1-3 answers
  * a missing prediction on an eligible cell is not scored, and coverage is reported;
    paired comparisons use only cells both methods predicted
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from . import metrics as M
from .split import load_split

PRED_DIR = data.REPO_ROOT / "results" / "predictions"
TARGETS = ("wave4", "wave1_3")


# --------------------------------------------------------------------------- predictions on disk

def save_predictions(exp_id: str, target: str, pred: pd.DataFrame, meta: dict) -> Path:
    """Write predictions (pid x scored columns, raw answer space) and their metadata."""
    if target not in TARGETS:
        raise ValueError(target)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred = pred.reindex(columns=M.SCORED_COLUMNS).apply(pd.to_numeric, errors="coerce")
    pred.index = pred.index.astype(int)
    pred.index.name = "pid"
    path = PRED_DIR / f"{exp_id}__{target}.parquet"
    pred.to_parquet(path)
    meta = dict(meta, exp_id=exp_id, target=target, n_pids=int(len(pred)),
                dataset_revision=data.DATASET_REVISION)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_predictions(exp_id: str, target: str) -> tuple[pd.DataFrame, dict]:
    path = PRED_DIR / f"{exp_id}__{target}.parquet"
    return pd.read_parquet(path), json.loads(path.with_suffix(".json").read_text())


def list_predictions() -> list[tuple[str, str]]:
    out = []
    for p in sorted(PRED_DIR.glob("*__*.parquet")):
        exp_id, target = p.stem.rsplit("__", 1)
        out.append((exp_id, target))
    return out


# --------------------------------------------------------------------------- ground truth

@lru_cache(maxsize=None)
def thresholds() -> dict[str, np.ndarray]:
    """Anchoring decile thresholds from training participants' wave 1-3 answers only."""
    w13 = data.load_responses("wave1_3")
    return M.decile_thresholds(w13.loc[load_split()["train"]])


@lru_cache(maxsize=None)
def truth_frames() -> dict[str, pd.DataFrame]:
    th = thresholds()
    return {
        "wave1_3": M.to_scoring_space(data.load_responses("wave1_3"), th),
        "wave4": M.to_scoring_space(data.load_responses("wave4"), th),
    }


def eligible(pids) -> pd.DataFrame:
    t = truth_frames()
    pids = list(pids)
    return M.eligibility_mask(t["wave1_3"].loc[pids], t["wave4"].loc[pids])


# --------------------------------------------------------------------------- scoring

@dataclass
class Score:
    respondent: pd.Series            # paper metric per person
    task: pd.Series                  # pooled cell accuracy per task
    column: pd.Series                # per scored column
    coverage: float                  # share of eligible cells that have a prediction
    people_missing: int              # people in the split with no prediction at all
    extra: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(self.respondent.mean())


def score(pred: pd.DataFrame, target: str, pids, mask: pd.DataFrame | None = None) -> Score:
    """Score raw-space predictions on `pids` against `target` with the paper metric."""
    pids = list(pids)
    truth = truth_frames()[target].loc[pids]
    elig = eligible(pids)
    P = M.to_scoring_space(pred.reindex(pids), thresholds())
    has_pred = P.notna() & elig
    m = has_pred if mask is None else (has_pred & mask.reindex_like(has_pred).fillna(False).astype(bool))
    acc = M.cell_accuracy(P, truth)
    return Score(
        respondent=M.respondent_scores(acc, m),
        task=M.task_scores(acc, m),
        column=M.column_scores(acc, m),
        coverage=float(has_pred.to_numpy().sum() / max(elig.to_numpy().sum(), 1)),
        people_missing=int((~P.notna().any(axis=1)).sum()),
    )


def paired_delta(pred_a: pd.DataFrame, pred_b: pd.DataFrame, target: str, pids, n_boot=2000, seed=0):
    """A minus B on the cells both predicted; participant bootstrap CI."""
    pids = list(pids)
    th = thresholds()
    both = M.to_scoring_space(pred_a.reindex(pids), th).notna() & M.to_scoring_space(pred_b.reindex(pids), th).notna()
    a = score(pred_a, target, pids, mask=both).respondent
    b = score(pred_b, target, pids, mask=both).respondent
    mean, lo, hi = M.paired_bootstrap_delta(a, b, n_boot=n_boot, seed=seed)
    return dict(delta=mean, ci_lo=lo, ci_hi=hi, n_people=int(len(a.index.intersection(b.index))))


def _paired_person_delta(pred_a: pd.DataFrame, pred_b: pd.DataFrame, target: str, pids) -> pd.Series:
    """Per person: score of A minus score of B, on the cells both predicted."""
    th = thresholds()
    both = M.to_scoring_space(pred_a.reindex(pids), th).notna() & M.to_scoring_space(pred_b.reindex(pids), th).notna()
    a = score(pred_a, target, pids, mask=both).respondent
    b = score(pred_b, target, pids, mask=both).respondent
    common = a.index.intersection(b.index)
    return a.loc[common] - b.loc[common]


def gap_closed(pred: pd.DataFrame, base: pd.DataFrame, ceiling: pd.DataFrame, target: str, pids,
               n_boot=2000, seed=0):
    """Share of the base -> ceiling gap that `pred` closes: mean(pred - base) / mean(ceiling - base).

    With base = E0 (population median) and ceiling = human retest, this removes the part of the
    score that predicting each item's population answer already earns: 0 = no better than E0,
    1 = as good as the person's own answer from the other wave, < 0 = worse than E0.
    Each difference is paired on the cells both methods predicted; the CI resamples people jointly."""
    pids = list(pids)
    num = _paired_person_delta(pred, base, target, pids)
    den = _paired_person_delta(ceiling, base, target, pids)
    mean, lo, hi = M.bootstrap_ratio_of_means(num, den, n_boot=n_boot, seed=seed)
    return dict(gap_closed=mean, ci_lo=lo, ci_hi=hi, n_people=int(len(num.index.intersection(den.index))))


# --------------------------------------------------------------------------- reference predictors

def retest_predictions(target: str) -> pd.DataFrame:
    """Setting B reference: predict the *other* wave's human answer (copy the earlier/later answer)."""
    other = "wave4" if target == "wave1_3" else "wave1_3"
    return data.load_responses(other)[M.SCORED_COLUMNS]
