"""Baselines fitted on training participants only.

E0 — population median. The same prediction for everyone in a condition:
  * non-pricing columns: lower median of the training participants' answers.
    Each between-subject condition has its own columns, so medians are per condition.
    The lower median always is a real answer option, and any median minimises
    absolute error, which is what the paper metric rewards.
  * pricing: lower median per (product, price), parsed from the question text,
    then mapped back to each person's slot. Fallback when a cell has too few
    training observations: per price multiplier, then the overall pricing median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from . import metrics as M
from .pricing import pricing_panel

PRICING_COLUMNS = [f"QID9_{i}" for i in range(1, 41)]
MIN_CELL = 20  # minimum training observations for a (product, price) median


def _lower_median(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    return float(np.percentile(x, 50, method="lower")) if len(x) else np.nan


def _answers(target: str) -> pd.DataFrame:
    return data.load_responses("wave4" if target == "wave4" else "wave1_3")


def fit_e0(train_pids, target: str) -> dict:
    """Learn E0 from training participants' `target` answers. Returns a model dict."""
    train_pids = list(train_pids)
    ans = _answers(target).loc[train_pids]
    col_median = {c: _lower_median(ans[c]) for c in M.SCORED_COLUMNS if c not in PRICING_COLUMNS}

    panel = pricing_panel()
    answer = "answer_w4" if target == "wave4" else "answer_w13"
    panel["multiplier"] = (panel.price_w4 / panel.groupby("product_w4").price_w4.transform("median")).round(1)
    tr = panel[panel.pid.isin(train_pids)]
    cell = tr.groupby(["product_w4", "price_w4"])[answer].agg([_lower_median, "size"])
    by_mult = tr.groupby("multiplier")[answer].agg(_lower_median)
    overall = _lower_median(tr[answer])
    return dict(
        target=target,
        n_train=len(train_pids),
        col_median=col_median,
        cell_median=cell["_lower_median"].to_dict(),
        cell_size=cell["size"].to_dict(),
        mult_median=by_mult.to_dict(),
        overall_pricing=overall,
    )


def predict_e0(model: dict, pids) -> tuple[pd.DataFrame, dict]:
    """Predict for `pids`. Uses only their pricing assignment (product, price shown), never answers."""
    pids = list(pids)
    pred = pd.DataFrame(index=pd.Index(pids, name="pid"), columns=M.SCORED_COLUMNS, dtype=float)
    for c, v in model["col_median"].items():
        pred[c] = v

    panel = pricing_panel()
    panel["multiplier"] = (panel.price_w4 / panel.groupby("product_w4").price_w4.transform("median")).round(1)
    sub = panel[panel.pid.isin(pids)][["pid", "slot", "product_w4", "price_w4", "multiplier"]]
    sources = {"product_price": 0, "multiplier": 0, "overall": 0}
    values = []
    for r in sub.itertuples(index=False):
        key = (r.product_w4, r.price_w4)
        if model["cell_size"].get(key, 0) >= MIN_CELL:
            values.append(model["cell_median"][key]); sources["product_price"] += 1
        elif r.multiplier in model["mult_median"]:
            values.append(model["mult_median"][r.multiplier]); sources["multiplier"] += 1
        else:
            values.append(model["overall_pricing"]); sources["overall"] += 1
    sub = sub.assign(pred=values)
    wide = sub.pivot(index="pid", columns="slot", values="pred")
    pred[PRICING_COLUMNS] = wide.reindex(index=pids, columns=PRICING_COLUMNS).to_numpy()
    return pred, dict(pricing_prediction_source=sources)
