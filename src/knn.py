"""E3: k nearest neighbours on the raw persona ("people like you answered ...").

Similarity: cosine between people's E2b feature vectors (D2 persona, 806 features),
each feature standardised on the training split. Neighbours are training people only.

  * ordinary scored columns: the k most similar training people who answered that column
    (so the same between-subject condition); predict their lower median, like E0 does for
    everyone. Anchoring answers are raw numbers; the lower median commutes with the decile
    transform, so this equals the median decile.
  * pricing: neighbours saw other products at other prices, so their answers are not copied.
    For the product person i is shown, take each neighbour's residual on *that product*
    (bought = 1 minus E0's buy rate for the cell they saw) and add the neighbours' mean residual
    to E0's buy rate for i's (product, price). "People like you buy beer more/less than usual."

With k = all training people both parts reduce exactly to E0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from . import metrics as M
from .baselines import PRICING_COLUMNS
from .features import make_features
from .pricing import pricing_panel


def _answers(target: str) -> pd.DataFrame:
    return data.load_responses("wave4" if target == "wave4" else "wave1_3")


def _lower_median(x: np.ndarray) -> float:
    return float(np.percentile(x, 50, method="lower")) if len(x) else np.nan


class KNNModel:
    def __init__(self, target: str, k: int, kind: str = "persona_raw"):
        self.target, self.k, self.kind = target, k, kind

    # ----------------------------------------------------------------- fit
    def fit(self, train_pids) -> "KNNModel":
        self.train = np.array(list(train_pids))
        self.features = make_features(self.kind).fit(self.train)
        X = self.features.transform(self.train).to_numpy()
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.Xtr = self._unit(X)
        self.Y = _answers(self.target).loc[self.train, [c for c in M.SCORED_COLUMNS if c not in PRICING_COLUMNS]]
        self.Y = self.Y.apply(pd.to_numeric, errors="coerce")
        self._fit_pricing()
        return self

    def _unit(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(n == 0, 1, n)

    def _fit_pricing(self):
        p = pricing_panel()
        answer = "answer_w4" if self.target == "wave4" else "answer_w13"
        tr = p[p.pid.isin(self.train)].copy()
        tr["buy"] = (tr[answer] == 1).astype(float)
        rate = tr.groupby(["product_w4", "price_w4"]).buy.mean()
        self.cell_rate = rate.to_dict()
        tr["residual"] = tr.buy - tr.set_index(["product_w4", "price_w4"]).index.map(rate).to_numpy()
        # train people x products: how much more (or less) than usual this person buys that product
        self.R = tr.pivot(index="pid", columns="product_w4", values="residual").reindex(self.train)

    # ----------------------------------------------------------------- neighbours
    def neighbour_order(self, pids) -> tuple[np.ndarray, np.ndarray]:
        """For each pid: training indices sorted from most to least similar, and the similarities."""
        Xq = self._unit(self.features.transform(list(pids)).to_numpy())
        S = Xq @ self.Xtr.T
        order = np.argsort(-S, axis=1, kind="stable")
        return order, np.take_along_axis(S, order, axis=1)

    # ----------------------------------------------------------------- predict
    def predict(self, pids) -> pd.DataFrame:
        pids = list(pids)
        order, _ = self.neighbour_order(pids)
        pred = pd.DataFrame(index=pd.Index(pids, name="pid"), columns=M.SCORED_COLUMNS, dtype=float)
        Yv = self.Y.to_numpy()
        for j, col in enumerate(self.Y.columns):
            answered = ~np.isnan(Yv[:, j])
            vals = np.empty(len(pids))
            for i in range(len(pids)):
                nb = order[i][answered[order[i]]][: self.k]  # k most similar people who answered it
                vals[i] = _lower_median(Yv[nb, j])
            pred[col] = vals
        self._predict_pricing(pids, order, pred)
        return pred

    def _predict_pricing(self, pids, order, pred):
        p = pricing_panel()
        sub = p[p.pid.isin(pids)]
        pos = {pid: i for i, pid in enumerate(pids)}
        Rv = self.R.to_numpy()
        prod_idx = {prod: j for j, prod in enumerate(self.R.columns)}
        out = []
        for r in sub.itertuples(index=False):
            nb = order[pos[r.pid]][: self.k]
            shift = np.nanmean(Rv[nb, prod_idx[r.product_w4]])
            p_buy = self.cell_rate.get((r.product_w4, r.price_w4), 0.5) + shift
            out.append((r.pid, r.slot, 1.0 if p_buy >= 0.5 else 2.0))
        wide = pd.DataFrame(out, columns=["pid", "slot", "pred"]).pivot(index="pid", columns="slot", values="pred")
        pred[PRICING_COLUMNS] = wide.reindex(index=pids, columns=PRICING_COLUMNS).to_numpy()
