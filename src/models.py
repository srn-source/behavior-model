"""E1 / E2: regularised regression from person features to each target column.

One model per scored column (fit on training participants who answered it, so per condition):
  * discrete columns (<= 11 answer levels, incl. anchoring deciles): multinomial logistic
    regression (L2, strength C). The prediction is the *median of the predicted distribution*,
    which minimises expected absolute error, i.e. what the paper metric rewards.
  * continuous columns (0-100 sliders, base rate, sunk cost): linear median (quantile) regression
    with L1 strength alpha.
  * pricing: one logistic model on all (person, product, price) rows of the training people:
    E0's buy rate per (product, price) cell as a fixed starting point (logit offset, never shrunk),
    plus person features and person features x price, which are regularised. Predict "buy" when P(buy) >= 0.5.
Every model uses every person feature (no dimensionality reduction).

With very strong regularisation every model collapses to its intercept, i.e. to E0.
Anchoring answers are modelled as deciles (train thresholds) and mapped back to a raw value
inside the predicted decile, so they score exactly as the predicted decile.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, QuantileRegressor

from . import data
from . import metrics as M
from .baselines import PRICING_COLUMNS
from .eval import thresholds
from .features import make_features
from .pricing import pricing_panel

MAX_DISCRETE_LEVELS = 11


def _answers(target: str) -> pd.DataFrame:
    return data.load_responses("wave4" if target == "wave4" else "wave1_3")


def _logistic_with_offset(D: np.ndarray, y: np.ndarray, offset: np.ndarray, C: float) -> np.ndarray:
    """Minimise C * sum(log-loss(y, offset + D w)) + ||w||^2 / 2 (sklearn's convention), no intercept."""
    def f(w):
        z = offset + D @ w
        p = 1.0 / (1.0 + np.exp(-z))
        loss = np.sum(np.logaddexp(0.0, z) - y * z)
        return C * loss + 0.5 * w @ w, C * (D.T @ (p - y)) + w
    res = minimize(f, np.zeros(D.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return res.x


def _ordered_median(classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per row, the smallest class whose cumulative probability reaches 0.5."""
    order = np.argsort(classes)
    cum = np.cumsum(proba[:, order], axis=1)
    return classes[order][np.argmax(cum >= 0.5, axis=1)]


class ColumnModels:
    """All per-column models for one target and one feature set."""

    def __init__(self, kind: str, target: str, C: float = 0.1, alpha: float = 0.01):
        self.kind, self.target, self.C, self.alpha = kind, target, C, alpha

    # ----------------------------------------------------------------- fit
    def fit(self, train_pids, reuse: "ColumnModels | None" = None) -> "ColumnModels":
        """Fit every column model. `reuse`: an already fitted model of the same kind, target and
        training people; its parts that do not depend on the changed setting are copied instead of
        refitted (logistic and pricing depend only on C, median regression only on alpha), which gives
        identical results and saves time during the search."""
        train_pids = list(train_pids)
        if reuse is not None and (reuse.kind, reuse.target, reuse.train_pids) != (self.kind, self.target, train_pids):
            raise ValueError("reuse must be fitted on the same kind, target and training people")
        same_C = reuse is not None and reuse.C == self.C
        same_alpha = reuse is not None and reuse.alpha == self.alpha
        self.train_pids = train_pids
        self.features = reuse.features if reuse is not None else make_features(self.kind).fit(train_pids)
        X = self.features.transform(train_pids)
        Y = M.to_scoring_space(_answers(self.target).loc[train_pids], thresholds())
        raw = _answers(self.target).loc[train_pids]
        self.models, self.decile_value = {}, {}
        for spec in M.SPEC:
            col = spec.column
            if col in PRICING_COLUMNS:
                continue
            y = Y[col].dropna()
            if spec.decile_group:  # remember a raw value inside each decile, to save predictions in raw space
                r = pd.to_numeric(raw.loc[y.index, col], errors="coerce")
                self.decile_value[col] = r.groupby(y).median().to_dict()
            levels = np.sort(y.unique())
            Xc = X.loc[y.index].to_numpy()
            if len(levels) == 1:
                self.models[col] = ("constant", levels[0])
            elif len(levels) <= MAX_DISCRETE_LEVELS:
                if same_C:
                    self.models[col] = reuse.models[col]
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    m = LogisticRegression(C=self.C, max_iter=2000).fit(Xc, y.to_numpy())
                self.models[col] = ("discrete", m)
            else:
                if same_alpha:
                    self.models[col] = reuse.models[col]
                    continue
                m = QuantileRegressor(quantile=0.5, alpha=self.alpha, solver="highs").fit(Xc, y.to_numpy())
                self.models[col] = ("continuous", (m, float(y.min()), float(y.max())))
        if same_C:
            self.cell_logit, self.pricing_w = reuse.cell_logit, reuse.pricing_w
        else:
            self._fit_pricing(train_pids, X)
        return self

    def _pricing_parts(self, panel: pd.DataFrame, X: pd.DataFrame):
        """Offset (E0's logit for the row's product x price cell) and person design matrix."""
        keys = list(zip(panel.product_w4, panel.price_w4))
        offset = np.array([self.cell_logit.get(k, 0.0) for k in keys])  # unseen cell: no prior
        Z = X.loc[panel.pid].to_numpy()
        mult = panel.multiplier.to_numpy()[:, None] - 1.0  # 0 at the reference price
        return offset, np.hstack([Z, Z * mult])

    def _fit_pricing(self, train_pids, X):
        """Two steps: (1) E0's buy rate per (product, price) cell, as a fixed logit offset;
        (2) an L2-regularised logistic model of how person features shift it (C = self.C).
        The offset is never shrunk, so the strongest regularisation gives exactly E0."""
        panel = pricing_panel()
        panel["multiplier"] = (panel.price_w4 / panel.groupby("product_w4").price_w4.transform("median")).round(1)
        tr = panel[panel.pid.isin(train_pids)]
        answer = "answer_w4" if self.target == "wave4" else "answer_w13"
        y = (tr[answer] == 1).astype(float).to_numpy()
        rate = pd.Series(y, index=pd.MultiIndex.from_arrays([tr.product_w4, tr.price_w4])).groupby(level=[0, 1]).mean()
        rate = rate.clip(0.01, 0.99)
        self.cell_logit = np.log(rate / (1 - rate)).to_dict()
        offset, D = self._pricing_parts(tr, X)
        self.pricing_w = _logistic_with_offset(D, y, offset, self.C)

    # ----------------------------------------------------------------- predict
    def predict(self, pids) -> pd.DataFrame:
        pids = list(pids)
        X = self.features.transform(pids)
        Xn = X.to_numpy()
        pred = pd.DataFrame(index=pd.Index(pids, name="pid"), columns=M.SCORED_COLUMNS, dtype=float)
        for col, (kind, m) in self.models.items():
            if kind == "constant":
                v = np.full(len(pids), m, dtype=float)
            elif kind == "discrete":
                v = _ordered_median(m.classes_.astype(float), m.predict_proba(Xn))
            else:
                reg, lo, hi = m
                v = np.clip(reg.predict(Xn), lo, hi)
            if col in self.decile_value:  # decile -> a raw value inside that decile
                lookup = self.decile_value[col]
                v = np.array([lookup.get(d, np.nan) for d in v])
            pred[col] = v
        panel = pricing_panel()
        panel["multiplier"] = (panel.price_w4 / panel.groupby("product_w4").price_w4.transform("median")).round(1)
        sub = panel[panel.pid.isin(pids)]
        offset, D = self._pricing_parts(sub, X)
        p_buy = 1.0 / (1.0 + np.exp(-(offset + D @ self.pricing_w)))
        sub = sub.assign(pred=np.where(p_buy >= 0.5, 1.0, 2.0))
        wide = sub.pivot(index="pid", columns="slot", values="pred")
        pred[PRICING_COLUMNS] = wide.reindex(index=pids, columns=PRICING_COLUMNS).to_numpy()
        return pred
