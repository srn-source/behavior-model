"""Per-item context for the LLM prompting experiments (RQ5, prompting vs retrieval), fitted on training people only.

Both classes turn other people's wave-4 answers into one short line per item, which src.ft_format puts under the
item in the prompt. Neither reads the answers of the person being predicted, nor of any validation or test person.

  PopulationPrior (A2)   what training people answered each item: the lower median (E0's point prediction), or the
                         share choosing each option; for pricing, the share who bought that product at that price
                         (E0's cell). "E0 in the prompt".
  NeighbourContext (B1)  what the k training people most similar to this person answered (E3's neighbours: cosine on
                         the standardized 806 persona features). For an ordinary item, the first k neighbours who
                         answered that column (so the same between-subject condition, as in E3). For a pricing item,
                         each of the first k neighbours saw the same product once, at their own price: "bought at $x"
                         or "not at $x". "Retrieval of similar people".

Training examples (C2: fine-tuning with this context) need the same conditions as prediction, where the person's own
answer is never part of the context: hint_for(pid) leaves a training person out of the population statistics
(leave-one-out), and a training person is never their own neighbour. For validation and test people nothing changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from . import metrics as M
from .baselines import PRICING_COLUMNS
from .knn import KNNModel
from .pricing import pricing_panel

NON_PRICING = [c for c in M.SCORED_COLUMNS if c not in PRICING_COLUMNS]


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _train_answers(train_pids) -> pd.DataFrame:
    return data.load_responses("wave4").loc[list(train_pids), NON_PRICING].apply(pd.to_numeric, errors="coerce")


class PopulationPrior:
    def __init__(self, train_pids, style: str = "median"):
        if style not in ("median", "dist"):
            raise ValueError(style)
        self.style = style
        self.Y = _train_answers(train_pids)
        self.values = {c: np.sort(self.Y[c].dropna().to_numpy()) for c in NON_PRICING if self.Y[c].notna().any()}
        p = pricing_panel()
        tr = p[p.pid.isin(set(self.Y.index))]
        buy = (tr.answer_w4 == 1).astype(float)
        g = buy.groupby([tr.product_w4, tr.price_w4])
        self.buy_sum, self.buy_n = g.sum().to_dict(), g.size().to_dict()
        # training pid -> {product: (price seen, bought)}, for leave-one-out
        self.own_pricing: dict[int, dict] = {}
        for r, b in zip(tr.itertuples(index=False), buy):
            self.own_pricing.setdefault(int(r.pid), {})[r.product_w4] = (r.price_w4, b)

    def _text(self, column: str, kind: str, vals: np.ndarray) -> str | None:
        if not len(vals):
            return None
        if self.style == "dist" and kind == "choice":
            u, n = np.unique(vals, return_counts=True)
            return "other respondents: " + ", ".join(f"{_num(k)} = {c / len(vals):.0%}" for k, c in zip(u, n))
        return f"typical answer of other respondents: {_num(float(np.percentile(vals, 50, method='lower')))}"

    def _pricing_text(self, pricing: dict, own: tuple | None) -> str | None:
        key = (pricing["product"], pricing["price"])
        s, n = self.buy_sum.get(key), self.buy_n.get(key)
        if n is None:
            return None
        if own is not None and own[0] == pricing["price"]:  # leave this training person out of their own cell
            s, n = s - own[1], n - 1
        return None if n == 0 else f"{s / n:.0%} of other respondents bought it at this price"

    def hint(self, column: str, kind: str, pricing: dict | None = None) -> str | None:
        """Context for a validation or test person (all training people)."""
        if pricing is not None:
            return self._pricing_text(pricing, None)
        return self._text(column, kind, self.values[column]) if column in self.values else None

    def hint_for(self, pid: int):
        """Context for `pid`; a training person is left out of every statistic (leave-one-out)."""
        pid = int(pid)
        if pid not in self.Y.index:
            return self.hint
        own_answers, own_pricing = self.Y.loc[pid], self.own_pricing.get(pid, {})

        def hint(column: str, kind: str, pricing: dict | None = None) -> str | None:
            if pricing is not None:
                return self._pricing_text(pricing, own_pricing.get(pricing["product"]))
            if column not in self.values:
                return None
            vals, a = self.values[column], own_answers.get(column)
            if a is not None and not np.isnan(a):
                i = np.searchsorted(vals, a)
                vals = np.delete(vals, i)
            return self._text(column, kind, vals)

        return hint


class NeighbourContext:
    def __init__(self, train_pids, k: int):
        self.k = k
        self.knn = KNNModel("wave4", k=k).fit(train_pids)
        self.Y = _train_answers(self.knn.train)
        p = pricing_panel()
        tr = p[p.pid.isin(set(self.knn.train))]
        # (train pid, product) -> (price shown, bought?)
        self.seen = {(r.pid, r.product_w4): (r.price_w4, r.answer_w4 == 1) for r in tr.itertuples(index=False)}
        self.order: dict[int, np.ndarray] = {}

    def prepare(self, pids) -> "NeighbourContext":
        """Rank all training people by similarity to each pid (done once for a batch)."""
        pids = [int(x) for x in pids]
        order, _ = self.knn.neighbour_order(pids)
        for i, pid in enumerate(pids):
            ranked = self.knn.train[order[i]]
            self.order[pid] = ranked[ranked != pid]  # a training person is never their own neighbour
        return self

    def neighbours(self, pid: int) -> np.ndarray:
        if pid not in self.order:
            self.prepare([pid])
        return self.order[pid]

    def hint_for(self, pid: int):
        ranked = self.neighbours(int(pid))

        def hint(column: str, kind: str, pricing: dict | None = None) -> str | None:
            if pricing is not None:
                parts = []
                for nb in ranked[: self.k]:
                    seen = self.seen.get((nb, pricing["product"]))
                    if seen is not None:
                        parts.append(("bought at" if seen[1] else "not at") + f" ${seen[0]:.2f}")
                return "similar respondents: " + "; ".join(parts) if parts else None
            if column not in self.Y.columns:
                return None
            vals = self.Y.loc[ranked, column].dropna().head(self.k)
            return "similar respondents answered: " + ", ".join(_num(v) for v in vals) if len(vals) else None

        return hint
