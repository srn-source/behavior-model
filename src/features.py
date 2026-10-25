"""Person-level features for Setting A, built only from persona sources.

  demographics  14 persona columns (QID11-QID24), one-hot encoded (E1)
  persona_raw   (E2b) every usable persona answer in wave1_3_response.csv (D2), see RawPersonFeatures
  persona_raw:<domain>+<domain>  (E2c) the same raw answers restricted to some persona blocks,
                e.g. persona_raw:demographics+personality (item_registry domain = D1 BlockName)

Everything fitted (category levels, medians, clipping bounds, scaling) is fitted on the
rows passed to `fit`, which callers restrict to training participants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .item_registry import build_registry



def demographic_columns() -> list[str]:
    reg = build_registry()
    return reg.loc[(reg.role == "persona") & (reg.domain == "demographics"), "column"].tolist()


class PersonFeatures:
    """E1: one-hot demographics. fit() on training people, transform() anyone."""

    def __init__(self, kind: str = "demographics"):
        if kind != "demographics":
            raise ValueError(kind)
        self.kind = kind

    def _raw(self, pids) -> pd.DataFrame:
        return data.load_responses("wave1_3").loc[list(pids), demographic_columns()]

    def fit(self, pids) -> "PersonFeatures":
        demo = self._raw(pids)
        self.levels = {c: sorted(demo[c].dropna().unique()) for c in demo.columns}
        return self

    def transform(self, pids) -> pd.DataFrame:
        demo = self._raw(pids)
        parts = []
        for c, levels in self.levels.items():
            # one-hot; the first level is the reference, unseen levels map to all zeros
            for lv in levels[1:]:
                parts.append((demo[c] == lv).astype(float).rename(f"{c}={lv:g}"))
        X = pd.concat(parts, axis=1)
        X.index.name = "pid"
        return X


class RawPersonFeatures:
    """E2b: every usable persona answer from D2 (wave1_3_response.csv), no precomputed scores.

    Columns come only from item_registry.setting_a_inputs (role == "persona"). Encoding by type:
      single-choice MC (SAVR/SAHR)  one-hot (option numbers are labels, not amounts);
                                    levels seen < MIN_LEVEL times in training are left in the reference
      multi-choice MC (MAVR/MAHR)   0/1, blank = not selected = 0
      Matrix / Slider               number, standardised (not clipped: the scale is already bounded)
      numeric text entry            number, clipped to the training 1st-99th percentile, standardised,
                                    blank = training median plus a 0/1 "was blank" column
      free text (TE, non-numeric)   dropped (Forward Flow words, essays, listed thoughts)
    Columns constant in training are dropped.
    domains: keep only persona columns of these item_registry domains (None = all, i.e. E2b).
    """

    MIN_LEVEL = 5

    def __init__(self, domains: list[str] | None = None):
        self.domains = tuple(domains) if domains else None
        self.kind = "persona_raw" + (":" + "+".join(self.domains) if self.domains else "")

    def _persona(self, pids) -> pd.DataFrame:
        from .item_registry import setting_a_inputs
        w = setting_a_inputs(data.load_responses("wave1_3"), self.reg)
        if self.domains:
            dom = self.reg.set_index("column").domain
            w = w[[c for c in w.columns if dom[c] in self.domains]]
        return w.loc[list(pids)]

    def fit(self, pids) -> "RawPersonFeatures":
        self.reg = build_registry()
        p = self.reg[self.reg.role == "persona"].set_index("column")
        w = self._persona(pids)
        self.onehot, self.multi, self.numeric = {}, [], []
        for col in w.columns:
            qtype, sel = p.at[col, "qtype"], p.at[col, "selector"]
            if qtype == "MC" and sel in ("SAVR", "SAHR"):
                counts = w[col].value_counts()
                levels = sorted(counts.index)
                self.onehot[col] = [lv for lv in levels[1:] if counts[lv] >= self.MIN_LEVEL]
            elif qtype == "MC" and sel in ("MAVR", "MAHR"):
                self.multi.append(col)
            else:
                x = pd.to_numeric(w[col], errors="coerce")
                if x.notna().sum() >= 0.9 * max(w[col].notna().sum(), 1):  # numeric, not free text
                    self.numeric.append(col)
        x = w[self.numeric].apply(pd.to_numeric, errors="coerce")
        # Clip only typed-in numbers (outliers such as 99999). Rating scales (Matrix/Slider) are bounded
        # already; clipping them would erase rare extreme answers (it flattened 49 columns before).
        free = [c for c in self.numeric if p.at[c, "qtype"] == "TE"]
        self.lo = x.quantile(0.01).where(x.columns.isin(free), -np.inf)
        self.hi = x.quantile(0.99).where(x.columns.isin(free), np.inf)
        self.median = x.median()
        self.flag_missing = [c for c in self.numeric if x[c].isna().mean() > 0.01]
        clipped = x.clip(self.lo, self.hi, axis=1).fillna(self.median)
        self.mean, self.std = clipped.mean(), clipped.std().replace(0, 1)
        self.keep = None
        self.keep = list(self._build(w).loc[:, lambda d: d.std() > 0].columns)
        return self

    def _build(self, w: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for col, levels in self.onehot.items():
            for lv in levels:
                parts.append((w[col] == lv).astype(float).rename(f"{col}={lv:g}"))
        parts.append(w[self.multi].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).astype(float))
        x = w[self.numeric].apply(pd.to_numeric, errors="coerce")
        parts.append(x[self.flag_missing].isna().astype(float).add_suffix("__blank"))
        parts.append((x.clip(self.lo, self.hi, axis=1).fillna(self.median) - self.mean) / self.std)
        X = pd.concat(parts, axis=1)
        X.index.name = "pid"
        return X if self.keep is None else X[self.keep]

    def transform(self, pids) -> pd.DataFrame:
        return self._build(self._persona(pids))


def make_features(kind: str):
    """'demographics' (E1), 'persona_raw' (E2-All) or 'persona_raw:<domain>+<domain>' (E2-P/C/E: raw answers of
    some persona blocks only)."""
    if kind == "persona_raw":
        return RawPersonFeatures()
    if kind.startswith("persona_raw:"):
        return RawPersonFeatures(domains=kind.split(":", 1)[1].split("+"))
    return PersonFeatures(kind)
