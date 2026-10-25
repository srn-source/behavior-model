"""Reconstruct (pid, slot, product, price) for the 40-item pricing study.

The CSV columns QID9_1..QID9_40 are *presentation slots*. The product and the
price a person saw in a slot exist only in that person's question text, so we
parse them from the wave_split JSON for each wave.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from . import data

_PAT = re.compile(
    r"product category: (?P<category>.+?)\. Suppose you are in a grocery store, and you see the "
    r"following product in that category: (?P<product>.+?)\. The product is priced at: "
    r"\$(?P<price>[0-9][0-9,]*(?:\.[0-9]+)?)\. Would you"
)

WAVE_FIELDS = {
    "wave1_3": "wave4_Q_wave1_3_A",
    "wave4": "wave4_Q_wave4_A",
}


def parse_pricing_question(text: str) -> dict:
    m = _PAT.search(text)
    if m is None:
        raise ValueError(f"unparseable pricing question: {text[:200]!r}")
    return dict(
        category=m["category"],
        product=m["product"],
        price=float(m["price"].replace(",", "")),
    )


def pricing_observations(wave: str) -> pd.DataFrame:
    """One row per (pid, slot) for the given wave, with the answer (1 = buy, 2 = not buy)."""
    ws = data.load_wave_split()
    field = WAVE_FIELDS[wave]
    rows = []
    for pid, blob in ws[field].items():
        for _, q in data.iter_questions(blob):
            qid = q["QuestionID"]
            if not qid.startswith("QID9_"):
                continue
            ans = (q.get("Answers") or {}).get("SelectedByPosition")
            rows.append(dict(pid=pid, slot=qid, wave=wave, answer=ans, **parse_pricing_question(q["QuestionText"])))
    df = pd.DataFrame(rows)
    products = sorted(df["product"].unique())
    df["product_id"] = df["product"].map({p: i for i, p in enumerate(products)})
    return df


def pricing_panel() -> pd.DataFrame:
    """Both waves joined on (pid, slot); one row per person-slot. Returns a fresh copy."""
    return _pricing_panel().copy()


@lru_cache(maxsize=1)
def _pricing_panel() -> pd.DataFrame:
    a = pricing_observations("wave1_3")
    b = pricing_observations("wave4")
    keys = ["pid", "slot"]
    m = a.merge(b, on=keys, suffixes=("_w13", "_w4"), how="outer", validate="one_to_one")
    return m
