"""One row per catalog CSV column: its role in the prediction task and how it is scored.

Role is the leakage barrier. Setting A models may read only columns with
role == "persona". Every column with role == "target" is a wave-4 item, and the
wave1_3_response.csv copy of it is the person's *earlier answer to the same
question*: allowed only in Setting B.
"""
from __future__ import annotations

import re

import pandas as pd

from . import data
from .metrics import SPEC_BY_COL

PERSONA_SOURCE = "wave1_3_persona_json"
TARGET_SOURCE = "wave4_Q_wave1_3_A"

# Paper categories (question_catalog_README) for persona blocks.
PERSONA_DOMAIN = {
    "Demographics": "demographics",
    "Personality": "personality",
    "Cognitive tests": "cognitive",
    "Forward Flow": "cognitive",
    "Economic preferences": "economic_preferences",
}

# Between-subject experiment blocks: "<experiment> - <condition>" or similar.
_CONDITION_PATTERNS = [
    (r"^Anchoring - (African countries|redwood) (high|low)$", lambda m: (f"Anchoring - {m[1]}", m[2])),
    (r"^Base-rate (30|70) engineers$", lambda m: ("Base-rate", f"{m[1]} engineers")),
    (r"^Disease ?- ?(gain|loss)$", lambda m: ("Disease", m[1])),
    (r"^Linda ?-(no conjunction|conjunction)$", lambda m: ("Linda", m[1])),
    (r"^Outcome bias - (success|failure)$", lambda m: ("Outcome bias", m[1])),
    (r"^Less is More Gamble ([ABC])$", lambda m: ("Less is More", m[1])),
    (r"^Proportion dominance ([12])([ABC])$", lambda m: (f"Proportion dominance {m[1]}", m[2])),
    (r"^Sunk cost - (yes|no)$", lambda m: ("Sunk cost", m[1])),
    (r"^Absolute vs\. relative - (calculator|jacket)$", lambda m: ("Absolute vs. relative", m[1])),
    (r"^WTA/WTP Thaler(?: problem)? - (.+)$", lambda m: ("WTA/WTP Thaler", m[1])),
    (r"^Allais Form ([12])$", lambda m: ("Allais", f"form {m[1]}")),
    (r"^Myside (Ford|German)$", lambda m: ("Myside", m[1])),
    (r"^Probability matching vs\. maximizing - Problem ([12])$", lambda m: ("Probability matching", f"problem {m[1]}")),
]


def split_condition(block: str) -> tuple[str, str | None]:
    """'Anchoring - redwood high' -> ('Anchoring - redwood', 'high'); within-subject blocks -> (block, None)."""
    for pat, fn in _CONDITION_PATTERNS:
        m = re.match(pat, block)
        if m:
            return fn(m)
    return block, None


def build_registry() -> pd.DataFrame:
    rows = []
    for entry in data.load_catalog():
        block = entry.get("BlockName", "").strip()
        settings = entry.get("Settings", {}) or {}
        source = entry.get("source")
        if source == PERSONA_SOURCE:
            role = "persona"
            domain = PERSONA_DOMAIN.get(block, "other")
        elif source == TARGET_SOURCE:
            role = "target"
            domain = "pricing" if block.startswith("Product Preferences") else "heuristics_biases"
        else:
            raise ValueError(f"unexpected source {source!r} for {entry['QuestionID']}")
        experiment, condition = split_condition(block) if role == "target" else (block, None)
        n_options = len(entry.get("Options", []) or entry.get("Columns", []) or [])
        for col in entry.get("csv_columns", []):
            spec = SPEC_BY_COL.get(col)
            rows.append(
                dict(
                    column=col,
                    qid=entry["QuestionID"],
                    block=block,
                    role=role,
                    domain=domain,
                    experiment=experiment,
                    condition=condition,
                    qtype=entry.get("QuestionType"),
                    selector=settings.get("Selector"),
                    n_options=n_options,
                    scored=spec is not None,
                    task=spec.task if spec else None,
                    score_min=spec.lo if spec else None,
                    score_max=spec.hi if spec else None,
                    question_text=entry.get("QuestionText", ""),
                )
            )
    reg = pd.DataFrame(rows)
    if reg.column.duplicated().any():
        dups = reg.column[reg.column.duplicated()].tolist()
        raise ValueError(f"duplicate catalog columns: {dups}")
    return reg


def persona_columns(reg: pd.DataFrame | None = None) -> list[str]:
    reg = build_registry() if reg is None else reg
    return reg.loc[reg.role == "persona", "column"].tolist()


def target_columns(reg: pd.DataFrame | None = None, scored_only: bool = True) -> list[str]:
    reg = build_registry() if reg is None else reg
    m = reg.role == "target"
    if scored_only:
        m &= reg.scored
    return reg.loc[m, "column"].tolist()


def setting_a_inputs(w13: pd.DataFrame, reg: pd.DataFrame | None = None) -> pd.DataFrame:
    """The only sanctioned way to get Setting A tabular inputs from wave1_3_response.csv."""
    cols = persona_columns(reg)
    out = w13[cols]
    assert not set(out.columns) & set(target_columns(reg, scored_only=False))
    return out
