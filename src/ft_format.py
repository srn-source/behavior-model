"""E5 / E6 (RQ4, bonus): turn one person into a chat example for a small language model.

Input, Setting A only:
  persona    the person's waves 1-3 answers from wave_split `wave1_3_persona_json` (D4), rendered as
             compact "Q: ... / A: ..." text without option lists. Never `full_persona` (D5): it holds the
             wave-4 answers to the repeated questions (the dataset's leakage trap).
  questions  the wave-4 questions this person was shown, from wave_split `wave4_Q_wave4_A` (D4) with the
             answers removed. One answer line per data column: Matrix rows, Slider statements and the 40
             pricing products are separate lines, named by their wave4_response.csv column (e.g. QID287_1).
             The 40 pricing questions share one instruction and option list, then one short line per product
             (category, product, price parsed by src/pricing.py), which saves ~3k tokens per person.
Target: the person's wave-4 answers from wave4_response.csv (D3), one line per column, "<column>: <answer>".
Choice questions are answered with the option number (= the code stored in D3), other questions with a number.

Prompt modes (build_example `prompt`), for the prompting vs retrieval comparison; only "persona" is used for training:
  persona       persona + questions (E5 zero-shot, E6 fine-tuned)                                      A1
  none          questions only, no information about the person                                        A0
  prior_median  questions + per item the training people's typical answer (E0 in the prompt)           A2
  prior_dist    questions + per item the share of training people choosing each option                  A2
  knn           questions + per item the answers of the k most similar training people (E3's neighbours)  B1
The per-item lines come from src/llm_context.py and use training people's answers only.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from . import data
from .pricing import parse_pricing_question

_FORMAT = ("Reply with one line per item, in the order given, in the form '<item id>: <answer>'. "
           "For multiple-choice items answer with the option number only; for other items answer with a number only.")
SYSTEM = ("You simulate one specific survey respondent. You are given this person's answers to an earlier survey. "
          "Answer the new questions exactly as this person would. " + _FORMAT)
SYSTEMS = {
    "persona": SYSTEM,
    "none": "You simulate a survey respondent from the adult US population. Answer the questions as a typical "
            "respondent would. " + _FORMAT,
    "prior_median": "You simulate a survey respondent. Under each item you are told how other respondents answered it. "
                    "Answer the questions as this respondent would. " + _FORMAT,
    "prior_dist": "You simulate a survey respondent. Under each item you are told how other respondents answered it. "
                  "Answer the questions as this respondent would. " + _FORMAT,
    "knn": "You simulate one specific survey respondent. Under each item you are told how the respondents most "
           "similar to this person answered it. Answer the questions as this person would. " + _FORMAT,
}
PROMPTS = tuple(SYSTEMS)


@dataclass(frozen=True)
class Item:
    column: str                  # wave4_response.csv column, e.g. QID287_1
    kind: str                    # "choice" (answer = option number) or "number"
    n_options: int = 0           # choice items: options are 1..n_options


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# --------------------------------------------------------------------------------------------- persona

def _persona_answer(q: dict) -> list[str]:
    a = q.get("Answers") or {}
    qtype, sel = q.get("QuestionType"), (q.get("Settings") or {}).get("Selector")
    if qtype == "Matrix":
        rows, picked = q.get("Rows") or [], a.get("SelectedText") or []
        return [f"- {_clean(r)}: {_clean(v)}" for r, v in zip(rows, picked)]
    if qtype == "Slider":
        return [f"- {_clean(s) or 'value'}: {_clean(v)}" for s, v in zip(q.get("Statements") or [""], a.get("Values") or [])]
    if qtype == "MC" and sel in ("MAVR", "MAHR"):
        picked = a.get("SelectedText") or []
        return ["A: " + ("; ".join(_clean(v) for v in picked) if picked else "(none selected)")]
    if qtype == "MC":
        return [f"A: {_clean(a.get('SelectedText'))}"]
    text = a.get("Text")
    if isinstance(text, list):  # TE FORM: list of {field: value}
        text = "; ".join(f"{k}: {v}" for d in text for k, v in d.items())
    return [f"A: {_clean(text)}"]


def render_persona(persona_json: str) -> str:
    """Compact text of the waves 1-3 persona: block headings, each question and the person's answer."""
    lines = []
    for block in json.loads(persona_json):
        questions = [q for q in block.get("Questions", []) if q.get("QuestionType") != "DB"]
        if not questions:
            continue
        lines.append(f"## {_clean(block.get('BlockName'))}")
        for q in questions:
            lines.append(f"Q: {_clean(q.get('QuestionText'))}")
            lines.extend(_persona_answer(q))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- wave-4 items

@lru_cache(maxsize=1)
def target_columns_by_qid() -> dict[str, list[str]]:
    """Wave-4 QuestionID -> its wave4_response.csv columns, from the question catalog (D1)."""
    out = {}
    for e in data.load_catalog():
        if e.get("source") == "wave1_3_persona_json" or e.get("QuestionType") == "DB":
            continue
        cols = e.get("csv_columns") or []
        if isinstance(cols, str):
            cols = json.loads(cols.replace("'", '"'))
        out[e["QuestionID"]] = list(cols)
    return out


def render_questions(wave4_json: str, hint=None) -> tuple[str, list[Item]]:
    """Wave-4 questions shown to the person (answers dropped) and the answer items in order.
    `hint(column, kind, pricing=None) -> str | None` adds one context line under each item (prompting modes)."""
    def add_hint(column: str, kind: str, pricing: dict | None = None):
        h = hint(column, kind, pricing) if hint is not None else None
        if h:
            lines.append(f"    ({h})")

    colmap = target_columns_by_qid()
    lines, items = [], []
    pricing_header = False
    for block in json.loads(wave4_json):
        for q in block.get("Questions", []):
            qtype = q.get("QuestionType")
            if qtype == "DB" or q["QuestionID"] not in colmap:
                continue
            if q["QuestionID"].startswith("QID9_"):  # pricing: shared header, one line per product
                if not pricing_header:
                    options = "; ".join(f"{i} = {_clean(o)}" for i, o in enumerate(q.get("Options") or [], 1))
                    lines.append("[Pricing] Suppose you are in a grocery store and see each product below at the "
                                 "price shown. Would you or would you not purchase it?")
                    lines.append(f"Options: {options}")
                    pricing_header = True
                p = parse_pricing_question(q.get("QuestionText", ""))
                lines.append(f"  {q['QuestionID']}: {p['product']} ({p['category']}) at ${p['price']:.2f}")
                items.append(Item(q["QuestionID"], "choice", len(q.get("Options") or [])))
                add_hint(q["QuestionID"], "choice", p)
                continue
            cols = colmap[q["QuestionID"]]
            text = _clean(q.get("QuestionText"))
            if qtype in ("MC", "Matrix"):
                options = q.get("Options") if qtype == "MC" else q.get("Columns")
                options = [_clean(o) for o in options or []]
                lines.append(f"[{q['QuestionID']}] {text}")
                lines.append("Options: " + "; ".join(f"{i} = {o}" for i, o in enumerate(options, 1)))
                if qtype == "MC":
                    lines.append(f"  {cols[0]}")
                    items.append(Item(cols[0], "choice", len(options)))
                    add_hint(cols[0], "choice")
                else:
                    for col, row in zip(cols, q.get("Rows") or []):
                        lines.append(f"  {col}: {_clean(row)}")
                        items.append(Item(col, "choice", len(options)))
                        add_hint(col, "choice")
            elif qtype == "Slider":
                rng = q.get("Range") or {}
                lines.append(f"[{q['QuestionID']}] {text} (answer with a number from {rng.get('Min', 0)} to {rng.get('Max', 100)})")
                for col, st in zip(cols, q.get("Statements") or [""]):
                    lines.append(f"  {col}" + (f": {_clean(st)}" if _clean(st) else ""))
                    items.append(Item(col, "number"))
                    add_hint(col, "number")
            else:  # TE with a numeric answer
                lines.append(f"[{q['QuestionID']}] {text} (answer with a number)")
                lines.append(f"  {cols[0]}")
                items.append(Item(cols[0], "number"))
                add_hint(cols[0], "number")
    return "\n".join(lines), items


# --------------------------------------------------------------------------------------------- examples

def format_answer(item: Item, value) -> str | None:
    """D3 value -> answer text ('4', '30', '2.5'); None when the person left it blank."""
    v = pd.to_numeric(value, errors="coerce")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(int(v)) if float(v).is_integer() else f"{float(v):g}"


def build_example(pid: int, wave4_answers: pd.Series | None = None, persona_pid: int | None = None,
                  prompt: str = "persona", hint=None) -> dict:
    """Chat messages for one person. With `wave4_answers` (a D3 row) the assistant turn is the target;
    items the person left blank are dropped from both the question list and the answer.
    `persona_pid` (ablation only): use that person's persona instead, keeping `pid`'s own wave-4 questions.
    `prompt` / `hint`: prompting modes (module docstring); `hint` comes from src/llm_context.py."""
    if prompt not in SYSTEMS:
        raise ValueError(prompt)
    if (hint is not None) != (prompt in ("prior_median", "prior_dist", "knn")):
        raise ValueError(f"prompt {prompt!r} {'needs' if hint is None else 'takes no'} per-item context")
    ws = data.load_wave_split()
    questions, items = render_questions(ws.at[pid, "wave4_Q_wave4_A"], hint)
    if wave4_answers is not None:
        answers = {it.column: format_answer(it, wave4_answers.get(it.column)) for it in items}
        items = [it for it in items if answers[it.column] is not None]
    tail = "\n\nItems to answer, in order: " + ", ".join(it.column for it in items)
    if prompt == "persona":
        persona = render_persona(ws.at[pid if persona_pid is None else persona_pid, "wave1_3_persona_json"])
        user = ("Earlier survey answers of this person:\n" + persona +
                "\n\nNew questions. Answer each item as this person would:\n" + questions + tail)
    else:
        user = "Questions. Answer each item:\n" + questions + tail
    messages = [{"role": "system", "content": SYSTEMS[prompt]}, {"role": "user", "content": user}]
    if wave4_answers is not None:
        messages.append({"role": "assistant",
                         "content": "\n".join(f"{it.column}: {answers[it.column]}" for it in items)})
    return {"pid": int(pid), "messages": messages, "items": [it.__dict__ for it in items]}
