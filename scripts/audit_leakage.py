"""Leakage audit for Setting A (persona-only prediction of wave-4 answers).

Checks, each written to results/metrics/leakage_audit.json:
  1. Registry: persona and target columns are disjoint (by column and by QuestionID).
  2. wave_split JSON: no wave-4 QuestionID or question text appears in any person's wave1_3 persona.
  3. Near-duplicate question text between persona and target items (for manual review).
  4. The trap: by design, full_persona holds the *wave-4* answers on repeated items
     (persona + ground truth), so it must never be a model input.
  5. persona_summary provenance: does it mention target-task content?
  6. wave1_3_response.csv contains earlier answers to every target column (Setting B data).
  7. Between-subject condition assignment is identical in both waves.
  8. Each person's old and new wave-4 questions are the same questions (JSON level).

Writes leakage_audit.json and a readable leakage_audit.md (scripts/leakage_audit_md.py).

Run: python -m scripts.audit_leakage
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src import data
from src.item_registry import build_registry

OUT = data.REPO_ROOT / "results" / "metrics"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def _tokens(text: str) -> set[str]:
    return set(_norm(text).split())


def check_registry(reg: pd.DataFrame) -> dict:
    p, t = reg[reg.role == "persona"], reg[reg.role == "target"]
    col_overlap = sorted(set(p.column) & set(t.column))
    qid_overlap = sorted(set(p.qid) & set(t.qid))
    return dict(
        persona_columns=len(p),
        target_columns=len(t),
        target_scored_columns=int(t.scored.sum()),
        # One QuestionID can expand to many columns (matrix rows, sliders, multi-select options).
        # Persona reuses 10 QIDs (QID268-279) across two blocks, hence fewer QIDs than (QID, block) pairs.
        persona_qids=int(p.qid.nunique()),
        persona_qid_block_pairs=int(len(p[["qid", "block"]].drop_duplicates())),
        target_qids=int(t.qid.nunique()),
        target_scored_qids=int(t[t.scored].qid.nunique()),
        target_unscored_qids=sorted(t[~t.scored].qid.unique()),
        column_overlap=col_overlap,
        qid_overlap=qid_overlap,
        passed=not col_overlap and not qid_overlap,
    )


def check_wave_split_json() -> dict:
    ws = data.load_wave_split()
    qid_hits, text_hits = 0, 0
    examples = []
    for pid, row in ws.iterrows():
        persona_q = list(data.iter_questions(row["wave1_3_persona_json"]))
        persona_qids = {q["QuestionID"] for _, q in persona_q}
        persona_texts = {_norm(q.get("QuestionText", "")) for _, q in persona_q}
        for block, q in data.iter_questions(row["wave4_Q_wave4_A"]):
            if q.get("is_descriptive"):
                continue
            if q["QuestionID"] in persona_qids:
                qid_hits += 1
                if len(examples) < 5:
                    examples.append(dict(pid=int(pid), qid=q["QuestionID"], block=block))
            t = _norm(q.get("QuestionText", ""))
            if t and t in persona_texts:
                text_hits += 1
    return dict(participants=len(ws), qid_overlaps=qid_hits, exact_text_overlaps=text_hits,
                examples=examples, passed=qid_hits == 0 and text_hits == 0)


def near_duplicate_texts(reg: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Token-Jaccard between persona and target question texts (incl. matrix rows)."""
    def texts(df):
        seen = {}
        for _, r in df.iterrows():
            t = r.question_text or ""
            if len(_tokens(t)) >= 4:
                seen.setdefault((r.qid, _norm(t)), (r.qid, r.block, t))
        return list(seen.values())

    cat = {(e["QuestionID"], e.get("BlockName", "").strip()): e for e in data.load_catalog()}
    persona = texts(reg[reg.role == "persona"])
    # Include matrix row labels: personality items are short statements.
    for (qid, block), e in cat.items():
        if e.get("source") == "wave1_3_persona_json":
            for row in e.get("Rows", []) or []:
                persona.append((qid, block, row))
    target = texts(reg[reg.role == "target"])
    out = []
    for tq, tb, tt in target:
        ttok = _tokens(tt)
        for pq, pb, pt in persona:
            ptok = _tokens(pt)
            if not ptok:
                continue
            j = len(ttok & ptok) / len(ttok | ptok)
            if j >= threshold:
                out.append(dict(target_qid=tq, target_block=tb, persona_qid=pq, persona_block=pb,
                                jaccard=round(j, 3), target_text=tt[:160], persona_text=pt[:160]))
    return sorted(out, key=lambda d: -d["jaccard"])


def check_full_persona_trap(reg: pd.DataFrame) -> dict:
    """Compare full_persona answers on target items with wave-1-3 and wave-4 answers.

    Unit = one question a person saw: (pid, block, QuestionID). A matrix question such as
    QID287 (10 rows) is one item, and counts as "differ" if any row changed. Sources:
      A_old  = wave_split.wave4_Q_wave1_3_A   (earlier answer)
      A_new  = wave_split.wave4_Q_wave4_A     (wave-4 answer, ground truth)
      A_full = full_persona.persona_json
    """
    fp = data.load_full_persona()
    ws = data.load_wave_split()

    def answers(blob):
        out = {}
        for block, q in data.iter_questions(blob):
            a = q.get("Answers")
            if a is not None and not q.get("is_descriptive"):
                out[(block.strip(), q["QuestionID"])] = json.dumps(a, sort_keys=True)
        return out

    n_items = n_eq_w4 = n_eq_w13 = n_differ = n_differ_eq_w4 = 0
    target_blocks = set(reg.loc[reg.role == "target", "block"])
    for pid in ws.index:
        full = answers(fp.at[pid, "persona_json"])
        w4 = answers(ws.at[pid, "wave4_Q_wave4_A"])
        w13 = answers(ws.at[pid, "wave4_Q_wave1_3_A"])
        for key, a4 in w4.items():
            if key[0] not in target_blocks or key not in full or key not in w13:
                continue
            n_items += 1
            f = full[key]
            n_eq_w4 += f == a4
            n_eq_w13 += f == w13[key]
            if a4 != w13[key]:
                n_differ += 1
                n_differ_eq_w4 += f == a4
    return dict(
        unit="question (pid x block x QuestionID); matrix rows compared together",
        participants=len(ws),
        items_per_participant=round(n_items / len(ws), 2),
        compared_items=n_items,
        full_persona_equals_wave4=round(n_eq_w4 / n_items, 4),
        full_persona_equals_wave1_3=round(n_eq_w13 / n_items, 4),
        items_where_waves_differ=n_differ,
        of_those_full_persona_equals_wave4=round(n_differ_eq_w4 / max(n_differ, 1), 4),
        contains_wave4_answers=n_differ_eq_w4 / max(n_differ, 1) > 0.99,
    )


def check_persona_summary() -> dict:
    """Keyword scan of persona_summary for target-task content."""
    fp = data.load_full_persona()
    patterns = {
        "pricing/purchase": r"\b(?:purchas\w*|pric\w*|willingness to pay)\b",
        "anchoring": r"\banchor\w*",
        "framing/disease": r"\b(?:framing|disease)\b",
        "sunk cost": r"\bsunk cost\b",
        "false consensus": r"\bfalse consensus\b",
        "wave 4": r"\bwave ?4\b",
    }
    s = fp["persona_summary"].fillna("")
    return {k: int(s.str.contains(p, case=False, regex=True).sum()) for k, p in patterns.items()} | {
        "participants": len(s)
    }


def check_setting_b_columns(reg: pd.DataFrame) -> dict:
    w13 = data.load_responses("wave1_3")
    tcols = reg.loc[reg.role == "target", "column"]
    present = [c for c in tcols if c in w13.columns]
    return dict(
        wave1_3_response_columns=w13.shape[1],
        target_columns_present_in_wave1_3_response=len(present),
        note="Selecting 'all columns' of wave1_3_response.csv silently gives Setting B.",
    )


def check_conditions(reg: pd.DataFrame) -> dict:
    """Each person sees one condition per between-subject experiment; is it the same in both waves?"""
    w13, w4 = data.load_responses("wave1_3"), data.load_responses("wave4")
    t = reg[(reg.role == "target") & reg.condition.notna()]
    res = {}
    for exp, g in t.groupby("experiment"):
        def assigned(df):
            seen = pd.DataFrame({c: df[g.loc[g.condition == c, "column"]].notna().any(axis=1)
                                 for c in g.condition.unique()})
            n = seen.sum(axis=1)
            return seen.idxmax(axis=1).where(n == 1), int((n > 1).sum()), int((n == 0).sum())
        a13, multi13, none13 = assigned(w13)
        a4, multi4, none4 = assigned(w4)
        both = a13.notna() & a4.notna()
        res[exp] = dict(
            same_condition=round(float((a13[both] == a4[both]).mean()), 4),
            n_both=int(both.sum()),
            multi_condition_w13=multi13, multi_condition_w4=multi4,
            no_condition_w13=none13, no_condition_w4=none4,
            counts=a4.value_counts().to_dict(),
        )
    return res


def check_old_new_same_questions() -> dict:
    """For each person, wave4_Q_wave1_3_A and wave4_Q_wave4_A must hold the same questions.

    Compares the (block, QuestionID) sets and the question text, which for pricing includes
    the product and the price shown. Checked straight from the wave_split JSON, independent
    of the CSV-based condition check.
    """
    ws = data.load_wave_split()

    def items(blob):
        return {(b.strip(), q["QuestionID"]): q.get("QuestionText", "")
                for b, q in data.iter_questions(blob) if not q.get("is_descriptive")}

    same_set = same_text = 0
    sizes: dict[str, int] = {}
    combos = set()
    mismatches = []
    for pid, row in ws.iterrows():
        old, new = items(row["wave4_Q_wave1_3_A"]), items(row["wave4_Q_wave4_A"])
        sizes[f"{len(old)}/{len(new)}"] = sizes.get(f"{len(old)}/{len(new)}", 0) + 1
        if old.keys() == new.keys():
            same_set += 1
            if all(old[k] == new[k] for k in old):
                same_text += 1
        elif len(mismatches) < 5:
            mismatches.append(int(pid))
        combos.add(frozenset(b for b, _ in new))
    return dict(
        participants=len(ws),
        questions_old_slash_new=sizes,
        same_question_set=same_set,
        same_question_text=same_text,
        mismatched_pids=mismatches,
        distinct_condition_combinations=len(combos),
        passed=same_set == same_text == len(ws),
    )


def main():
    from scripts.leakage_audit_md import render

    OUT.mkdir(parents=True, exist_ok=True)
    reg = build_registry()
    report = dict(
        dataset_revision=data.DATASET_REVISION,
        registry=check_registry(reg),
        wave_split_json=check_wave_split_json(),
        full_persona_trap=check_full_persona_trap(reg),
        persona_summary_keywords=check_persona_summary(),
        setting_b=check_setting_b_columns(reg),
        conditions=check_conditions(reg),
        old_new_same_questions=check_old_new_same_questions(),
    )
    near = near_duplicate_texts(reg)
    report["near_duplicate_texts"] = dict(threshold=0.5, n_pairs=len(near), top=near[:15])
    with open(OUT / "leakage_audit.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    (OUT / "leakage_audit.md").write_text(render(json.loads((OUT / "leakage_audit.json").read_text())))
    print(json.dumps({k: v for k, v in report.items() if k != "near_duplicate_texts"}, indent=1, default=str)[:6000])
    print(f"near-duplicate pairs (jaccard>=0.5): {len(near)}")
    for d in near[:15]:
        print(f"  {d['jaccard']:.2f} {d['target_qid']} [{d['target_block']}] ~ {d['persona_qid']} [{d['persona_block']}]")
        print(f"       T: {d['target_text'][:110]}")
        print(f"       P: {d['persona_text'][:110]}")


if __name__ == "__main__":
    main()
