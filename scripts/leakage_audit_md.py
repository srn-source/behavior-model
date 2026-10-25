"""Render results/metrics/leakage_audit.json as a readable Markdown report.

Every JSON key gets: what it checks, how it is computed, its values, and how to read them.
scripts.audit_leakage calls render() after each run. To rebuild only the Markdown
from an existing JSON:

    python -m scripts.leakage_audit_md
"""
from __future__ import annotations

import json

from src import data

OUT = data.REPO_ROOT / "results" / "metrics"
RERUN = "n/a (re-run `python -m scripts.audit_leakage`)"

PASS, WARN, INFO, FAIL = "✅ PASS", "⚠️ WARNING", "ℹ️ INFO", "❌ FAIL"


def _get(d: dict, key: str, default=RERUN):
    return d.get(key, default) if isinstance(d, dict) else default


def _table(rows: list[tuple]) -> str:
    out = ["| Key | Value | Meaning |", "|---|---|---|"]
    for key, value, meaning in rows:
        out.append(f"| `{key}` | {value} | {meaning} |")
    return "\n".join(out)


def _pct(x) -> str:
    return f"{x:.2%}" if isinstance(x, (int, float)) else str(x)


def _num(x) -> str:
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else str(x)


def section_registry(r: dict) -> tuple[str, str]:
    ok = _get(r, "passed", False) is True
    verdict = PASS if ok else FAIL
    body = f"""## 1. `registry`: persona vs. target, at column and QuestionID level — {verdict}

**What it checks.** The question catalog is split into input columns (`role = persona`) and columns
to predict (`role = target`). The two groups must not share a CSV column or a QuestionID.

**How it is computed.** `src/item_registry.py` labels each catalog column by the JSON it came from:
`wave1_3_persona_json` → persona, `wave4_Q_wave1_3_A` → target. `check_registry()` then intersects
the two sets twice, once on column names and once on QuestionIDs.

The two overlaps catch different problems. One QuestionID can expand to many columns
(matrix rows, sliders, multi-select options). So a question could be split across persona and target
with no column shared (only `qid_overlap` would catch that). Column names are built as
`QID + "_" + row`, so two different QIDs could also produce the same column name
(only `column_overlap` would catch that).

{_table([
    ("persona_columns", _num(_get(r, "persona_columns")), "Input columns allowed in Setting A"),
    ("persona_qids", _num(_get(r, "persona_qids")), "Distinct persona QuestionIDs"),
    ("persona_qid_block_pairs", _num(_get(r, "persona_qid_block_pairs")),
     "Persona (QID, block) pairs; higher than `persona_qids` because QID268–279 are reused in two blocks"),
    ("target_columns", _num(_get(r, "target_columns")), "Wave-4 columns, across all experimental conditions"),
    ("target_qids", _num(_get(r, "target_qids")), "Wave-4 QuestionIDs, across all conditions (each person sees 63)"),
    ("target_scored_columns", _num(_get(r, "target_scored_columns")), "Columns the paper's metric scores"),
    ("target_scored_qids", _num(_get(r, "target_scored_qids")), "QuestionIDs the paper's metric scores"),
    ("target_unscored_qids", _get(r, "target_unscored_qids"),
     "Anchoring lead questions ('more or fewer than X?'): the anchor manipulation, not the outcome"),
    ("column_overlap", _get(r, "column_overlap"), "Columns in both groups. Must be empty"),
    ("qid_overlap", _get(r, "qid_overlap"), "QuestionIDs in both groups. Must be empty"),
    ("passed", _get(r, "passed"), "True when both overlaps are empty"),
])}

**How to read it.** Both overlaps empty means the catalog cleanly separates inputs from targets.
Catalog check: persona (QID, block) pairs + target QIDs + 14 descriptive blocks = 256 catalog entries.
"""
    return verdict, body


def section_wave_split(r: dict) -> tuple[str, str]:
    ok = _get(r, "passed", False) is True
    verdict = PASS if ok else FAIL
    body = f"""## 2. `wave_split_json`: does any person's persona contain a wave-4 question? — {verdict}

**What it checks.** The same separation as Section 1, but in each person's own JSON rather than in
the catalog. The catalog being correct does not guarantee that every one of the 2,058 per-person
files is.

**How it is computed.** For every pid, `check_wave_split_json()` collects the QuestionIDs and the
normalised question text in `wave1_3_persona_json` (P) and in `wave4_Q_wave4_A` (N), and counts the
wave-4 questions found in P.

Note what is and is not compared. The **old** and **new** wave-4 fields (`wave4_Q_wave1_3_A` vs.
`wave4_Q_wave4_A`) are *supposed* to hold the same questions (Section 8). This check is about the
**persona**: the repeated questions must have been removed from it.

{_table([
    ("participants", _num(_get(r, "participants")), "People checked"),
    ("qid_overlaps", _num(_get(r, "qid_overlaps")),
     "Times a wave-4 QuestionID appeared in someone's persona (P ∩ N). Must be 0"),
    ("exact_text_overlaps", _num(_get(r, "exact_text_overlaps")),
     "Times a wave-4 question text appeared in someone's persona. Must be 0"),
    ("examples", _get(r, "examples"), "Up to 5 offending (pid, QID) pairs, if any"),
    ("passed", _get(r, "passed"), "True when both counts are 0"),
])}

**How to read it.** Zero means the dataset authors really did strip every repeated question from the
persona, for every person.
"""
    return verdict, body


def section_near_dup(r: dict) -> tuple[str, str]:
    n = _get(r, "n_pairs", None)
    verdict = PASS if n == 0 else (WARN if isinstance(n, int) else INFO)
    top = _get(r, "top", [])
    examples = ""
    if top:
        examples = "\n**Top pairs (review by eye):**\n\n| Jaccard | Target | Persona |\n|---|---|---|\n" + "\n".join(
            f"| {d['jaccard']} | {d['target_qid']}: {d['target_text'][:80]} | {d['persona_qid']}: {d['persona_text'][:80]} |"
            for d in top
        )
    body = f"""## 3. `near_duplicate_texts`: reworded copies of wave-4 questions — {verdict}

**What it checks.** Section 2 only catches identical text. A question could be lightly reworded,
e.g. "Would you buy this product?" vs. "Would you purchase this product?".

**How it is computed.** `near_duplicate_texts()` compares every target question with every persona
question and every persona matrix row, using token Jaccard similarity:
`|shared words| / |all distinct words|`. Pairs at or above the threshold are listed.
The example above scores 4/6 = 0.67.

{_table([
    ("threshold", _get(r, "threshold"), "Minimum Jaccard similarity to flag a pair"),
    ("n_pairs", _num(n), "Flagged pairs. 0 means no persona question resembles a target question"),
    ("top", f"{len(top)} listed", "Highest-similarity pairs, for manual review"),
])}
{examples}
**How to read it.** A flagged pair is not automatically leakage. It is a prompt to read both
questions. Related *constructs* (e.g. risk aversion in the persona vs. the Allais problem in wave 4)
are signal, not leakage, and text similarity does not detect them.
"""
    return verdict, body


def section_full_persona(r: dict) -> tuple[str, str]:
    contains = _get(r, "contains_wave4_answers", _get(r, "leaks_ground_truth", None))
    verdict = WARN if contains else (PASS if contains is False else INFO)
    body = f"""## 4. `full_persona_trap`: does `full_persona` contain the wave-4 answers? — {verdict}

**What it checks.** Whether the `full_persona` config can be used as model input. Wave 4 repeats
earlier questions, so each repeated question has two answers: `A_old` (waves 1–3) and `A_new`
(wave 4, the ground truth). This finds out which one `full_persona` keeps.

**How it is computed.** `check_full_persona_trap()` takes three answers for every (pid, block, QuestionID):

| Name | Source |
|---|---|
| `A_old` | `wave_split.wave4_Q_wave1_3_A` |
| `A_new` | `wave_split.wave4_Q_wave4_A` (ground truth) |
| `A_full` | `full_persona.persona_json` |

**Unit = one question one person saw**, not a CSV column. A matrix question such as QID287
(10 rows) is one item, and it counts as "differ" if any row changed.

{_table([
    ("unit", _get(r, "unit"), "What one item is"),
    ("participants", _num(_get(r, "participants")), "People checked"),
    ("items_per_participant", _get(r, "items_per_participant"),
     "Wave-4 questions per person: 40 pricing + 5 non-experimental + 2 + 2 anchoring + 1 false consensus + 13 other experiments = 63"),
    ("compared_items", _num(_get(r, "compared_items")), "Total items = participants × items per person"),
    ("full_persona_equals_wave4", _pct(_get(r, "full_persona_equals_wave4")), "Share of items where `A_full == A_new`"),
    ("full_persona_equals_wave1_3", _pct(_get(r, "full_persona_equals_wave1_3")),
     "Share where `A_full == A_old`. Since `A_full` is always `A_new`, this is simply the share of questions people answered identically twice"),
    ("items_where_waves_differ", _num(_get(r, "items_where_waves_differ")), "Items where `A_old != A_new`"),
    ("of_those_full_persona_equals_wave4", _pct(_get(r, "of_those_full_persona_equals_wave4")),
     "**The deciding number.** On items where the two waves differ, `full_persona` has to pick one; this is how often it picks wave 4"),
    ("contains_wave4_answers", contains, "True when the deciding number is above 99%"),
])}

**How to read it.** Looking only at `full_persona_equals_wave4` is not enough: people give the same
answer twice on most items, and there you cannot tell which wave `full_persona` used. The deciding
evidence is the items where the waves differ. `full_persona` matches wave 4 on all of them.

**This is a usage trap, not a data error.** The dataset README states it: "for questions that appear
in both waves 1-3 and wave 4, the wave 4 responses are used". So
`full_persona = wave1_3_persona + wave4_Q_wave4_A`. Putting it in a prompt to predict wave 4 shows the
model the answer key, with no error raised. **Decision:** `full_persona` (`persona_json`,
`persona_text`) is never a model input. `wave_split` already holds the same information, separated.
"""
    return verdict, body


def section_summary(r: dict) -> tuple[str, str]:
    hits = {k: v for k, v in r.items() if k != "participants"} if isinstance(r, dict) else {}
    verdict = WARN
    rows = [(k, _num(v), "People whose summary contains a word from this group") for k, v in hits.items()]
    rows.append(("participants", _num(_get(r, "participants")), "Summaries scanned"))
    body = f"""## 5. `persona_summary_keywords`: does `persona_summary` mention wave-4 tasks? — {verdict}

**What it checks.** `persona_summary` exists only in the `full_persona` config, which holds the
wave-4 answers (Section 4). If the summary had been written from `full_persona`, it could carry
wave-4 results in compressed form.

**How it is computed.** `check_persona_summary()` runs a case-insensitive regex per keyword group,
named after the wave-4 tasks, and counts **people** (not occurrences) with at least one match:

| Group | Pattern |
|---|---|
| pricing/purchase | `purchas*`, `pric*`, `willingness to pay` |
| anchoring | `anchor*` |
| framing/disease | `framing`, `disease` |
| sunk cost | `sunk cost` |
| false consensus | `false consensus` |
| wave 4 | `wave 4`, `wave4` |

{_table(rows)}

**How to read it.** The few matches were read by hand. All are incidental words in text the person
wrote in waves 1–3:

- Forward Flow word chains: "... store -> shopping -> customers -> purchasing ...", "... boat -> dock -> anchor ..."
- Open answers: "... getting down with terminal or disease ..."
- Ultimatum-game thoughts: "... good price; split it ..."

Stronger evidence than the keyword scan: every summary has the same section headings (demographics,
Big 5, need for cognition, CRT, numeracy, risk aversion, trust game, ...), all of them wave 1–3 constructs.

**Why this is a warning, not a pass.** (1) A keyword scan only finds exact words. (2) The summary's
percentiles (e.g. "26th percentile") are computed over all 2,058 people, test set included: a mild
population-level leak. **Decision:** do not use `persona_summary` directly; build a compact persona
from `raw_data/wave 1-3 scores.csv` with percentiles fitted on the training split.
"""
    return verdict, body


def section_setting_b(r: dict) -> tuple[str, str]:
    verdict = WARN
    body = f"""## 6. `setting_b`: earlier target answers inside `wave1_3_response.csv` — {verdict}

**What it checks.** Whether the convenient tabular file contains the earlier answers to the questions
we must predict.

**How it is computed.** `check_setting_b_columns()` counts the file's columns and how many of the
target columns it contains.

{_table([
    ("wave1_3_response_columns", _num(_get(r, "wave1_3_response_columns")), "Columns in the file (excluding pid)"),
    ("target_columns_present_in_wave1_3_response", _num(_get(r, "target_columns_present_in_wave1_3_response")),
     "Target columns present, i.e. `A_old` for every wave-4 item"),
    ("note", _get(r, "note"), ""),
])}

**How to read it.** 760 = 634 persona columns + 126 earlier answers. The file merges
`wave1_3_persona_json` and `wave4_Q_wave1_3_A`, but its name suggests persona only. Code like
`X = df.drop(columns="pid")` raises no error, yet the model sees the previous answer to the very
question it predicts. It learns to copy, and scores close to human retest (~82%), without using the
persona at all. That is **Setting B**, not Setting A.

| | Setting A (primary) | Setting B (diagnostic) |
|---|---|---|
| Input | 634 persona columns | 634 + 126 earlier answers |
| Question | Can the profile predict behaviour? | How much does the previous answer to the same question help? |

**Guard.** Inputs are selected only through `item_registry.setting_a_inputs()` (`role == "persona"`), which
asserts that no target column is present.
"""
    return verdict, body


def section_conditions(r: dict) -> tuple[str, str]:
    ok = isinstance(r, dict) and r and all(
        v["same_condition"] == 1.0 and not (v["multi_condition_w13"] or v["multi_condition_w4"]
                                             or v["no_condition_w13"] or v["no_condition_w4"])
        for v in r.values())
    verdict = PASS if ok else WARN
    rows = ["| Experiment | Conditions (people in wave 4) | same_condition | n_both | multi (w1-3 / w4) | none (w1-3 / w4) |",
            "|---|---|---|---|---|---|"]
    for exp, v in (r or {}).items():
        counts = " · ".join(f"{k}: {n:,}" for k, n in v["counts"].items())
        rows.append(f"| {exp} | {counts} | {v['same_condition']} | {v['n_both']:,} | "
                    f"{v['multi_condition_w13']} / {v['multi_condition_w4']} | {v['no_condition_w13']} / {v['no_condition_w4']} |")
    body = f"""## 7. `conditions`: between-subject experiments — {verdict}

**What it checks.** In a between-subject experiment, people are randomised into groups that see
different versions of a question (e.g. Disease problem framed as lives *saved* vs. lives *lost*).
Each person must see exactly one condition, and the **same** one in both waves.

**How it is computed.** `check_conditions()` works from the CSVs. Every condition has its own columns,
and the survey forced a response, so a non-blank column means "this person saw this condition". It
infers each person's condition in `wave1_3_response.csv` and in `wave4_response.csv`, then compares them.

| Field | Meaning | Expected |
|---|---|---|
| `same_condition` | Share of people with the same condition in both waves | 1.0 |
| `n_both` | People whose condition could be identified in both waves | 2,058 |
| `multi_condition_*` | People with answers in more than one condition (data error) | 0 |
| `no_condition_*` | People with no answer in any condition | 0 |
| `counts` | People per condition (randomisation balance) | roughly equal |

{chr(10).join(rows)}

**How to read it.** All three properties hold for every experiment. Less is More and both
Proportion-dominance experiments have identical counts (735 / 672 / 651): they share one
randomisation, so they are not independent experiments.

**Why it matters.** (1) Retest compares the same question only if the condition is unchanged.
(2) Population baselines must be per condition (anchoring low vs. high medians: 20 vs. 45 countries).
(3) A person's answer in one condition says nothing directly about their answer in the other condition.
"""
    return verdict, body


def section_old_new(r: dict) -> tuple[str, str]:
    if not r:
        return INFO, f"""## 8. `old_new_same_questions`: same questions in old and new, per person — {INFO}

{RERUN}
"""
    ok = _get(r, "passed", False) is True
    verdict = PASS if ok else FAIL
    body = f"""## 8. `old_new_same_questions`: same questions in old and new, per person — {verdict}

**What it checks.** For each person, `wave4_Q_wave1_3_A` (old) and `wave4_Q_wave4_A` (new) must contain
the same questions. This confirms Section 7 directly from the JSON, independent of the CSVs.

**How it is computed.** `check_old_new_same_questions()` compares, per pid, the set of
(block, QuestionID) pairs and then the question text. For pricing the text includes the product and
the price shown, so equal text also means the same product at the same price in the same slot.

{_table([
    ("participants", _num(_get(r, "participants")), "People checked"),
    ("questions_old_slash_new", _get(r, "questions_old_slash_new"), "Question counts (old/new) → number of people"),
    ("same_question_set", _num(_get(r, "same_question_set")), "People whose old and new hold the same (block, QID) set"),
    ("same_question_text", _num(_get(r, "same_question_text")), "...and identical question text"),
    ("mismatched_pids", _get(r, "mismatched_pids"), "Up to 5 people who failed, if any"),
    ("distinct_condition_combinations", _num(_get(r, "distinct_condition_combinations")),
     "Different question sets across people. High because ~15 experiments are randomised independently"),
    ("passed", _get(r, "passed"), "True when every person matches on set and text"),
])}

**How to read it.** Questions differ **between** people (randomised conditions) but are identical
**within** a person across waves. So retest compares the same question, and rows must always be joined
on `pid`, never by position.
"""
    return verdict, body


def render(report: dict) -> str:
    sections = [
        ("registry", section_registry),
        ("wave_split_json", section_wave_split),
        ("near_duplicate_texts", section_near_dup),
        ("full_persona_trap", section_full_persona),
        ("persona_summary_keywords", section_summary),
        ("setting_b", section_setting_b),
        ("conditions", section_conditions),
        ("old_new_same_questions", section_old_new),
    ]
    rendered = [(key, *fn(report.get(key, {}))) for key, fn in sections]
    summary = "\n".join(f"| {i} | `{key}` | {verdict} |" for i, (key, verdict, _) in enumerate(rendered, 1))
    return f"""# Leakage audit

Generated by `python -m scripts.audit_leakage` from `leakage_audit.json`. Do not edit by hand.
Dataset revision `{report.get('dataset_revision', '?')}`.

**Setting A** (the primary task) predicts wave-4 answers from the wave 1–3 persona only. This audit
shows that the persona input contains no wave-4 question or answer, and documents the two sources that
do contain them.

| Verdict | Meaning |
|---|---|
| {PASS} | Check passed |
| {WARN} | A known hazard, documented and guarded against; not a defect in our inputs |
| {FAIL} | Blocker: stop and investigate |
| {INFO} | Not computed in this JSON |

| # | Key | Verdict |
|---|---|---|
{summary}

**Summary.** The inputs we use (`wave_split` persona, 634 columns, selected through the item registry)
contain no wave-4 question or answer. Two sources do contain wave-4 answers and are never used as
Setting A input: `full_persona` (the wave-4 answers) and the target columns of `wave1_3_response.csv`
(the earlier answers).

{chr(10).join(body for _, _, body in rendered)}"""


def main():
    report = json.loads((OUT / "leakage_audit.json").read_text())
    (OUT / "leakage_audit.md").write_text(render(report))
    print(f"wrote {OUT / 'leakage_audit.md'}")


if __name__ == "__main__":
    main()
