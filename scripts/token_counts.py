"""Measure persona and question lengths in tokens, to decide the long-persona strategy.

Tokenizers: Qwen2.5-0.5B-Instruct (the POC model, E5/E6, 32,768-token context) and o200k_base
(GPT-4.1 family, used by the precomputed simulations).

Persona formats measured:
  persona_text     wave_split.wave1_3_persona_text (the dataset's text rendering)
  persona_json     wave_split.wave1_3_persona_json
  persona_compact  the compact "Q: / A:" rendering used for E6 (src/ft_format.render_persona)
  persona_summary  full_persona.persona_summary (length only; never a model input, its percentiles use all people)

Two budgets in Qwen's context, after OVERHEAD_TOKENS for the chat template and the answer:
  one question     persona + the longest single wave-4 question (prompting one item at a time)
  all questions    persona + all of that person's wave-4 questions (one prompt per person, as E6 does)

Outputs results/metrics/token_counts.json and token_counts_by_participant.csv.

Run: python -m scripts.token_counts
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tiktoken
from transformers import AutoTokenizer

from src import data
from src.ft_format import render_persona

OUT = data.REPO_ROOT / "results" / "metrics"
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_CONTEXT = 32768
# Room kept for the chat template, instructions and the answer.
OVERHEAD_TOKENS = 256
FORMATS = ["persona_text", "persona_json", "persona_compact", "persona_summary"]


def question_prompt(q: dict) -> str:
    """Render one wave-4 question without its answer, the way a prompt would show it."""
    parts = [q.get("QuestionText", "")]
    for key in ("Options", "Rows", "Columns", "Statements"):
        if q.get(key):
            parts.append(f"{key}: " + " | ".join(map(str, q[key])))
    return "\n".join(parts)


def stats(x) -> dict:
    x = np.asarray(x)
    return dict(mean=float(x.mean().round(1)), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)), max=int(x.max()), min=int(x.min()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qwen = AutoTokenizer.from_pretrained(QWEN_ID)
    o200k = tiktoken.get_encoding("o200k_base")
    ws = data.load_wave_split()
    fp = data.load_full_persona()

    def n_qwen(texts):
        return [len(ids) for ids in qwen(list(texts), add_special_tokens=False)["input_ids"]]

    def n_o200k(texts):
        return [len(ids) for ids in o200k.encode_batch(list(texts))]

    texts = dict(persona_text=ws.wave1_3_persona_text, persona_json=ws.wave1_3_persona_json,
                 persona_compact=ws.wave1_3_persona_json.map(render_persona),
                 persona_summary=fp.persona_summary.reindex(ws.index).fillna(""))
    per = pd.DataFrame(index=ws.index)
    for name, t in texts.items():
        per[f"{name}_qwen"] = n_qwen(t)
        per[f"{name}_o200k"] = n_o200k(t)

    # Wave-4 questions (answers removed): per-question and per-person totals.
    q_rows = []
    for pid, blob in ws.wave4_Q_wave4_A.items():
        for block, q in data.iter_questions(blob):
            if q.get("is_descriptive"):
                continue
            q_rows.append(dict(pid=pid, block=block.strip(), qid=q["QuestionID"], text=question_prompt(q)))
    qdf = pd.DataFrame(q_rows)
    qdf["qwen"] = n_qwen(qdf.text)
    per["wave4_questions_qwen"] = qdf.groupby("pid").qwen.sum()

    longest_q = int(qdf.qwen.max())
    budget_one = QWEN_CONTEXT - OVERHEAD_TOKENS - longest_q
    budget_all = QWEN_CONTEXT - OVERHEAD_TOKENS - per.wave4_questions_qwen  # per person

    report = dict(
        tokenizers=dict(qwen=QWEN_ID, openai="o200k_base"),
        qwen_context=QWEN_CONTEXT,
        overhead_reserved=OVERHEAD_TOKENS,
        **{name: {"qwen": stats(per[f"{name}_qwen"]), "o200k": stats(per[f"{name}_o200k"])} for name in FORMATS},
        wave4_question_qwen=dict(per_question=stats(qdf.qwen), per_person_total=stats(per.wave4_questions_qwen),
                                 longest_block=qdf.loc[qdf.qwen.idxmax(), "block"]),
        persona_budget_one_question_qwen=int(budget_one),
        persona_budget_all_questions_qwen=stats(budget_all),
        share_fits_one_question_qwen={name: round(float((per[f"{name}_qwen"] <= budget_one).mean()), 4) for name in FORMATS},
        share_fits_all_questions_qwen={name: round(float((per[f"{name}_qwen"] <= budget_all).mean()), 4) for name in FORMATS},
    )
    with open(OUT / "token_counts.json", "w") as f:
        json.dump(report, f, indent=2)
    per.to_csv(OUT / "token_counts_by_participant.csv")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
