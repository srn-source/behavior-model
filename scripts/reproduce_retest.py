"""Human test-retest reliability under the paper's metric, with participant-bootstrap CIs.

Outputs (results/metrics/):
  human_retest_overall.json   headline numbers (paper aggregation + item-macro)
  human_retest_by_task.csv    per task: retest, uniform-random expectation
  human_retest_by_column.csv  per scored column

Also checks the published simulation files: re-scoring them with src.metrics must
reproduce the paper's numbers (stop rule set before the experiments).

Run: python -m scripts.reproduce_retest
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import data
from src import metrics as M

OUT = data.REPO_ROOT / "results" / "metrics"

# Published "Accuracy Summary - task level" values (mad_accuracy_summary.xlsx in the dataset).
PUBLISHED = {
    "Text Persona - GPT4.1-mini": (0.817, 0.717),
    "JSON Persona - GPT4.1": (0.818, 0.712),
    "Text Persona - Gemini-Flash2.5": (0.817, 0.694),
}
TOLERANCE = 0.0015  # published values are rounded to 3 decimals


def expected_random(truth: pd.DataFrame) -> pd.DataFrame:
    """Expected accuracy of a uniform random integer answer in [lo, hi], per cell."""
    out = pd.DataFrame(index=truth.index, columns=truth.columns, dtype=float)
    for spec in M.SPEC:
        levels = np.arange(spec.lo, spec.hi + 1)
        t = truth[spec.column].to_numpy(dtype=float)
        acc = 1 - np.abs(levels[None, :] - t[:, None]) / spec.range
        out[spec.column] = np.where(np.isnan(t), np.nan, acc.mean(axis=1))
    return out


def check_published() -> dict:
    w13, w4 = data.load_responses("wave1_3"), data.load_responses("wave4")
    th = M.decile_thresholds(w13)  # paper: thresholds from all participants' wave 1-3 answers
    T13, T4 = M.to_scoring_space(w13, th), M.to_scoring_space(w4, th)
    out = {}
    for spec, (pub_retest, pub_llm) in PUBLISHED.items():
        llm = M.to_scoring_space(data.load_simulation(spec), th).reindex(T13.index)
        mask = T13.notna() & T4.notna() & llm.notna()  # paper requires all three present
        retest = M.respondent_scores(M.cell_accuracy(T4, T13), mask).mean()
        model = M.respondent_scores(M.cell_accuracy(llm, T13), mask).mean()
        ok = abs(retest - pub_retest) <= TOLERANCE and abs(model - pub_llm) <= TOLERANCE
        out[spec] = dict(retest=round(retest, 4), published_retest=pub_retest,
                         llm=round(model, 4), published_llm=pub_llm, reproduced=bool(ok))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    published = check_published()
    print(json.dumps(published, indent=1))
    if not all(v["reproduced"] for v in published.values()):
        raise SystemExit("STOP: scorer does not reproduce the published numbers")

    w13, w4 = data.load_responses("wave1_3"), data.load_responses("wave4")
    th = M.decile_thresholds(w13)
    T13, T4 = M.to_scoring_space(w13, th), M.to_scoring_space(w4, th)
    mask = M.eligibility_mask(T13, T4)

    acc = M.cell_accuracy(T4, T13)  # symmetric: same as scoring wave 1-3 against wave 4
    rand13 = expected_random(T13)
    rand4 = expected_random(T4)

    resp = M.respondent_scores(acc, mask)
    mean, lo, hi = M.bootstrap_mean(resp)
    r_resp13 = M.respondent_scores(rand13, mask)
    r_resp4 = M.respondent_scores(rand4, mask)
    col = M.column_scores(acc, mask)

    # Exact-agreement rate: share of cells where the person gave the identical answer.
    exact = (T13 == T4).where(mask).stack().mean()

    overall = dict(
        dataset_revision=data.DATASET_REVISION,
        participants=int(len(resp)),
        eligible_cells=int(mask.to_numpy().sum()),
        scored_columns=len(M.SCORED_COLUMNS),
        tasks=len(M.TASKS),
        retest_paper_metric=dict(mean=round(mean, 4), ci95=[round(lo, 4), round(hi, 4)]),
        retest_item_macro=round(float(col.mean()), 4),
        retest_exact_agreement=round(float(exact), 4),
        random_expected_vs_wave1_3=round(float(r_resp13.mean()), 4),
        random_expected_vs_wave4=round(float(r_resp4.mean()), 4),
        decile_thresholds={k: v.round(2).tolist() for k, v in th.items()},
        published_check=published,
    )

    by_task = pd.DataFrame({
        "retest": M.task_scores(acc, mask),
        "random_expected_vs_w4": M.task_scores(rand4, mask),
        "n_cells": mask.T.groupby(pd.Series({c.column: c.task for c in M.SPEC})).sum().T.sum(),
    })
    # Per-task participant-bootstrap CI on the paper's per-respondent task mean.
    task_of = pd.Series({c.column: c.task for c in M.SPEC})
    per_resp_task = acc.where(mask).T.groupby(task_of).mean().T
    cis = {t: M.bootstrap_mean(per_resp_task[t].dropna())[1:] for t in M.TASKS}
    by_task["ci_lo"] = [cis[t][0] for t in by_task.index]
    by_task["ci_hi"] = [cis[t][1] for t in by_task.index]
    by_task = by_task.sort_values("retest")

    by_col = pd.DataFrame({
        "task": [M.SPEC_BY_COL[c].task for c in col.index],
        "retest": col,
        "exact_agreement": (T13 == T4).where(mask).mean(),
        "n": mask.sum(),
    })

    with open(OUT / "human_retest_overall.json", "w") as f:
        json.dump(overall, f, indent=2)
    by_task.round(4).to_csv(OUT / "human_retest_by_task.csv")
    by_col.round(4).to_csv(OUT / "human_retest_by_column.csv")
    print(json.dumps({k: v for k, v in overall.items() if k != "published_check"}, indent=1))
    print(by_task.round(3).to_string())


if __name__ == "__main__":
    main()
