"""Exploratory tables for reports/01_data_exploration.md; figures via scripts/eda_figures.py.

Everything labelled "in-sample" here is fitted and scored on all 2,058 people.
It describes the data; it is NOT the E0 baseline, which is fitted on the
training split only (RQ1).

Run after scripts.audit_leakage, scripts.reproduce_retest and scripts.token_counts:
    python -m scripts.eda
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from src import data
from src import metrics as M
from src.item_registry import build_registry
from src.pricing import pricing_panel
from scripts import eda_figures

OUT = data.REPO_ROOT / "results" / "metrics"
FIG = data.REPO_ROOT / "reports" / "figures"


def demographics() -> pd.DataFrame:
    reg = build_registry()
    lab = data.load_responses("wave1_3", labels=True)
    rows = []
    for _, r in reg[reg.domain == "demographics"].iterrows():
        vc = lab[r.column].value_counts(normalize=True)
        for level, share in vc.items():
            rows.append(dict(column=r.column, question=r.question_text[:60], level=level, share=round(share, 3)))
    return pd.DataFrame(rows)


def missingness() -> pd.DataFrame:
    reg = build_registry()
    w13, w4 = data.load_responses("wave1_3"), data.load_responses("wave4")
    rows = []
    for (role, domain), g in reg.groupby(["role", "domain"]):
        cols = g.column.tolist()
        src = w13[cols]
        # Between-subject items are blank by design for people in other conditions.
        by_design = g.condition.notna().to_numpy()
        rows.append(dict(role=role, domain=domain, columns=len(cols),
                         missing_rate_w13=round(float(src.isna().to_numpy().mean()), 4),
                         missing_rate_w4=round(float(w4.reindex(columns=cols).isna().to_numpy().mean()), 4)
                         if role == "target" else None,
                         between_subject_columns=int(by_design.sum()),
                         columns_over_5pct_missing=int((src.isna().mean() > 0.05).sum())))
    return pd.DataFrame(rows)


def constant_predictions(T_fit: pd.DataFrame, truth: pd.DataFrame, panel: pd.DataFrame | None, mode: str):
    """In-sample population median per column; for pricing optionally per product x price."""
    med = T_fit.median()
    pred = pd.DataFrame(np.tile(med.to_numpy(), (len(truth), 1)), index=truth.index, columns=truth.columns)
    if panel is not None and mode == "product_price":
        cell = panel.groupby(["product_w4", "price_w4"]).answer_w13.median()
        p = panel.assign(pred=panel.set_index(["product_w4", "price_w4"]).index.map(cell))
        wide = p.pivot(index="pid", columns="slot", values="pred")
        pred[wide.columns] = wide.reindex(truth.index)
    return pred


def population_baselines(th) -> pd.DataFrame:
    """In-sample constant predictors, scored with the paper metric against each wave."""
    w13, w4 = data.load_responses("wave1_3"), data.load_responses("wave4")
    T13, T4 = M.to_scoring_space(w13, th), M.to_scoring_space(w4, th)
    mask = M.eligibility_mask(T13, T4)
    panel = pricing_panel()
    rows = []
    for name, mode in [("median per column (pricing by slot)", "slot"),
                       ("median per column, pricing by product x price", "product_price")]:
        pred = constant_predictions(T13, T13, panel, mode)
        for target, truth in [("wave4", T4), ("wave1_3", T13)]:
            acc = M.cell_accuracy(pred, truth)
            s = M.respondent_scores(acc, mask)
            rows.append(dict(predictor=name, target=target, paper_metric=round(s.mean(), 4),
                             pricing_task=round(M.task_scores(acc, mask)["pricing"], 4)))
    return pd.DataFrame(rows)


def per_task_ceiling_vs_floor(th) -> pd.DataFrame:
    w13, w4 = data.load_responses("wave1_3"), data.load_responses("wave4")
    T13, T4 = M.to_scoring_space(w13, th), M.to_scoring_space(w4, th)
    mask = M.eligibility_mask(T13, T4)
    panel = pricing_panel()
    pred = constant_predictions(T13, T13, panel, "product_price")
    retest = pd.read_csv(OUT / "human_retest_by_task.csv", index_col=0)
    df = retest[["retest", "ci_lo", "ci_hi", "random_expected_vs_w4"]].copy()
    df["population_median_in_sample"] = M.task_scores(M.cell_accuracy(pred, T4), mask)
    df["headroom_pp"] = (df.retest - df.population_median_in_sample) * 100
    return df.sort_values("headroom_pp")


def demand_curve() -> pd.DataFrame:
    p = pricing_panel()
    p["multiplier"] = (p.price_w4 / p.groupby("product_w4").price_w4.transform("median")).round(1)
    p["buy_w13"], p["buy_w4"] = p.answer_w13.eq(1), p.answer_w4.eq(1)
    g = p.groupby("multiplier").agg(n=("pid", "size"), buy_w13=("buy_w13", "mean"), buy_w4=("buy_w4", "mean"),
                                    agree=("answer_w13", lambda s: np.nan))
    g["agree"] = p.assign(a=p.answer_w13.eq(p.answer_w4)).groupby("multiplier").a.mean()
    return g


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    th = M.decile_thresholds(data.load_responses("wave1_3"))
    demo = demographics()
    demo.to_csv(OUT / "eda_demographics.csv", index=False)
    miss = missingness()
    miss.to_csv(OUT / "eda_missingness.csv", index=False)
    pop = population_baselines(th)
    pop.to_csv(OUT / "eda_population_baselines_in_sample.csv", index=False)
    tasks = per_task_ceiling_vs_floor(th)
    tasks.round(4).to_csv(OUT / "eda_task_headroom.csv")
    dc = demand_curve()
    dc.round(4).to_csv(OUT / "eda_demand_curve.csv")
    eda_figures.main(FIG)
    pd.set_option("display.width", 200)
    print(miss.to_string(index=False))
    print(pop.to_string(index=False))
    print(tasks.round(3).to_string())
    print(dc.round(3).to_string())


if __name__ == "__main__":
    main()
