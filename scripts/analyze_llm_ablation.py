"""RQ5: prompting vs retrieval vs fine-tuning on the same 0.5B LLM. Test split, target = wave 4.

Reads only saved predictions (no model, CPU, under a minute):
  1. scores          A0, A1, A2, B1 (untrained Qwen2.5-0.5B-Instruct, four prompts) and C1 = E6, C2-prior, C2-knn
                     (the same model fine-tuned with each context), against E0 and E3; participant bootstrap CI.
  2. comparisons     paired deltas that change one thing at a time (prompt, training, context).
  3. decomposition   for the fine-tuned models: model − E0 split into pricing and the rest, by replacing the model's
                     pricing answers with E0's; the rest is also set against E3.
  4. pricing         share of pricing rows predicted "buy", mean P(buy) in slot 1 vs slots 2-40, and the correlation of
                     P(buy) with E0's buy rate for that product x price (the number written in the C2-prior prompt).
  5. per task        each model − E0, per task.

Outputs: results/metrics/llm_ablation.csv, llm_ablation_comparisons.csv, llm_ablation_decomposition.csv,
         llm_ablation_by_task.csv, llm_ablation.md; reports/figures/llm_ablation.png
Run: python -m scripts.analyze_llm_ablation
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import eda_figures  # noqa: F401  (applies the shared plot style)
from src import data
from src import metrics as M
from src.baselines import PRICING_COLUMNS
from src.eval import PRED_DIR, load_predictions, paired_delta, score
from src.llm_context import PopulationPrior
from src.pricing import pricing_panel
from src.split import load_split

OUT = data.REPO_ROOT / "results" / "metrics"
FIG = data.REPO_ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
E0, E3 = "E0_population_median", "E3_knn_persona"

# code, experiment id, approach, context in the prompt
MODELS = [
    ("A0", "E5a_qwen05_none", "Prompting", "questions only"),
    ("A1", "E5_qwen05_zeroshot", "Prompting", "persona"),
    ("A2", "E5b_qwen05_prior_median", "Prompting", "population median"),
    ("B1", "E5c_qwen05_knn_k20", "Retrieval", "20 similar people"),
    ("C1", "E6_qwen05_lora", "Fine-tuning", "persona"),
    ("C2-prior", "E6p_qwen05_lora_prior", "Fine-tuning", "population median"),
    ("C2-knn", "E6k_qwen05_lora_knn20", "Fine-tuning", "20 similar people"),
]
FINE_TUNED = ["C1", "C2-prior", "C2-knn"]
# (A, B, what changes)
COMPARISONS = [
    ("A1", "A0", "untrained: persona vs questions only"),
    ("A2", "A0", "untrained: population median vs questions only"),
    ("B1", "A0", "untrained: 20 similar people vs questions only"),
    ("C1", "A1", "training, persona prompt (E6 − E5)"),
    ("C2-prior", "A2", "training, population-median prompt"),
    ("C2-knn", "B1", "training, similar-people prompt"),
    ("C2-prior", "C1", "fine-tuned: population median vs persona"),
    ("C2-knn", "C1", "fine-tuned: 20 similar people vs persona"),
    ("C2-knn", "C2-prior", "fine-tuned: 20 similar people vs population median"),
]


def random_score() -> float:
    """Expected score of uniform random answers on the test split (written by scripts.evaluate)."""
    lb = pd.read_csv(OUT / "leaderboard.csv")
    return float(lb[(lb.exp_id == "REF_random") & (lb.target == "wave4") & (lb.split == "test")].score.iloc[0])


def with_e0_pricing(pred: pd.DataFrame, e0: pd.DataFrame) -> pd.DataFrame:
    hybrid = pred.copy()
    hybrid[PRICING_COLUMNS] = e0[PRICING_COLUMNS]
    return hybrid


def _delta(pred_a, pred_b, test, **row) -> dict:
    d = paired_delta(pred_a, pred_b, "wave4", test)
    return dict(**row, delta=d["delta"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n_people=d["n_people"])


def pricing_stats(exp_id: str, pred: pd.DataFrame, cell_rate: pd.Series, test) -> dict:
    stacked = pred.loc[test, PRICING_COLUMNS].stack()
    out = dict(predicts_buy=float((stacked == 1).mean()))
    path = PRED_DIR / "option_probs" / f"{exp_id}__wave4.parquet"
    if not path.exists():
        return out
    probs = pd.read_parquet(path)
    buy = probs[probs.column.isin(PRICING_COLUMNS) & (probs.option == 1) & probs.pid.isin(test)]
    position = buy.column.str.split("_").str[1].astype(int)
    rate = cell_rate.reindex(pd.MultiIndex.from_arrays([buy.pid, buy.column])).to_numpy()
    ok = ~np.isnan(rate)
    out.update(p_buy_slot1=float(buy.prob[position == 1].mean()), p_buy_slots_2_40=float(buy.prob[position > 1].mean()),
               corr_p_buy_vs_e0_rate=float(np.corrcoef(buy.prob.to_numpy()[ok], rate[ok])[0, 1]))
    return out


def e0_cell_rate(train, test) -> pd.Series:
    """For each test (pid, slot): the share of training people who bought that product at that price."""
    prior = PopulationPrior(train, "median")
    p = pricing_panel()
    p = p[p.pid.isin(test)]
    rate = [prior.buy_sum.get(k, np.nan) / prior.buy_n[k] if k in prior.buy_n else np.nan
            for k in zip(p.product_w4, p.price_w4)]
    return pd.Series(rate, index=pd.MultiIndex.from_arrays([p.pid, p.slot]))


def main():
    split = load_split()
    test = list(split["test"])
    pred = {code: load_predictions(exp, "wave4")[0] for code, exp, _, _ in MODELS}
    e0, e3 = load_predictions(E0, "wave4")[0], load_predictions(E3, "wave4")[0]
    cell_rate = e0_cell_rate(split["train"], test)

    rows, tasks = [], {}
    for code, exp, approach, context in MODELS + [("E0", E0, "Reference", "–"), ("E3", E3, "Reference", "–")]:
        p = pred.get(code, e0 if code == "E0" else e3)
        s = score(p, "wave4", test)
        mean, lo, hi = M.bootstrap_mean(s.respondent)
        row = dict(code=code, exp_id=exp, approach=approach, context=context,
                   fine_tuned=None if approach == "Reference" else code in FINE_TUNED,
                   score=mean, ci_lo=lo, ci_hi=hi, pricing_task=float(s.task["pricing"]))
        if code != "E0":
            d = paired_delta(p, e0, "wave4", test)
            row.update(delta_vs_E0=d["delta"], delta_ci_lo=d["ci_lo"], delta_ci_hi=d["ci_hi"])
        row.update(pricing_stats(exp, p, cell_rate, test))
        rows.append(row)
        tasks[code] = s.task
    table = pd.DataFrame(rows)
    by_task = pd.DataFrame(tasks)
    by_task = by_task.drop(columns="E0").sub(by_task["E0"], axis=0)

    comps = pd.DataFrame([_delta(pred[a], pred[b], test, a=a, b=b, change=c) for a, b, c in COMPARISONS]
                         + [_delta(pred["C2-knn"], e3, test, a="C2-knn", b="E3", change="best fine-tuned vs E3")])
    dec = []
    for code in FINE_TUNED:
        hybrid = with_e0_pricing(pred[code], e0)
        dec += [_delta(pred[code], e0, test, model=code, part="total (model − E0)"),
                _delta(pred[code], hybrid, test, model=code, part="pricing (model − model with E0's pricing)"),
                _delta(hybrid, e0, test, model=code, part="rest (model with E0's pricing − E0)"),
                _delta(hybrid, e3, test, model=code, part="rest vs E3 (model with E0's pricing − E3)")]
    dec = pd.DataFrame(dec)

    table.to_csv(OUT / "llm_ablation.csv", index=False, float_format="%.4f")
    comps.to_csv(OUT / "llm_ablation_comparisons.csv", index=False, float_format="%.4f")
    dec.to_csv(OUT / "llm_ablation_decomposition.csv", index=False, float_format="%.4f")
    by_task.to_csv(OUT / "llm_ablation_by_task.csv", float_format="%.4f")
    rnd = random_score()
    write_markdown(table, comps, dec, by_task, rnd)
    plot(table, dec, rnd)
    pd.set_option("display.width", 200)
    print(table.drop(columns=["exp_id"]).round(4).to_string(index=False))
    print(comps.round(4).to_string(index=False))
    print(dec.round(4).to_string(index=False))


def write_markdown(table, comps, dec, by_task, rnd):
    pp = lambda x: f"{x * 100:+.2f}"
    ci = lambda r: f"{pp(r.delta)} [{pp(r.ci_lo)}, {pp(r.ci_hi)}]"
    pct = lambda x: "–" if pd.isna(x) else f"{x:.1%}"
    num = lambda x, f="{:.2f}": "–" if pd.isna(x) else f.format(x)
    lines = ["# Prompting vs retrieval vs fine-tuning, Qwen2.5-0.5B-Instruct (test split, target = wave 4)", "",
             "Generated by `python -m scripts.analyze_llm_ablation`. Do not edit by hand.", "",
             "## Scores (95% CI) and pricing behaviour", "",
             "| Code | Approach | Context in the prompt | Fine-tuned | Score [95% CI] | − E0 [95% CI] | Pricing task | "
             "Predicts buy | P(buy) slot 1 → 2-40 | corr(P(buy), E0 rate) |",
             "|---|---|---|---|---|---|---:|---:|---|---:|"]
    for r in table.itertuples():
        d = "–" if pd.isna(getattr(r, "delta_vs_E0", np.nan)) else \
            f"{pp(r.delta_vs_E0)} [{pp(r.delta_ci_lo)}, {pp(r.delta_ci_hi)}]"
        ft = "–" if r.fine_tuned is None else "yes" if r.fine_tuned else "no"
        slots = "–" if pd.isna(getattr(r, "p_buy_slot1", np.nan)) else f"{r.p_buy_slot1:.2f} → {r.p_buy_slots_2_40:.2f}"
        lines.append(f"| {r.code} | {r.approach} | {r.context} | {ft} | {pct(r.score)} "
                     f"[{pct(r.ci_lo)}, {pct(r.ci_hi)}] | {d} | {pct(r.pricing_task)} | {pct(r.predicts_buy)} | {slots} | "
                     f"{num(getattr(r, 'corr_p_buy_vs_e0_rate', np.nan), '{:+.3f}')} |")
    lines += ["", f"Uniform random answers: {rnd:.1%}.", "",
              "## Comparisons that change one thing (points, paired per person)", "",
              "| A − B | What changes | Δ [95% CI] |", "|---|---|---|"]
    lines += [f"| {r.a} − {r.b} | {r.change} | {ci(r)} |" for r in comps.itertuples()]
    lines += ["", "## Fine-tuned models: pricing vs the rest (points, paired per person)", "",
              "| Model | Part | Δ [95% CI] |", "|---|---|---|"]
    lines += [f"| {r.model} | {r.part} | {ci(r)} |" for r in dec.itertuples()]
    lines += ["", "## Per task, model − E0 (points)", "",
              "| Task | " + " | ".join(by_task.columns) + " |", "|---|" + "---:|" * len(by_task.columns)]
    lines += [f"| {t} | " + " | ".join(f"{v * 100:+.1f}" for v in r.values) + " |" for t, r in by_task.iterrows()]
    (OUT / "llm_ablation.md").write_text("\n".join(lines) + "\n")


def plot(table, dec, rnd):
    colour = {"Prompting": GRAY, "Retrieval": ORANGE, "Fine-tuning": AQUA, "Reference": BLUE}
    t = table.iloc[::-1].reset_index(drop=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        ax.errorbar(r.score, yi, xerr=[[r.score - r.ci_lo], [r.ci_hi - r.score]], fmt="o", ms=7,
                    color=colour[r.approach], ecolor=colour[r.approach], capsize=2)
        ax.text(r.ci_hi + 0.004, yi, f"{r.score:.1%}", va="center", fontsize=8.5, color=INK)
    ax.axvline(rnd, color=MUTED, lw=1, ls=":")
    ax.text(rnd + 0.002, 0.5, f"uniform random {rnd:.1%}", fontsize=8, color=MUTED, va="center")
    ax.set_yticks(y, [f"{r.code}{' (E6)' if r.code == 'C1' else ''} · {r.approach.lower()} · {r.context}" if r.approach != "Reference" else r.code
                      + (" population median" if r.code == "E0" else " kNN (k = 100)") for r in t.itertuples()],
                  fontsize=8.5)
    ax.set_xlim(0.54, 0.81)
    ax.set_xlabel("Paper metric, test split")
    ax.set_title("Same 0.5B model: untrained prompts vs fine-tuned", loc="left")

    parts = {"pricing (model − model with E0's pricing)": ("pricing", ORANGE),
             "rest (model with E0's pricing − E0)": ("everything else", AQUA)}
    d = dec[dec.part.isin(parts)]
    models = list(dict.fromkeys(d.model))
    yy = np.arange(len(models))[::-1]
    for k, (part, (lab, c)) in enumerate(parts.items()):
        s = d[d.part == part].set_index("model").reindex(models)
        off = 0.18 * (1 - 2 * k)
        bx.barh(yy + off, s.delta * 100, height=0.32, color=c, label=lab)
        bx.errorbar(s.delta * 100, yy + off, xerr=[(s.delta - s.ci_lo) * 100, (s.ci_hi - s.delta) * 100], fmt="none",
                    ecolor=INK, lw=1, capsize=2)
    bx.axvline(0, color=INK, lw=1)
    bx.set_yticks(yy, models)
    bx.set_xlabel("Points vs E0 (paired, 95% CI)")
    bx.set_title("Fine-tuned models: where the gap to E0 comes from", loc="left")
    bx.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "llm_ablation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
