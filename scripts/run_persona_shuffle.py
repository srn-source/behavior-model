"""Persona-shuffle ablation: do the models use the person's own persona? (test split, target = wave 4)

Every test person gets another test person's persona (src/ablation.donor_map: a derangement within the test
split), keeping their own wave-4 items, condition and products x prices. The models are not refitted: the
same fitted model (hyperparameters from the saved prediction metadata) predicts from the donor's features.

  real − shuffled > 0   the model's gain comes partly from the right person's information
  real − shuffled ≈ 0   the model does not use person-specific information (or uses it no better than chance)

Tabular models (E1, E2-P, E2-All, E3) are shuffled with seeds 0-4. The fine-tuned LLM (E6) needs a GPU run first:
  python -m scripts.ft_predict --exp E6_qwen05_lora --adapter outputs/qwen25_05b_lora --persona-shuffle-seed 0
and is scored here if results/predictions/ablation/E6_qwen05_lora_persona_shuffle_s0__wave4.parquet exists.

Outputs: results/metrics/persona_shuffle.csv, persona_shuffle.md
Run: python -m scripts.run_persona_shuffle      (~3-4 minutes, CPU)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data
from src import metrics as M
from src.ablation import DonorFeatures, donor_map
from src.baselines import PRICING_COLUMNS
from src.eval import PRED_DIR, gap_closed, load_predictions, paired_delta, retest_predictions, score, thresholds, eligible
from src.knn import KNNModel
from src.models import ColumnModels
from src.split import load_split

OUT = data.REPO_ROOT / "results" / "metrics"
SEEDS = range(5)
TABULAR = [  # exp_id, docs name
    ("E1_demographics", "E1"),
    ("E2c_personality_raw", "E2-P"),
    ("E2b_persona_raw", "E2-All"),
    ("E3_knn_persona", "E3"),
]
LLM = ("E6_qwen05_lora", "E6", 0)  # exp_id, docs name, shuffle seed run on GPU


def fit(exp_id: str, train):
    meta = load_predictions(exp_id, "wave4")[1]
    if exp_id.startswith("E3"):
        return KNNModel("wave4", k=int(meta["k"])).fit(train)
    return ColumnModels(meta["inputs"], "wave4", C=meta["C"], alpha=meta["alpha"]).fit(train)


def changed_share(a: pd.DataFrame, b: pd.DataFrame, pids) -> tuple[float, float]:
    """Share of eligible cells whose prediction differs: (non-pricing items, pricing)."""
    th = thresholds()
    A, B = M.to_scoring_space(a.reindex(pids), th), M.to_scoring_space(b.reindex(pids), th)
    m = eligible(pids) & A.notna() & B.notna()
    diff = (A - B).abs() > 1e-9
    non = [c for c in M.SCORED_COLUMNS if c not in PRICING_COLUMNS]
    share = lambda cols: float((diff[cols] & m[cols]).to_numpy().sum() / max(m[cols].to_numpy().sum(), 1))
    return share(non), share(PRICING_COLUMNS)


def compare(name, seed, real, shuffled, test, e0, retest) -> dict:
    d = paired_delta(real, shuffled, "wave4", test)
    g_real = gap_closed(real, e0, retest, "wave4", test)
    g_shuf = gap_closed(shuffled, e0, retest, "wave4", test)
    ch_items, ch_price = changed_share(real, shuffled, test)
    return dict(model=name, seed=seed, score_real=score(real, "wave4", test).mean,
                score_shuffled=score(shuffled, "wave4", test).mean,
                delta_real_minus_shuffled=d["delta"], delta_ci_lo=d["ci_lo"], delta_ci_hi=d["ci_hi"],
                gap_closed_real=g_real["gap_closed"], gap_closed_shuffled=g_shuf["gap_closed"],
                gap_shuffled_ci_lo=g_shuf["ci_lo"], gap_shuffled_ci_hi=g_shuf["ci_hi"],
                changed_items=ch_items, changed_pricing=ch_price)


def main():
    split = load_split()
    train, test = split["train"], split["test"]
    e0 = load_predictions("E0_population_median", "wave4")[0]
    retest = retest_predictions("wave4")
    rows = []
    for exp_id, name in TABULAR:
        model = fit(exp_id, train)
        real = model.predict(test)
        saved = load_predictions(exp_id, "wave4")[0]
        same = 1 - max(changed_share(real, saved, test))
        print(f"{name}: refit reproduces the saved predictions on {same:.2%} of cells")
        base = model.features
        for seed in SEEDS:
            model.features = DonorFeatures(base, donor_map(test, seed))
            shuffled = model.predict(test)
            model.features = base
            rows.append(compare(name, seed, real, shuffled, test, e0, retest))
            r = rows[-1]
            print(f"  seed {seed}: real {r['score_real']:.4f} shuffled {r['score_shuffled']:.4f} "
                  f"delta {r['delta_real_minus_shuffled'] * 100:+.2f} [{r['delta_ci_lo'] * 100:+.2f}, {r['delta_ci_hi'] * 100:+.2f}]")
    exp_id, name, seed = LLM
    path = PRED_DIR / "ablation" / f"{exp_id}_persona_shuffle_s{seed}__wave4.parquet"
    if path.exists():
        shuffled = pd.read_parquet(path)
        rows.append(compare(name, seed, load_predictions(exp_id, "wave4")[0], shuffled, test, e0, retest))
    else:
        print(f"{name}: no shuffled run at {path} (GPU step, see the docstring); skipped")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "persona_shuffle.csv", index=False, float_format="%.4f")
    write_markdown(res)
    print(res.round(4).to_string(index=False))


def write_markdown(res: pd.DataFrame):
    pct = lambda x: f"{x * 100:+.2f}"
    lines = ["# Persona-shuffle ablation (test split, target = wave 4)", "",
             "Generated by `python -m scripts.run_persona_shuffle`. Do not edit by hand.", "",
             "Each test person is given another test person's persona; their own items, conditions and products × "
             "prices are kept. Same fitted models, no refitting. Δ = real − shuffled, paired per person (2,000 "
             "bootstrap resamples). Gap closed = share of the E0 → human-retest gap closed.", "",
             "| Model | Real | Shuffled (seed 0) | Δ real − shuffled [95% CI] (seed 0) | Δ over seeds (mean, min–max) | "
             "Gap closed: real → shuffled | Predictions changed: items / pricing |",
             "|---|---:|---:|---|---|---|---|"]
    for name, g in res.groupby("model", sort=False):
        s0 = g[g.seed == g.seed.min()].iloc[0]
        over = (f"{pct(g.delta_real_minus_shuffled.mean())} ({pct(g.delta_real_minus_shuffled.min())} to "
                f"{pct(g.delta_real_minus_shuffled.max())}), {len(g)} seeds") if len(g) > 1 else "1 seed"
        lines.append(f"| {name} | {s0.score_real:.1%} | {s0.score_shuffled:.1%} | {pct(s0.delta_real_minus_shuffled)} "
                     f"[{pct(s0.delta_ci_lo)}, {pct(s0.delta_ci_hi)}] | {over} | {s0.gap_closed_real:+.1%} → "
                     f"{s0.gap_closed_shuffled:+.1%} | {s0.changed_items:.0%} / {s0.changed_pricing:.0%} |")
    (OUT / "persona_shuffle.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
