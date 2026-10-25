"""Teaching figure: how the paired bootstrap CI of "E0 minus LLM" is formed.

Per test person: E0 score minus the paper's main LLM (text persona, GPT-4.1-mini) score on the
cells both predicted. Resample the 311 people with replacement 2,000 times (same seed and
procedure as src.metrics.paired_bootstrap_delta), plot the 2,000 means and the 2.5 / 97.5
percentiles.

Needs: scripts.run_e0 and scripts.run_e4.
Writes: reports/figures/e0_bootstrap_vs_llm.png
Run:    python -m scripts.plot_bootstrap_example [--out PATH]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import data
from src import metrics as M
from src.eval import load_predictions, score, thresholds
from src.split import load_split

LLM = "E4_text_persona_gpt4_1_mini"
N_BOOT, SEED = 2000, 0


def paired_differences():
    test = load_split()["test"]
    th = thresholds()
    e0 = load_predictions("E0_population_median", "wave4")[0]
    llm = load_predictions(LLM, "wave4")[0]
    both = M.to_scoring_space(e0.reindex(test), th).notna() & M.to_scoring_space(llm.reindex(test), th).notna()
    a = score(e0, "wave4", test, mask=both).respondent
    b = score(llm, "wave4", test, mask=both).respondent
    return (a - b).dropna()


def main(out):
    x = paired_differences().to_numpy()
    rng = np.random.default_rng(SEED)
    boots = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    m = x.mean() * 100

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(boots, bins=50, color="#5B7FA6", edgecolor="white", linewidth=0.6)
    top = ax.get_ylim()[1]
    for v, label in [(lo, f"2.5th pct = {lo:.1f}"), (hi, f"97.5th pct = {hi:.1f}")]:
        ax.axvline(v, color="#333", lw=1.2, ls="--")
        ax.text(v, top * 0.93, " " + label, fontsize=9)
    ax.axvline(m, color="#C8553D", lw=2)
    ax.text(m, top * 0.72, f" mean of {len(x)} differences = {m:.1f}", fontsize=9, color="#C8553D")
    ax.set_xlim(-0.5, 6.5)
    ax.axvline(0, color="#999", lw=1)
    ax.text(0.05, top * 0.5, "0\n(no difference)\nno round reaches it", fontsize=8, color="#555")
    ax.set_xlabel("E0 − LLM (points), one value per bootstrap round")
    ax.set_ylabel("number of rounds")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{N_BOOT:,} bootstrap rounds of the paired difference (Test {len(x)}, wave 4)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"mean {m:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  rounds <= 0: {int((boots <= 0).sum())}  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(data.REPO_ROOT / "reports" / "figures" / "e0_bootstrap_vs_llm.png"))
    main(ap.parse_args().out)
