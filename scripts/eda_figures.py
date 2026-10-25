"""The three EDA figures used in reports/01_data_exploration.md, plus the shared plot style.

Called at the end of `python -m scripts.eda`. Reads tables that scripts.eda, scripts.reproduce_retest and
scripts.token_counts write to results/metrics/. A figure whose inputs are missing is skipped with a message.

  retest_by_task.png   human retest vs population median vs random, per task
  demand_curve.png     share buying by price in both waves, and how consistent people are by price
  persona_tokens.png   persona length vs the Qwen2.5 context window

Render into another folder (nothing in reports/ is touched):
    python -m scripts.eda_figures --out /tmp/some_dir/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data

RESULTS = data.REPO_ROOT / "results" / "metrics"
DEFAULT_OUT = data.REPO_ROOT / "reports" / "figures"

# Reference palette (dataviz skill): categorical slots 1-3, ink, muted ink, neutral.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRAY = "#0b0b0b", "#52514e", "#8a8984"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titlecolor": INK,
})


def _save(fig, out: Path, name: str):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _need(*paths: Path) -> bool:
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"  skipped: needs {p.relative_to(data.REPO_ROOT)} (run the step that writes it)")
    return not missing


def fig_retest_by_task(out: Path):
    path = RESULTS / "eda_task_headroom.csv"
    if not _need(path):
        return
    t = pd.read_csv(path, index_col=0).sort_values("retest")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.barh(y, t.retest, height=0.55, color=BLUE, label="Human retest (wave 4 vs waves 1-3)")
    ax.errorbar(t.retest, y, xerr=[t.retest - t.ci_lo, t.ci_hi - t.retest], fmt="none", ecolor=INK, lw=1, capsize=2)
    ax.scatter(t.population_median_in_sample, y, marker="D", s=30, color=ORANGE, zorder=3,
               edgecolor="white", linewidth=1, label="Population median (in-sample, exploratory)")
    ax.scatter(t.random_expected_vs_w4, y, marker="|", s=120, color=GRAY, zorder=3, label="Uniform random (expected)")
    ax.set_yticks(y, t.index)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("Paper accuracy metric (1 - |a - b| / range)")
    ax.set_title("Retest reliability vs. trivial baselines, by task")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.12), frameon=False, fontsize=8)
    _save(fig, out, "retest_by_task.png")


def fig_demand(out: Path):
    path = RESULTS / "eda_demand_curve.csv"
    if not _need(path):
        return
    g = pd.read_csv(path, index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))
    a.plot(g.index, g.buy_w13, color=BLUE, lw=2, marker="o", ms=5, label="Waves 1-3")
    a.plot(g.index, g.buy_w4, color=ORANGE, lw=2, marker="o", ms=5, label="Wave 4")
    a.set_ylim(0, 1)
    a.set_xlabel("Price shown / product's median price")
    a.set_ylabel("Share who would buy")
    a.set_title("Demand falls with price, the same in both waves")
    a.legend(frameon=False)
    b.plot(g.index, g.agree, color=BLUE, lw=2, marker="o", ms=5)
    b.set_ylim(0.75, 0.92)
    b.set_xlabel("Price shown / product's median price")
    b.set_ylabel("Same answer in both waves")
    b.set_title("Consistency is lowest near the reference price")
    fig.tight_layout()
    _save(fig, out, "demand_curve.png")


def fig_tokens(out: Path):
    paths = [RESULTS / "token_counts_by_participant.csv", RESULTS / "token_counts.json"]
    if not _need(*paths):
        return
    per = pd.read_csv(paths[0], index_col=0)
    tc = json.loads(paths[1].read_text())
    items = [("persona_json_qwen", "Persona (JSON)"), ("persona_text_qwen", "Persona (text)"),
             ("persona_compact_qwen", "Compact Q/A (E6)"), ("persona_summary_qwen", "persona_summary")]
    fig, ax = plt.subplots(figsize=(10, 3.1))
    for i, (c, lab) in enumerate(items):
        ax.barh(i, per[c].median(), height=0.5, color=BLUE)
        ax.text(per[c].median() + 300, i, f"{per[c].median():,.0f}", va="center", color=INK, fontsize=9)
    one, allq = tc["persona_budget_one_question_qwen"], tc["persona_budget_all_questions_qwen"]["median"]
    ax.axvline(tc["qwen_context"], color=ORANGE, lw=2, label=f"Qwen2.5 context ({tc['qwen_context']:,})")
    ax.axvline(one, color=ORANGE, lw=1, ls="--", label=f"Left with one question ({one:,})")
    ax.axvline(allq, color=ORANGE, lw=1, ls=":", label=f"Left with all wave-4 questions ({allq:,.0f})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    ax.set_yticks(range(len(items)), [l for _, l in items])
    ax.set_xlabel("Median tokens per participant (Qwen2.5-0.5B tokenizer)")
    ax.set_title("Only the compact persona fits next to a whole wave-4 questionnaire")
    _save(fig, out, "persona_tokens.png")


FIGURES = [fig_retest_by_task, fig_demand, fig_tokens]


def main(out: Path = DEFAULT_OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        print(f"{fn.__name__} ...")
        fn(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="folder for the PNGs")
    main(Path(ap.parse_args().out))
