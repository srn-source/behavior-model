"""E4: the dataset's precomputed LLM simulations, re-scored on our split.

These were produced by the dataset authors on all participants; nothing is trained here.
We only (1) check each file is keyed to the right people, (2) keep our val/test people,
and (3) save them in the common prediction format. The same predictions are scored
against both targets.

What each spec's input was is taken from its folder name; the prompts were not
re-run or verified here.

Run: python -m scripts.run_e4
"""
from __future__ import annotations

import re

import pandas as pd

from src import data
from src import metrics as M
from src.eval import TARGETS, save_predictions
from src.split import load_split

# Folder name -> (input described by the folder name, regime, caveat)
SPEC_INFO = {
    "Demographics Only": ("demographics only", "item-zero-shot", ""),
    "Text Persona": ("wave 1-3 persona as text", "item-zero-shot", ""),
    "JSON Persona": ("wave 1-3 persona as JSON", "item-zero-shot", ""),
    "Persona Summary": ("persona_summary (full_persona config; full-sample percentiles)", "item-zero-shot",
                        "summary percentiles use all participants"),
    "LLM Finetuning": ("fine-tuned on 500 participants' answers", "supervised-on-items",
                       "only the 1,558 people outside its training set have predictions"),
}


def describe(spec: str) -> tuple[str, str, str]:
    for key, info in SPEC_INFO.items():
        if spec.startswith(key):
            inputs, regime, caveat = info
            if "(Predicted Output)" in spec:
                caveat = "meaning of 'Predicted Output' not documented; verify before relying on it"
            return inputs, regime, caveat
    return "unknown", "unknown", "spec not described"


def slug(spec: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", spec.lower())).strip("_")


def alignment(spec: str) -> float:
    """Share of cells where the spec's own copy of the human wave-4 answers equals ours."""
    theirs = data.load_simulation(spec, "wave4")
    ours = data.load_responses("wave4")
    common = theirs.index.intersection(ours.index)
    a = theirs.loc[common, M.SCORED_COLUMNS]
    b = ours.loc[common, M.SCORED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return float(((a == b) | (a.isna() & b.isna())).to_numpy().mean())


def main():
    split = load_split()
    keep = set(split["val"] + split["test"])
    test = set(split["test"])
    for spec in data.list_simulation_specs():
        align = alignment(spec)
        if align < 0.999:
            print(f"SKIP {spec}: human answers in the file do not match ours ({align:.4f}); pids may be misaligned")
            continue
        sim = data.load_simulation(spec)
        pred = sim[sim.index.isin(keep)]
        inputs, regime, caveat = describe(spec)
        exp_id = "E4_" + slug(spec)
        for target in TARGETS:
            save_predictions(exp_id, target, pred, dict(
                label=f"E4 {spec}",
                regime=regime,
                setting="A",
                inputs=inputs,
                source=f"LLM_simulation_results/llm_simulations_all_specifications/{spec}",
                caveat=caveat,
                alignment_with_our_wave4=round(align, 4),
                test_people_with_predictions=int(len(test & set(pred.index))),
            ))
        print(f"{exp_id:60s} test people {len(test & set(pred.index)):3d}/{len(test)}  {caveat}")


if __name__ == "__main__":
    main()
