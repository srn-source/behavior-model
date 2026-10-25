"""E1, E2c, E2b: regularised regression per column from person features (docs call them E1, E2-P/C/E, E2-All).

  E1   demographics (D2, 14 columns)
  E2c  demographics + the raw answers of ONE other persona block (D1 BlockName), to see which block helps:
       Personality / Cognitive tests (+ Forward Flow) / Economic preferences      (docs: E2-P, E2-C, E2-E)
  E2b  every usable raw persona answer, all blocks (D2, 590 columns -> 806 features) (docs: E2-All)
A model on the 54 precomputed trait scores (D7) was tried and dropped: their formulas could be verified
for only 5 of 54 scores.

Protocol: fit on train; choose the regularisation strength on val (paper metric, same target);
refit nothing on val or test; save predictions for val and test. The test split is only
scored by scripts.evaluate, never looked at here.

Search: C (logistic, also used for pricing) over C_GRID with alpha fixed, then alpha
(median regression of the continuous columns) over ALPHA_GRID with the best C. Every model uses
every feature (no PCA). Parts that do not depend on the setting being changed are reused, not refitted
(median regression depends only on alpha; logistic and pricing only on C): same results, less time.

Outputs:
  results/predictions/{E1_demographics,E2b_persona_raw,E2c_*_raw}__{target}.parquet
  results/metrics/e1_e2_tuning.csv  (val score of every setting tried)

Run all:        python -m scripts.run_e1_e2                (about 30 minutes on CPU)
Run E2c only:   python -m scripts.run_e1_e2 --only E2c_personality_raw E2c_cognitive_raw E2c_economic_raw
Run only one:   python -m scripts.run_e1_e2 --only E2b_persona_raw
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src import data
from src.eval import TARGETS, save_predictions, score
from src.models import ColumnModels
from src.split import load_split

OUT = data.REPO_ROOT / "results" / "metrics"
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
C_GRID_WIDE = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]  # 806 features need stronger regularisation
C_GRID_MID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]  # E2c: 227-413 features, between E1 (60) and E2b (806)
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0]
ALPHA_DEFAULT = 0.01
EXPERIMENTS = {  # exp_id: (feature kind, label, C grid)
    "E1_demographics": ("demographics", "E1 demographics → regularised regression (train)", C_GRID),
    "E2b_persona_raw": ("persona_raw", "E2b all raw persona answers → regularised regression (train)", C_GRID_WIDE),
    "E2c_personality_raw": ("persona_raw:demographics+personality",
                            "E2c demographics + raw Personality block → regularised regression (train)", C_GRID_MID),
    "E2c_cognitive_raw": ("persona_raw:demographics+cognitive",
                          "E2c demographics + raw Cognitive tests block → regularised regression (train)", C_GRID_MID),
    "E2c_economic_raw": ("persona_raw:demographics+economic_preferences",
                         "E2c demographics + raw Economic preferences block → regularised regression (train)", C_GRID_MID),
}


def val_score(kind, target, split, C, alpha, reuse=None) -> tuple[float, ColumnModels]:
    m = ColumnModels(kind, target, C=C, alpha=alpha).fit(split["train"], reuse=reuse)
    return score(m.predict(split["val"]), target, split["val"]).mean, m


def main(only: list[str] | None = None):
    split = load_split()
    log = []
    for exp_id, (kind, label, c_grid) in EXPERIMENTS.items():
        if only and exp_id not in only:
            continue
        for target in TARGETS:
            t0 = time.time()
            best, prev = (-1.0, None, None, None), None
            for C in c_grid:  # step 1: C, with alpha fixed (median regressions fitted once, then reused)
                s, m = val_score(kind, target, split, C, ALPHA_DEFAULT, reuse=prev)
                prev = m
                log.append(dict(exp_id=exp_id, target=target, C=C, alpha=ALPHA_DEFAULT, val_score=s))
                if s > best[0]:
                    best = (s, C, ALPHA_DEFAULT, m)
            C_best, at_C_best = best[1], best[3]
            for alpha in ALPHA_GRID:  # step 2: alpha, with the best C (logistic and pricing reused)
                if alpha == ALPHA_DEFAULT:
                    continue
                s, m = val_score(kind, target, split, C_best, alpha, reuse=at_C_best)
                log.append(dict(exp_id=exp_id, target=target, C=C_best, alpha=alpha, val_score=s))
                if s > best[0]:
                    best = (s, C_best, alpha, m)
            s, C, alpha, model = best
            pred = model.predict(split["val"] + split["test"])
            save_predictions(exp_id, target, pred, dict(
                label=label, regime="supervised-on-items", setting="A",
                inputs=kind, n_features=int(model.features.transform(split["train"][:1]).shape[1]),
                fitted_on=f"train ({len(split['train'])} people), {target} answers",
                selected_on="val", C=C, alpha=alpha, val_score=round(s, 4),
            ))
            print(f"{exp_id:24s} {target:8s} best C={C:<6} alpha={alpha:<6} val={s:.4f}  ({time.time() - t0:.0f}s)")
    # keep earlier experiments' rows when only some experiments were re-run
    path = OUT / "e1_e2_tuning.csv"
    new = pd.DataFrame(log)
    if path.exists() and only:
        old = pd.read_csv(path)
        new = pd.concat([old[~old.exp_id.isin(new.exp_id.unique())], new])
    new.to_csv(path, index=False, float_format="%.4f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), help="run only these experiments")
    main(ap.parse_args().only)
