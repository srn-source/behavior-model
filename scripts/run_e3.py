"""E3: k nearest neighbours on the raw persona (src/knn.py).

Protocol: neighbours and E0 buy rates from train; k chosen on val (paper metric, same target);
predictions saved for val and test. Test is only scored by scripts.evaluate.

Outputs:
  results/predictions/E3_knn_persona__{target}.parquet / .json
  results/predictions/E3_knn_neighbours__{target}.csv   10 nearest training people per val/test person
                                                        (person-level, kept out of git like all predictions)
  results/metrics/e3_tuning.csv                          val score for every k tried

Run: python -m scripts.run_e3      (~1 minute)
"""
from __future__ import annotations

import pandas as pd

from src import data
from src.eval import PRED_DIR, TARGETS, save_predictions, score
from src.knn import KNNModel
from src.split import load_split

OUT = data.REPO_ROOT / "results" / "metrics"
EXP_ID = "E3_knn_persona"
K_GRID = [10, 25, 50, 100, 200, 400]
N_NEIGHBOURS_SAVED = 10


def main():
    split = load_split()
    log = []
    for target in TARGETS:
        model = KNNModel(target, k=K_GRID[0]).fit(split["train"])
        best = (-1.0, None)
        for k in K_GRID:
            model.k = k
            s = score(model.predict(split["val"]), target, split["val"]).mean
            log.append(dict(exp_id=EXP_ID, target=target, k=k, val_score=s))
            if s > best[0]:
                best = (s, k)
        model.k = best[1]
        pids = split["val"] + split["test"]
        save_predictions(EXP_ID, target, model.predict(pids), dict(
            label=f"E3 k nearest neighbours on raw persona (k={best[1]}, train)",
            regime="supervised-on-items", setting="A", inputs="persona_raw (cosine similarity)",
            fitted_on=f"train ({len(split['train'])} people), {target} answers",
            selected_on="val", k=best[1], val_score=round(best[0], 4),
        ))
        order, sim = model.neighbour_order(pids)
        rows = [dict(pid=pid, rank=r + 1, neighbour_pid=int(model.train[order[i, r]]), similarity=round(float(sim[i, r]), 4))
                for i, pid in enumerate(pids) for r in range(N_NEIGHBOURS_SAVED)]
        pd.DataFrame(rows).to_csv(PRED_DIR / f"E3_knn_neighbours__{target}.csv", index=False)
        print(f"{EXP_ID} {target:8s} best k={best[1]:<4} val={best[0]:.4f}")
    pd.DataFrame(log).to_csv(OUT / "e3_tuning.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
