"""E0: population median, fitted on the training split, predicted for val and test.

Run: python -m scripts.run_e0
"""
from __future__ import annotations

from src.baselines import fit_e0, predict_e0
from src.eval import TARGETS, save_predictions
from src.split import load_split

EXP_ID = "E0_population_median"


def main():
    split = load_split()
    pids = split["val"] + split["test"]
    for target in TARGETS:
        model = fit_e0(split["train"], target)
        pred, info = predict_e0(model, pids)
        path = save_predictions(EXP_ID, target, pred, dict(
            label="E0 population median (train)",
            regime="supervised-on-items",
            setting="A",
            inputs="none (pricing: product and price shown)",
            fitted_on=f"train ({model['n_train']} people), {target} answers",
            **info,
        ))
        print(f"{target}: {path.name}  pricing sources {info['pricing_prediction_source']}")


if __name__ == "__main__":
    main()
