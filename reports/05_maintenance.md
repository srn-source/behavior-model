# 05 · Long-run maintenance

**Deliverable 5.** How to keep the model trustworthy over months and years: what to monitor, how to detect drift, when to retrain, how to version, and how to govern it. People's answers change. The panel changes. Prices inflate. Questions get reworded. The plan below treats each of these as a measurable event with a defined response.

What this repo already has is marked ✅. What is proposed is marked ➕. The dataset cannot test long-run drift: it has two waves about two weeks apart. So the monitoring thresholds below are starting values to be tuned on the first real refresh cycles, not validated numbers.

---

## 0. Summary

> A behavior model should be treated as a continuously validated measurement system, not a model that is trained once and assumed to remain accurate.

1. **Keep the population prior and the person-specific shift separate**, and maintain them on different clocks:
   - The **prior** (E0, the population answer per item and per product × price) is cheap to refresh from any new human sample and absorbs most population drift.
   - The **shift** (what personal information adds) is retrained only when it stops beating the refreshed prior.
   This works because the model is built as prior + shift ([02_model_plan.md](02_model_plan.md) §3).
2. **Ground truth comes from a standing human calibration panel**: a fresh sample that answers the same items on a schedule. Without new human answers, nothing below can detect that the model has gone wrong.
3. **Monitor the gain over the prior, not only accuracy**: the paired difference from a refreshed E0 and the gap closed, per subgroup. Raw accuracy can stay flat while the person-specific shift stops adding anything.
4. **Everything is versioned and re-scorable**: data revision, splits, item registry, features, model, metrics. Any old result can be reproduced and any new model compared on the same people.
5. **Governance**: consent-based data, deletion by retraining, a use policy enforced in the product ([04_business.md](04_business.md) §3–4), and a release gate that includes subgroup checks.

---

## 1. What changes over time

| Change | Example | What it breaks | Detected by |
|---|---|---|---|
| **Population drift** (data drift): the average answer moves | inflation shifts willingness to pay; an election shifts political opinions | the prior (E0) and every model built on it | calibration panel: E0 refit on new answers differs from the deployed E0 |
| **Relationship drift** (concept drift): what personal information predicts changes | income stops predicting purchase in a downturn | the person shift | gap closed on the calibration panel falls; per-feature contributions change |
| **Panel drift**: who is in the panel changes | recruiting skews younger; long-time panelists learn the questions | representativeness; subgroup accuracy | input monitoring of demographics and persona features |
| **Instrument drift**: the questionnaire changes | reworded question, new response option, new product, new price grid | item identity; scoring ranges; pricing cells | item registry validation at ingest |
| **Pipeline drift**: a code or dependency change alters outputs silently | a library update changes a solver; a feature encoder changes | anything | regression tests and frozen-test re-scoring |

**This dataset is a warning on each of these.**
- *Pipeline drift:* a pricing bug made E2-All 8.7 points worse than E0 on pricing, and the total score did not show it.
- *Representation changes:* removing PCA moved one conclusion (E2-C on waves 1–3 lost significance).

---

## 2. Monitoring

### 2.1 Signals and thresholds

| Layer | Signal | How computed | Starting threshold ➕ | Cadence |
|---|---|---|---|---|
| **Inputs** | Demographic mix of scored people vs training | share per category | any category moves > 5 points | per batch |
| | Persona feature distributions | population stability index (PSI) per feature | PSI > 0.2 on any of the top features | per batch |
| | Out-of-scope requests | unknown items, products, price levels, populations | any request is refused and counted; rising counts trigger review | continuous |
| **Outputs** | Predicted answer distributions | per item, vs the last release | distribution distance above the release-to-release baseline | per batch |
| | Share of predictions equal to E0 | per item, vs the last release | rises above the last release's value (the shift is fading toward the prior) | per batch |
| **Against humans** (calibration panel) | Accuracy, gap closed vs a **refit** E0 | [03_evaluation.md](03_evaluation.md) §1, §5 | gap closed CI includes 0 → the shift no longer adds value | each panel wave |
| | Per-task accuracy vs refit E0 | [03_evaluation.md](03_evaluation.md) §1.2 | a task where the model falls below E0 with the CI excluding 0 | each panel wave |
| | Subgroup accuracy | by sex, age, education, race | a group > 3 points below overall | each panel wave |
| | Retest of the panel itself | panelists re-answer a few items | retest falls, i.e. people are less consistent, so the ceiling moved | each panel wave |
| **Pipeline** | Invariant tests | `pytest` (leakage, "maximum regularization = E0", metric reproduction) ✅ | any failure blocks the release | every change |
| | Frozen-test re-score | `scripts/evaluate.py` on the frozen test split ✅ | any change in an existing row without a logged reason | every change |

The thresholds are placeholders: 5 points, PSI 0.2 (a common industry rule of thumb) and "below the last release". Set them from the variation seen in the first 2–3 calibration waves.

**Why the human-facing triggers are CI-based, not fixed drops.** A fixed rule such as "alert when the score falls 2 points" would not catch the failure that matters most here. All personal information together adds +0.4 to +0.8 points over E0, and the whole E0 → retest gap is 4.55 points. The person-specific shift could stop working entirely and the score would fall by less than 1 point. So each trigger compares against a reference measured on the same wave (a refit E0, the panel's own retest) with a CI.

### 2.2 The calibration panel ➕

| | |
|---|---|
| **What** | A fresh sample from the target population answers a fixed set of **anchor items** (items from each task, plus a rotating subset of pricing products and price levels) and provides the persona inputs |
| **Why** | The only source of new ground truth; also re-estimates the human retest reference, because the ceiling itself can move |
| **Size** | Enough for the decision it supports. The current test split (311 people) gives CIs of about ±0.3–0.4 points on paired differences and ±0.75 points on absolute scores; segment questions need more. |
| **Cadence** | Quarterly to start; monthly for fast-moving domains (prices) |
| **Kept separate** | Calibration answers are used to **evaluate** and to **refresh the prior**, never to tune the person model on the same wave (otherwise the evaluation leaks) |

---

## 3. Retraining

### 3.1 Two clocks

| Component | Retrain how | Cost | When |
|---|---|---|---|
| **Prior** (E0: medians per item and condition, buy rates per product × price) | recompute from the latest calibration wave(s) | seconds, CPU | every panel wave, or when population drift is detected |
| **Tabular shift** (E1/E2 regressions) | refit on the pooled panel with the refreshed prior as offset; regularization re-chosen on validation | about 20–30 minutes, CPU (current run times) | when gap closed vs the refreshed prior falls, or relationship drift is detected |
| **LLM shift** (E6 line) | LoRA fine-tune on the updated training people | about 1 GPU-hour for 1,440 people at 24k tokens (E6: 58.5 min on one H100) | as for the tabular shift, and only if the LLM line beats the tabular line on validation |

Because every model collapses to the prior under maximum regularization, a refreshed prior also refreshes the fallback behaviour of every model at once.

### 3.2 Triggers

| Type | Trigger | Action |
|---|---|---|
| **Scheduled** | each calibration wave | refresh the prior; re-score; publish the monitoring report |
| **Metric** | gap closed CI includes 0 on the panel | refit the shift; if it still does not beat the prior, serve the prior alone and say so |
| | a task falls below the refit E0 (CI excludes 0) | check whether the prior is stale (refresh); then the shift |
| | subgroup more than 3 points below overall | investigate; reweight or add data for that group; block the release if unexplained |
| **Event** | new or reworded question, new product, new price grid | treat as a **new item**: no prior exists, so collect human answers before serving it (see §4.2 on unseen pricing cells) |
| | major external event (inflation shock, election, policy change) | an unscheduled calibration wave for the affected domains |
| | deletion request from a panelist | remove their data; refresh the prior; retrain the shift at the next release (the tabular shift is cheap enough to retrain at once) |

**No fixed retraining schedule for the shift.** A small or unrepresentative wave can make a refit worse, so the shift is refit only on a trigger and released only through the gate below. The prior is refreshed each wave because it is cheap and directly measured, with E0's minimum of 20 people per cell and its fallback chain.

### 3.3 Release gate

A retrained model is released only if, on a **frozen test split** and the latest calibration wave:

1. It passes the invariant tests (`pytest`).
2. It is not worse than the deployed model: paired difference CI does not lie entirely below 0.
3. Its gap closed is not lower than the deployed model's on the calibration wave.
4. No task falls below E0 where the deployed model did not.
5. No subgroup flag is unexplained.
6. The model card is updated (§4.3).
7. It has run in **shadow** next to the deployed model for one calibration wave (both score the same people; only the deployed model's outputs are served), and the deployed release is kept as the **rollback target**.

Comparisons use `src/eval.py: paired_delta` and `gap_closed` on the same people. Test data is frozen per release generation. A fresh holdout is drawn when the generation changes, because the current test split has been used for many comparisons ([03_evaluation.md](03_evaluation.md) §3).

---

## 4. Versioning

### 4.1 What is versioned

| Artifact | Identifier | Status |
|---|---|---|
| Source data | dataset revision `f883165a` pinned in `src/data.py` and `configs/experiment.yaml` | ✅ |
| Splits | `data/splits.json` (seed 0, stratified) | ✅ |
| Item registry | which columns are persona / target, condition, task, scoring range (`src/item_registry.py`) | ✅ (code); ➕ version it as a data file with a hash when the instrument changes |
| Features | built by documented rules in `src/features.py`; each feature name keeps its source column (e.g. `QID11=2`) | ✅ |
| Predictions | one file per model and target with metadata (inputs, regularization chosen, fitted-on, selected-on) (`results/predictions/*.json`; per person, not published) | ✅ |
| Training runs | config, token counts, loss log (`results/metrics/e6_run_config.json`, `e6_train_manifest.json`, `e6_train_log.csv`) | ✅ |
| Metrics | one scorer writes every leaderboard and per-task table (`scripts/evaluate.py`) | ✅ |
| Model weights | LoRA adapter, tabular model objects | ➕ store with a content hash and the data revision, splits hash and code commit that produced them |
| Code | git commit | ➕ the repository has no commits yet; tag each release |

### 4.2 Instrument changes

A reworded question is a **new item**, even if the column name is unchanged. Its answers are not comparable, so the old prior and model do not apply. The item registry should store a hash of the question text and options and refuse to score an item whose hash changed.

**Pricing cells the model has not seen.** Today all 440 product × price cells have 95–163 training people. Two parts of the code handle a missing cell differently:
- E0 falls back from the cell to the price level, then to the overall rate (`src/baselines.py`, minimum 20 people per cell).
- The pricing models of E1/E2 and E3 start from a 50% prior instead (an offset of 0 in `src/models.py`; `cell_rate.get(..., 0.5)` in `src/knn.py`).

➕ Before new products or prices are served, both should use E0's fallback chain. Better still, the scope check should refuse the cell until humans have answered it.

### 4.3 Model card per release ➕

A one-page record, kept with the weights. It contains:
- the intended uses and the uses ruled out, from [04_business.md](04_business.md);
- training and calibration data with dates;
- the population frame;
- the metrics from [03_evaluation.md](03_evaluation.md) §6, including subgroups;
- known failure modes (e.g. E6's pricing collapse);
- the differences from the previous release.

---

## 5. Governance and ethics

| Area | Policy |
|---|---|
| **Consent** | Build twins only from people who agreed to their answers being used to model behaviour; record the consent version with the data |
| **Data minimisation** | Persona inputs are survey answers, not identities; no names or contact data in training sets; free-text answers (dropped by the current models anyway) are reviewed before any future use |
| **Deletion** | A deletion request removes the person from the data and from the next prior refresh and retraining. The tabular shift can be retrained the same day. The LoRA shift is retrained at the next release; until then the person's data is in the weights, and that is disclosed in the consent terms. |
| **Use policy** | Enforced in the product (scope check, labels, reference row, restricted individual-level access; [04_business.md](04_business.md) §4), reviewed for every new use case |
| **Fairness** | Subgroup audit at every release; small groups reported with their size; reweighting to the target population when claims are about that population (the panel over-represents graduates, the non-religious and Democrats) |
| **LLM components** | Pin model versions; if a hosted model is used, re-run the evaluation when the provider updates it, since its behaviour can change without notice; compare it with E0 on the same people at each release |
| **No training on the model's own outputs** | Predicted answers are never used as training data or as ground truth for evaluation. Every label comes from a human. A model trained on its own outputs relearns its own errors and hardens them: E6 already answers "do not buy" to almost every pricing item, and training on those answers would make that permanent. Outputs are labelled "predicted" in storage, not only in reports, so they cannot leak into a training set. |
| **Oversight** | A named owner for the model, the calibration panel and the use policy; an incident process for wrong estimates that reached a decision, including a write-up and a regression test |

---

## 6. Operating calendar

| Cadence | Task | Owner |
|---|---|---|
| Every code change | `pytest`; re-score the frozen test split; a row change must have a logged reason | engineer |
| Per scoring batch | input and output monitoring (§2.1) | automated, reviewed weekly |
| Each calibration wave (quarterly to start) | refresh the prior; evaluate vs humans; subgroup audit; monitoring report | model owner |
| When triggered (§3.2) | refit the shift; release gate | model owner |
| Yearly | review the use policy, consent terms, thresholds and the population frame; draw a fresh holdout | owner + governance |

---

## 7. Limits of this plan

- The dataset has two waves about two weeks apart. No threshold or cadence here was validated against real long-run drift. They must be tuned on the first calibration waves.
- The calibration panel is a cost that dominates the model's own cost. If the business cannot fund it, the honest options are to serve only the prior from the latest available survey or to stop serving.
- The release gate uses the same bootstrap and paired comparisons as [03_evaluation.md](03_evaluation.md), so it inherits their limits: one split, no correction for multiple comparisons.
