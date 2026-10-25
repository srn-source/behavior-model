# 03 · Evaluation strategy

**Deliverable 3.** How to tell whether a behavior model is good: which metric for which question type, what it is compared against, the train/test protocol, how leakage is prevented, and explicit acceptance criteria. Every criterion is applied here to the models already built, and the result is reported, including the ones that fail.

Numbers about this repo come from `results/metrics/` (test split, 311 people, scored against wave 4 unless stated, 95% participant-bootstrap CIs with 2,000 resamples). Model names follow [02_model_plan.md](02_model_plan.md): E0 population median, E1 demographics, E2-P / E2-C / E2-E one persona block, E2-All all blocks, E3 nearest neighbours, E5/E6 Qwen2.5-0.5B before/after fine-tuning, C2 the fine-tuned model with context in the prompt, E4 the paper's published LLM simulations.

---

## 0. Summary

1. **Compare against two references, not one.** The ceiling is the person's own answer from the other wave (human retest, **81.8%**). The trivial baseline is the median answer of training participants to each item (E0, **77.3%**). E0 already reaches 94.4% of retest, so "% of retest" mostly measures E0. The headline is therefore the **share of the E0 → retest gap closed**.
2. **Report the difference from E0, not the raw score.** A model can score well by predicting the population center for everyone. Every model is reported as a paired difference from E0 with a CI, per task, and with a persona-shuffle check that it uses the right person's data.
3. **The trap:** `full_persona` contains the wave-4 answers. A second, quieter version: the response CSV mixes persona columns with earlier answers to the target items. Both are blocked by an item registry and tested.
4. **Result of the criteria today:** four models beat E0 with a CI excluding 0 (E1, E2-P, E2-All, E3), but none reaches the +1 point bar. The best closes 17% of the gap. Every LLM, published or fine-tuned here, is below E0.

---

## 1. The metric, per question type

### 1.1 Primary metric (the paper's, reproduced)

```
cell accuracy   = 1 − |prediction − truth| / (max − min)          on the item's scale
person score    = mean over the 17 tasks of ( mean cell accuracy within the task )
model score     = mean over people of person score
```

- A cell counts only if **both human waves answered it**, so retest and every model are scored on identical cells.
- The implementation (`src/metrics.py`) reproduces the paper: retest **81.73%** vs 81.7%; GPT-4.1-mini text persona **71.71%** vs 71.7%.
- Task-macro weighting means pricing (40 cells) counts as one task, the same as omission (1 cell). This is the paper's choice. It is kept for comparability, and per-task and per-item results are reported next to it.

### 1.2 Which secondary view for which question type

The primary metric treats everything as an ordered scale. Each question type has its own way to go wrong, so results are also read per type:

| Question type | Tasks (scored columns) | Primary metric reduces to | Also report | Why |
|---|---|---|---|---|
| **Binary choice** | Allais (2), probability matching (16), denominator neglect (1), absolute vs. relative savings (2) | exact match | accuracy **per experimental condition** | the conditions have different majority answers; a single average hides one of them |
| **Ordinal scale** (Likert 1–4 … 1–10) | framing, Linda, outcome bias, myside, non-separability, omission, less is more, WTA/WTP, false consensus 1–5 rows | normalized absolute error | mean error in scale points | "0.25" is easier to read as "one point off on a 1–5 scale" |
| **Bounded number** | false consensus % (0–100), base rate (0–100) | normalized absolute error | mean error in percentage points | same |
| **Free number, anchoring** | anchoring estimates (4) | error on deciles 1–10 (thresholds fitted on **training** people) | error per anchor condition (high vs. low) | the raw numbers are unbounded, so deciles keep one outlier from dominating |
| **Free number, fixed range** | sunk cost (0–20, can score below 0) | normalized absolute error | share of answers outside 0–20 | an out-of-range answer scores below 0 and can sink a whole model (prompt A2, [02 §6.3](02_model_plan.md)) |
| **Pricing** (buy / not buy) | 40 products × 11 price levels, randomized to slots | exact match, scored per (product, price), **never per slot** | accuracy by price level; share predicted "buy" vs. real | a model that says "no" to everything still scores 54% (E6) |

### 1.3 Comparison metrics (definitions)

```
paired difference  = mean_people( score_model − score_E0 )                 on the cells both predicted
gap closed         = mean_people( score_model − score_E0 ) / mean_people( score_retest − score_E0 )
persona shuffle    = score with own persona − score with another test person's persona
```

Implementation rules:
- **Paired, per person.** Both models are scored on the same cells of the same people, and the 95% CI comes from 2,000 resamples of people (`src/eval.py: paired_delta`).
- **The gap-closed CI** resamples people jointly for the numerator and the denominator, so the uncertainty of retest is included (`src/metrics.py: bootstrap_ratio_of_means`).
- **Check that personal information is used.** Persona shuffle: each test person is given another test person's persona while keeping their own items, conditions, products and prices (`src/ablation.py`). If the score does not drop, the model is not using the person's information.

Each metric has a sanity test: E0 gives 0% gap closed and retest gives 100% (checked by an automated test).

---

## 2. What to compare against

### 2.1 The reference ladder

| Reference | Score (test, wave 4) | Role |
|---|---:|---|
| Uniform random answer (analytic expectation) | 59.4% | floor; a model near it has not learned the task (E5, untrained 0.5B model: 59.2%) |
| **E0: median of training people, per item and condition; pricing per product × price** | **77.3%** | **the trivial baseline every model must beat** |
| E1: demographics only (regularized regression) | 77.7% | the cheapest personal information; a persona model should beat it, not just E0 |
| **Human retest: the same person's answer from the other wave** | **81.8%** | **the reference ceiling** (Setting B) |

**Why E0 and not random.** Random is 18 points below E0, so beating random says nothing. The published LLM simulations score 68.3–72.8% here: far above random, all below E0.

**Why retest is a reference, not an upper bound:**
- On 4 tasks, E0 scores at or above retest on the test split: probability matching 77.0 vs 74.2, base rate 83.2 vs 83.1, Linda 82.5 vs 82.4, framing 80.6 vs 80.6. There, a person's own earlier answer is noisier than the crowd's.
- Retest also benefits from repeated exposure (people had seen each item before), and it mixes measurement noise with real change over about two weeks.

Beating retest on a task therefore does not show that a model knows the person. The tasks where personalization can show are those with the largest retest − E0 gap: pricing, Allais, denominator neglect, omission, absolute vs. relative.

### 2.2 Controlled vs. cross-regime comparisons

| Regime | Methods | What they saw |
|---|---|---|
| supervised-on-items | E0, E1, E2-x, E3, E6, C2 | training people's answers to the **same** wave-4 items |
| in-context-on-items | A2, B1 (untrained 0.5B model, training people's answers written in the prompt) | the same, but only in the prompt |
| item-zero-shot | E4 (published LLMs), E5 | no one's answers to these items |

Only comparisons **within a regime** are controlled:
- E6 vs E5: the effect of training, +16.65 [+15.70, +17.58].
- E2-x vs E1: the effect of adding a persona block.

"E0 beats the published LLMs by 4.5–5 points" crosses regimes. It shows that the zero-shot LLMs do not recover what the population of training answers contains. It does not show that LLMs are worse models. Every leaderboard row carries its regime.

### 2.3 Two targets

The primary target is the **wave-4 answer** (the assignment's task). The paper scores model accuracy against each person's **wave 1–3 answer** instead (its evaluation code, `mad_accuracy_evaluation.py`, lines 493–495). Both are reported: wave 1–3 for comparison with published numbers, wave 4 for the prediction task. A result that holds on only one target is treated as unstable. Example: E3 closes 17.1% of the gap on wave 4 but 2.9% [−5.3, +10.2] on wave 1–3.

---

## 3. Train / validation / test protocol

| Component | Rule |
|---|---|
| Split | 2,058 people → 1,440 / 307 / 311 train / validation / test, **participant-disjoint**, stratified by age × sex, seed 0, frozen in `data/splits.json` |
| Generalization claim | new people, items seen in training (not new questions) |
| Fitted statistics | everything learned (medians, encoders, scalers, anchoring decile cut points, feature filters, PCA when it was used) is fitted on **training people only** |
| Model selection | hyperparameters (regularization strength, k, epoch) chosen on **validation** with the paper metric |
| Test | scored once per model by one script (`scripts/evaluate.py`), same cells and same people for every method |
| Uncertainty | 2,000 bootstrap resamples **of people**, seed 0; paired differences use the cells both methods predicted |
| Reproducibility | dataset revision pinned (`f883165a`); seeds fixed; predictions saved per model (`results/predictions/`, per person, so not published) and re-scored from disk |

**Known weakness.** Many models were compared on the same 311 test people, so the test set has been looked at many times. The next model generation should freeze a fresh holdout or use repeated cross-validation, with a correction for multiple comparisons. One result already moved with a change of representation: E2-C's wave 1–3 gain lost significance when PCA was removed.

---

## 4. Leakage: the trap and the guards

### 4.1 The trap

Wave 4 asks **no new questions**. It repeats earlier experiments and the pricing study. The dataset's `full_persona` config is, by design, the wave 1–3 persona **with the wave-4 answers** substituted for the repeated items. Where a person's two answers differ (41,242 items), `full_persona` holds the wave-4 answer in **100%** of cases.

The name suggests "everything we know about the person", the dataset card's first example loads it, and nothing fails at run time. A model given `full_persona` sees the answer key and its score rises with no visible error.

**A real case, pid 574** ([01 §1.1](01_data_exploration.md)). Of this person's 63 wave-4 items, 25 were answered differently in the two waves; on all 25, `full_persona` holds the wave-4 answer. Base rate (QID156, "probability this person is an engineer", 0–100): waves 1–3 answer **50**, wave-4 answer **87**, `full_persona` **87**. A model that reads `full_persona` is shown the 87 it is asked to predict.

### 4.2 All leakage paths and how each is blocked

| Path | What happens | Guard | Status |
|---|---|---|---|
| `full_persona` as input | answer key in the prompt | only `wave_split.wave1_3_persona_*` (repeated items removed) is used; `full_persona` is read only by the audit | ✅ audited, tested |
| `wave1_3_response.csv` "all columns" | contains the earlier answers to all 126 target columns, which silently turns the model into retest (Setting B) | item registry: only `role == persona` columns can become features | ✅ tested |
| Pricing slots as items | slots are randomized products; slot-level modelling costs 17.5 points on pricing | per (product, price) everywhere | ✅ tested |
| Statistics fitted on all people | anchoring cut points, scalers, medians would include test people | fitted on training people only | ✅ tested |
| `persona_summary` | its percentiles are computed on all participants, test included | not used as an input; flagged on the E4 rows that use it | ✅ flagged |
| Wave-4 question text in the persona | would leak which items are coming | 0 of 2,058 personas contain a wave-4 QuestionID; 0 exact or near-duplicate question texts | ✅ audited |
| Test answers in training | – | models are fitted only on training people; a test shuffles the test people's answers and checks that their predictions do not change | ✅ tested |
| Per-item training examples (planned for the LLM, [02 §6.2](02_model_plan.md)) | expanding people into one example per item before splitting would put one person's items in train and test | split people first, expand second; a test asserts that no `pid` appears in two splits after expansion | ➕ proposed |
| Retrieval indexes and feature selection | an index of persona answers, or a per-item feature ranking, built on all people would carry validation and test people's answers | built and selected on training people only (E3 already draws neighbours only from training people) | ➕ proposed (✅ for E3) |
| Held-out-question evaluation | the E0 prior and any feature selection would use the answers to the held-out questions | split by QuestionID; hold out all conditions of an experiment together; fit priors and selection without the held-out questions' answers | ➕ proposed (axis not tested yet) |
| **Pre-training contamination** | the dataset and paper are public (2025); a large LLM may know the experiments' typical answers | cannot be audited for closed models; compare against E0, which cannot be contaminated; treat zero-shot LLM results as a reference | ⚠️ residual risk |

**Not leakage:** the persona measures constructs related to the targets (risk aversion and Allais, numeracy and denominator neglect). That is the signal the model is supposed to use. "Zero item overlap" does not mean "independent domains".

Audit: `python -m scripts.audit_leakage` → `results/metrics/leakage_audit.md`.

---

## 5. Acceptance criteria

### 5.1 Criteria set before the results

A model claims individual-level value only if **both** hold (written down before the experiments, plus the +1 point bar set before E1 was run):

| # | Criterion |
|---|---|
| G1 | beats E0: paired difference > 0 with the 95% CI excluding 0 |
| G2 | beats E0 by **≥ +1 point** (practical size) |

### 5.2 Results against the criteria

Test, wave 4:

| Model | G1 beats E0 | G2 ≥ +1 pt | Gap closed | **Pass** |
|---|:---:|:---:|---:|:---:|
| Human retest (reference) | ✅ +4.55 | ✅ | 100% | (ref.) |
| E3 kNN | ✅ +0.78 | ❌ | 17.1% | ❌ |
| E2-All | ✅ +0.67 | ❌ | 14.8% | ❌ |
| E2-P | ✅ +0.50 | ❌ | 11.0% | ❌ |
| E1 | ✅ +0.44 | ❌ | 9.7% | ❌ |
| E2-C | ❌ +0.18 | ❌ | 4.0% | ❌ |
| E2-E | ❌ +0.14 | ❌ | 3.1% | ❌ |
| C2-knn (0.5B fine-tuned, similar people in the prompt) | ❌ −1.17 | ❌ | −25.7% | ❌ |
| E6 (0.5B fine-tuned) | ❌ −1.48 | ❌ | −32.5% | ❌ |
| Best published LLM (GPT-4.1 JSON) | ❌ −4.48 | ❌ | −98.5% | ❌ |
| Main published LLM (GPT-4.1-mini text) | ❌ −4.86 | ❌ | −107.0% | ❌ |

**No model passes.** Four models beat E0, but by +0.4 to +0.8 points; none reaches +1.

**Do the models use the person's persona?** With another test person's persona (same fitted model, own items and prices kept), every tabular model loses 0.9–1.7 points (CIs exclude 0, 5 shuffles each) and falls **below E0**: E2-All 78.0% → 76.3%, E3 78.1% → 77.0% ([persona_shuffle.md](../results/metrics/persona_shuffle.md)). The models' predictions depend on the alignment between persona and respondent, and for the tabular models this dependence is useful but small (+0.4 to +0.8 over E0). The fine-tuned LLM (E6) depends on it much less: it loses only 0.39 [+0.19, +0.57] points (one shuffle), stays below E0, and its pricing answers do not change at all, because it predicts "do not buy" for everyone. A shuffle tests dependence on the right person's data, not behavioural understanding: the swapped persona is also mismatched with the person's items and context.

### 5.3 What the results say about the criteria themselves

1. **G2 is strict for this dataset.** E0 leaves 4.5 points to retest, so +1 point means closing 22% of the gap. The best model closes 17%.
2. **A single split cannot separate close models.** E3 and E2-All differ by +0.10 [−0.26, +0.46]; on the paper's target (waves 1–3) their order flips. Model choice among them should not rest on this split.
3. **Proposed headline for the next generation:** gap closed with its CI, with a target of ≥ 25% and a CI lower bound > 0, on both targets. This is set after seeing the current results (best today 17%), so it is a target for future models, not a re-judgment of these.

---

## 6. Reporting standard for any new model

A new model is evaluated with one command and reported with all of the following:

| Item | Source |
|---|---|
| Score, CI, regime, coverage, Δ vs E0 with CI, **gap closed** with CI (both targets) | `scripts/evaluate.py` → `leaderboard.md` |
| Per-task scores | `leaderboard_by_task.csv` |
| Persona shuffle: does the score drop with another person's persona? | `scripts/run_persona_shuffle.py` |
| Paired comparison against the previous best model in the same regime | `src/eval.py: paired_delta` |
| Statement of which criteria in §5 pass and fail | write-up |

---

## 7. Limitations of this evaluation

- One split, 311 test people, many comparisons. Differences under ~0.5 points are within noise, and no multiple-comparison correction is applied.
- Retest is two waves about two weeks apart. It mixes noise with real change and is inflated by repeated exposure. It is a reference, not a ceiling.
- Stated, not revealed, behaviour: pricing purchases are hypothetical.
- Online-panel sample: over-represents graduates, the non-religious and Democrats. Results hold for "an online US panel like this".
- Task-macro weighting is the paper's choice. Other weightings change the ranking of models within noise.
- E6's persona shuffle uses one seed (a GPU run of about 45 minutes), against five for the tabular models.
- Claims about other papers come from abstracts, not full texts.

## Sources in this repo

| Claim | File |
|---|---|
| Scores, Δ vs E0, gap closed | `results/metrics/leaderboard.md` |
| Per task | `results/metrics/leaderboard_by_task.csv` |
| Persona shuffle | `results/metrics/persona_shuffle.md` |
| Prompting vs retrieval vs fine-tuning | `results/metrics/llm_ablation.md` |
| Metric reproduction | `results/metrics/human_retest_overall.json`, [01_data_exploration.md](01_data_exploration.md) §2 |
| Leakage audit | `results/metrics/leakage_audit.md` |
| Protocol | `configs/experiment.yaml` |
