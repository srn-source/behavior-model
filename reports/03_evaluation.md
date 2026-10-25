# 3. Evaluation Strategy — Beat the Right Baseline

**Core question.** Does a model predict a held-out person's Wave-4 answers better than predicting the usual answer to the same question?

## The evaluation contract

| Decision | Protocol |
|---|---|
| **Task** | Predict Wave 4 from non-target Wave 1–3 persona, the question and its condition |
| **Metric** | The paper's task-balanced score, averaged over 17 tasks, then people |
| **Baseline** | **E0: 77.3%** (train-only median per item/condition; pricing per product × price) |
| **Human reference** | **81.8%** retest (the person's own earlier answer); a reference, not a ceiling |
| **Other references** | Random 59.4%; E1 77.7%; E2-All 78.0%; E3 78.1%; the paper's LLM twins |
| **Split** | 1,440 train / 307 validation / 311 test people; no person in two sets |
| **Uncertainty** | Paired 95% CI from 2,000 participant bootstrap resamples |
| **Acceptance** | **G1:** paired gain over E0 with CI above 0; **G2:** gain ≥ **+1.0 point** |

## Metric by answer type

- **Choice / scale:** `1 - |pred - truth| / (max - min)`; near misses get partial credit.
- **Numbers:** 0–100 and sunk cost on their scale; anchoring estimates in training-fitted deciles.
- **Binary and pricing:** exact match, scored per product × price, never per slot.
- **Aggregation:** average within each task first, so the 40 pricing answers do not dominate.

Every model is reported as a **paired difference from E0**, not only as a score.

## Results

| Model | Test score | Paired change vs E0 [95% CI] | G1 | G2 |
|---|---:|---|:---:|:---:|
| Human retest (reference) | 81.8% | +4.55 | – | – |
| E3, kNN | **78.1%** | **+0.78** [+0.49, +1.06] | ✅ | ❌ |
| E2-All | 78.0% | +0.67 [+0.28, +1.09] | ✅ | ❌ |
| E1, demographics | 77.7% | +0.44 [+0.12, +0.77] | ✅ | ❌ |
| **E0, population median** | **77.3%** | – | – | – |
| C2-knn, fine-tuned LLM + similar people | 76.1% | −1.17 [−1.66, −0.70] | ❌ | ❌ |
| E6, fine-tuned 0.5B LLM | 75.8% | −1.48 [−1.70, −1.26] | ❌ | ❌ |
| Best published LLM twin | 72.8% | −4.48 | ❌ | ❌ |

**No model passes both gates.** Three models beat E0 with a CI above zero, but none by +1 point. The paper's zero-shot twins never saw anyone's Wave-4 answers, so E0 vs twins compares different regimes, not architectures.

### Does the model use the right person's persona?

Give each test person someone else's persona, keeping their own questions, conditions, products and prices. E2-All drops **78.0% → 76.3%**, E3 **78.1% → 77.0%**: both fall below E0. The models do use the right person's data; that data carries little signal.

## Leakage prevention

1. **Never use `full_persona` as input:** it holds the Wave-4 answers (100% match on 41,242 changed answers).
2. **Never take all columns of `wave1_3_response.csv`:** it also holds earlier answers to the target questions.
3. **Role separation:** inputs come only from the registry's persona columns; Wave-4 answers are labels only.
4. **Pricing:** reconstruct product × price, not slot.
5. **Split people before building training rows;** fit medians, deciles, scalers and features on training people only.
6. **New questions:** hold out whole question IDs. Proposed, not run.

Run `python -m scripts.audit_leakage` → `results/metrics/leakage_audit.md`.

## Limits

- **New people, familiar questions only:** says nothing about new questions, populations or time periods.
- **One test set, many comparisons:** differences under ~0.5 points are within noise. Final claims need a fresh holdout.
- **Retest** mixes noise with real change and can score below the population median on some tasks.

**Decision:** keep E0 as the mandatory comparison, use paired uncertainty, and reject complexity that does not show a meaningful gain.

*Sources: `results/metrics/leaderboard.md`, `persona_shuffle.md`, `leakage_audit.md`.*
