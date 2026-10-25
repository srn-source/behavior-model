# 5. Long-run Maintenance — Revalidate as Behavior Changes

**Principle:** behavior, prices, survey questions and panel make-up change. A model is trustworthy only while it keeps agreeing with **fresh, consented human answers**. Twin-2K-500 has two waves about two weeks apart, so everything below is a **proposed operating plan**, not measured drift.

## What to monitor

| What changes? | Check | When |
|---|---|---|
| **Who answers, what is asked** | Demographic mix, missing values, unseen questions, product and price coverage | Each data batch |
| **Prediction quality** | Score and **paired gain over a refreshed E0**, per-task error | Each new human wave |
| **Use of the person's data** | Persona-shuffle drop | Each new human wave |
| **Subgroups** | Error by demographic group, with group sizes | Each release |
| **Pipeline** | Leakage audit, item registry, pricing mapping, re-score of the frozen test set | Every code or data change |

**Why gain over a refreshed E0?** The current personal gain is under one point. A model can keep a good total score while its personal information stops adding anything; only the paired gain over a fresh E0 shows that.

## Drift and retraining loop

```
New human answers
      ↓
Validate data, population and question definitions
      ↓
Refresh E0 (cheap: medians and buy rates)
      ↓
Compare the personalized model with the refreshed E0
      ↓
Diagnose: changed behavior, changed population, or pipeline bug?
      ↓
Refit the personal part only if justified
      ↓
Fresh holdout + subgroup review + shadow run
      ↓
Release a versioned model with a rollback target
```

**Triggers to investigate** (tune after the first few waves):
- The paired gain over a refreshed E0 has a CI that includes zero.
- A task falls below E0 with a CI excluding zero.
- People, question wording, products or prices move outside the validated range.
- Any leakage, broken mapping, consent problem or reproducibility failure: **block the release**.

An alert starts a diagnosis, not automatic retraining. Never train on model-generated answers presented as human answers.

## Release criteria and versioning

A release must: pass data, consent and leakage checks; beat the refreshed E0 on the tasks it claims; show no unacceptable regression on the frozen benchmark or subgroups; pass a shadow run; and have a rollback.

Version together: dataset revision, participant splits, question IDs **and wording**, item registry, model weights, training configuration, code commit and evaluation output. A reworded question is a new item.

This repository already pins the dataset revision (`f883165a`) and records the splits, training configuration and logs, and evaluation outputs (`results/metrics/`). Model weights, question hashes and release approvals would be added for production.

## Governance

Use consented data for defined purposes; restrict individual-level access; support deletion (remove from future training and rebuild affected models). Each release has a named owner, and new uses are reviewed against [04_business.md](04_business.md).

## Not yet verified

Long-run drift, monitoring thresholds, refresh cadence and costs need real operation to set. The current test set has been used for many comparisons, so the next model generation needs a fresh, time-based holdout.

**Decision:** keep the population prior easy to refresh, track the personal gain over it, and release only after human-grounded evaluation.
