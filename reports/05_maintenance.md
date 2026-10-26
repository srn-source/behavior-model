# 5. Long-run Maintenance — Revalidate as Behavior Changes

**Principle:** behavior, prices, survey questions and panel make-up change. A model is trustworthy only while it keeps agreeing with **fresh, consented human answers**. Twin-2K-500 covers about one month (waves 1–3 a week apart, wave 4 two weeks after wave 3), so everything below is a **proposed operating plan**, not measured drift.

**Example.** A model learns in 2026 that about 40% of people buy a product at its usual price. In 2027 prices rise and fewer people buy; in 2028 new products appear that it has never seen. The model keeps answering like 2026. A good score today says nothing about next year, so maintenance means checking predictions against **new human answers**, not just keeping a server up.

## What to monitor

| What changes? | Check | When | Example alert |
|---|---|---|---|
| **Who answers, what is asked** | Demographic mix, missing values, unseen questions, product and price coverage | Each data batch | A new batch is 40% aged 65+ (13.5% in this panel), or adds a product never seen |
| **Prediction quality** | Score and **paired gain over a refreshed E0**, per-task error | Each new human wave | E3's gain over E0 falls from +0.78 to +0.1 with a CI crossing 0 |
| **Use of the person's data** | Persona-shuffle drop | Each new human wave | Today E3 drops 1.1 points with a wrong persona (78.1 → 77.0); near 0 means the persona stopped mattering |
| **Subgroups** | Error by demographic group, with group sizes | Each release | One age group falls 3 points below the rest |
| **Pipeline** | Leakage audit, item registry, pricing mapping, re-score of the frozen test set | Every code or data change | Wave-4 answers leak into the input, or pricing is mapped by slot again (pricing E0 73.4% → 55.9%) |

### Why compare with a refreshed E0, not just watch the score?

| | Today (measured) | Next year (hypothetical) |
|---|---:|---|
| E3, personalized | 78.1% | 76.0% |
| E0 refit on the new wave | 77.3% | 77.0% |
| **E3 − E0** | **+0.78** | **−1.0** |

76% alone looks acceptable. But a rule that knows nobody now does better, so the personal model has stopped earning its complexity. Only the paired gain over a fresh E0 shows that.

The persona shuffle is a sensitivity check (does the model use the *right* person's data?), not proof that it understands personality.

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

### Triggers to investigate

Thresholds are starting points, to be tuned after the first few waves.

| Trigger | Meaning | Response |
|---|---|---|
| Gain over refreshed E0 has a CI that includes 0 | No clear evidence the personal model still helps | Check whether the persona still adds value |
| A task falls below E0 with a CI excluding 0 | The model is worse than the baseline there | Per-task error analysis |
| New questions, wording, products or prices | Outside what was validated | Check coverage; collect human answers first |
| Leakage, broken mapping, consent or reproducibility failure | Results or data rights are not trustworthy | **Block the release** |

**What "CI includes 0" means.** Today E3 − E0 = +0.78, 95% CI [+0.49, +1.06]: the whole interval is above 0, so E3 really is better. If a new wave gave +0.4 with CI [−0.2, +1.0], the gain could be zero, so stop claiming it. Even a CI above 0 is not automatically worth the cost: the gain must also be large enough to matter (report 03 asks for +1 point).

### An alert starts a diagnosis, not automatic retraining

Say E3 used to score 78% and a new wave gives 70%. Three possible causes need three different fixes:

| Cause | Sign | What to do |
|---|---|---|
| **Behavior drift** | The whole demand curve shifts, e.g. buying at the usual price falls from 40% to 30% | Refresh E0, refit the personal part, revalidate |
| **Population shift** | Error jumps for one group, e.g. far more 65+ respondents than before | Fix sampling or reweight the evaluation; retraining may not be needed |
| **Pipeline bug** | Every task moves at once, right after a code or data change | Fix the code; the model may be fine |

Never train on model-generated answers presented as human answers.

## Release criteria

A release must: pass data, consent and leakage checks; beat the refreshed E0 on the tasks it claims; show no unacceptable regression on the frozen benchmark or subgroups; pass a shadow run; and have a rollback.

**Shadow run, example.** The current E3 (*champion*) stays in use. The retrained E3 (*challenger*) gets the same authorized inputs and predicts in parallel, but its answers are not used. When the real human answers arrive, both are scored; the challenger is promoted only if it wins. If problems appear after release, **roll back** to the champion.

## Versioning

Version together: dataset revision, participant splits, question IDs **and wording**, item registry, model weights, training configuration, code commit and evaluation output.

**Example.** Six months later someone asks: *"Why does v2 score 79% and v1 78%?"* With everything versioned you can check whether the model improved or the questions and test people changed. A reworded question is a new item: "Would you buy it at $4.70?" and "Would you buy it today at $4.70?" can get different answers, so the second must not reuse the first's E0 or training rows.

## Governance

Use consented data for defined purposes; restrict individual-level access; each release has a named owner; new uses are reviewed against [04_business.md](04_business.md).

**Example.** A respondent withdraws consent → delete their rows, refit E0 and the personal models without them, and record the new version.

## Done vs proposed

| Already in this repository | Proposed for production, not done |
|---|---|
| Pinned dataset revision (`f883165a`) | Monitoring over months of real use |
| Frozen participant split | A refresh cadence and alert thresholds |
| E0 vs personalized comparisons with paired CIs | Shadow runs and rollback in production |
| Leakage audit and scoring checks | Release-approval and deletion workflows |
| Training configuration, logs and evaluation outputs (`results/metrics/`) | Versioned model weights and question-wording hashes |

The current test set has been used for many comparisons, so the next model generation needs a fresh, time-based holdout.

*Examples marked hypothetical are illustrative; measured numbers come from `results/metrics/leaderboard.md`, `persona_shuffle.md`, `eda_demand_curve.csv` and `eda_population_baselines_in_sample.csv`.*

**Decision:** refresh the baseline first, retrain the personal part only when new human evidence justifies it, and release only after validation against fresh human answers.
