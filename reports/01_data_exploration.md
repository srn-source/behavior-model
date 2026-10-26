# 1. Data Exploration — What Can We Predict Reliably?

**Goal.** Given a participant's Wave 1–3 persona, predict their answers to repeated questions in Wave 4. The first priority is a trustworthy dataset, scorer and human reference, not a large model.

## Key findings

| Finding | Evidence | Why it matters |
|---|---|---|
| People are not perfectly consistent with themselves | **81.73%** test–retest score (95% CI 81.42–82.04); **62.8%** of answers identical across waves | Retest is a human reliability reference, not a hard ceiling |
| The population answer is a strong predictor | Train-only population median (E0) scores **77.3%** on held-out people | Every model must beat this simple baseline |
| `full_persona` contains the answers we want to predict | On **41,242** answers that changed between waves, `full_persona` matches Wave 4 **100%** of the time | Using it leaks the answer key |
| Pricing needs product × price reconstruction | Median scored by slot **55.9%**, by product × price **73.4%** | A survey slot is a different product for each person |

## What is in the dataset?

Twin-2K-500 follows **2,058** US participants across **four waves**. Wave 4 repeats earlier questions: behavioral-economics experiments and **40 pricing decisions** per person. Our item registry separates **634 persona columns** (inputs) from **126 Wave-4 target columns** (**122 scored**, in **17 tasks**).

| Source | Contains | Safe use |
|---|---|---|
| `wave_split` | Wave 1–3 persona, Wave-4 questions, Wave-4 answers, stored separately | Persona as input; Wave-4 answers as labels only |
| `full_persona` | Persona with the **Wave-4 answers** substituted into repeated items | **Never** an input |

A second trap: `wave1_3_response.csv` mixes persona columns with each person's **earlier answers to the target questions**. Copying those is the retest task, not persona-based prediction. The item registry and leakage audit (`scripts/audit_leakage.py`) enforce the separation.

## How are answers scored?

The paper's metric gives partial credit for near misses:

```
item_score   = 1 - |prediction - truth| / (scale_max - scale_min)
task_score   = mean(item_score within a task)
person_score = mean(task_score over 17 tasks)
model_score  = mean(person_score over participants)
```

Binary answers reduce to exact match; anchoring estimates are scored in deciles. Our scorer reproduces the paper: **81.73%** retest (paper 81.7%) and **71.71%** for the GPT-4.1-mini text-persona twin (paper 71.7%).

![Retest by task](figures/retest_by_task.png)

*How to read: per task, the blue bar is a person's own earlier answer used to predict their wave-4 answer (retest, with 95% CI); the orange diamond predicts everyone with the population median; the grey tick is a random guess. All 2,058 people, exploratory.*

The gap between diamond and bar end is what personal information could add: large for Allais (+11.1 points) and pricing (+10.4). On some tasks (probability matching, framing, Linda) the diamond is past the bar end: the population median beats a person's own earlier answer, because within-person noise is larger than between-person differences there.

## Pricing

`QID9_1..40` are randomized slots. Each person sees each of 40 products once, at one of 11 price levels (0×, 0.2×, …, 2.0× the product's reference price). Predictions must be made per product × price.

![Demand curve](figures/demand_curve.png)

*How to read: the x-axis is the price shown ÷ the product's median price (0 = free, 1 = usual price, 2 = double). Left: share who would buy, per wave. Right: share of people giving the same buy / not-buy answer in both waves (the y-axis starts at 0.76).*

About 80% would take a product for free and 17–18% at twice the price; the two waves overlap, so the crowd's demand is stable. Individuals are least consistent near the usual price (81% same answer vs 86–88% at the extremes), where the decision is closest.

## Limits

- **Long personas:** the text persona is about **27.5k tokens** (median); the compact Q/A form used for fine-tuning is about **16.9k**. Input format is a real constraint for small models.
- **Sample:** an online US panel that over-represents college graduates, the non-religious and Democrats, and under-represents Hispanic adults (9.4% vs about 17% in the US), people 65+ (13.5% vs about 21%) and people without a high-school diploma. Results hold for "a panel like this".
- **Retest is a reference, not a ceiling:** it mixes noise with real change between the earlier wave and wave 4 (launched two weeks after wave 3; waves 1–3 were a week apart).

**Decision:** use only leakage-safe Wave 1–3 persona information, score on Wave-4 answers with the paper's task-balanced metric, and require every model to beat a train-only population baseline.

*Sources: `results/metrics/human_retest_overall.json`, `leakage_audit.md`, `eda_task_headroom.csv`, `eda_demand_curve.csv`, `token_counts.json`. Dataset pinned to revision `f883165a`.*
