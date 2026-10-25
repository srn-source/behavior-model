# 04 · Business applications and guardrails

**Deliverable 4.** Who could use this work, for which job, and what it must not be used for. Every "can" and "cannot" below is tied to a measured accuracy result. Where a use is plausible but untested, it says so.

Numbers come from `results/metrics/` (test split, 311 people, wave-4 answers as truth): `leaderboard.md` for scores and paired differences, `leaderboard_by_task.csv` for per-task scores. Models are defined in [02_model_plan.md](02_model_plan.md); criteria in [03_evaluation.md](03_evaluation.md) §5.

---

## 0. Summary

**Positioning.** The evidence supports this work as **a benchmark and a baseline for digital-twin claims**, and as a low-stakes helper for survey work. It does not support a system that predicts the decisions of specific people. *We use the model to support decisions about what to research next, not to make decisions about individual people.*

**What the evidence supports today:**
- **A baseline that any twin must beat.** The population answer (E0) scores 77.3%, 94% of human retest (81.8%). All 13 published LLM twins score below it (68.3–72.8%).
- **A map of where people differ from the crowd.** Per task, the gap between a person's own earlier answer and the population answer ranges from 13.5 points (Allais) to none (framing, Linda, base rate, probability matching).
- One primary use (benchmarking twin vendors and models) and two secondary ones (§2).

**What it does not support:** decisions about an individual person. The best model adds +0.78 points over E0 and closes only 17% of the gap to retest.

**The most important guardrail:** a zero-shot LLM twin is less accurate than simply answering with the population median. The paper's main twin is below E0 on 14 of 17 tasks, by 12.3 points on anchoring estimates and 11.6 on Allais.

---

## 1. What the model can and cannot do (measured)

| Capability | Evidence (test) | Usable? |
|---|---|---|
| Tell whether a twin beats the population answer | E0 77.3% [76.5, 78.1]; paired person-level CIs for any model (`src/eval.py`) | ✅ |
| Say where personal information can matter | retest − E0 per task: Allais +13.5, pricing +11.4, denominator neglect +8.7; framing, Linda, base rate, probability matching ≤ 0 | ✅ for these 17 tasks |
| Fill a skipped answer for a known panelist on a trained item | best model 78.1% vs E0 77.3% (+0.78 [+0.49, +1.06]) | ⚠️ low-stakes only, labelled |
| One person's answer to one question, as a basis for a decision | +0.4 to +0.8 points over the population median; ≤ 17% of the gap closed | ❌ |
| New questions, new products, new populations, later years | not tested (one panel, two waves about two weeks apart, items seen in training) | ❌ until validated |

---

## 2. Use cases

### 2.1 Primary: benchmarking digital-twin vendors and models

| | |
|---|---|
| **Who** | Market-research agencies, panel companies and product teams evaluating a "digital twin" or "synthetic respondent" product |
| **Job** | Before buying or building a twin, measure whether it beats the population answer on the buyer's own items, and by how much |
| **Output** | Score, paired difference from E0 with a CI, share of the E0 → retest gap closed, per-task table |
| **Evidence** | The 13 published twins look strong as "% of retest" (83–89%) but are all below E0 here; E0 alone earns 94% of retest. The ladder (random → E0 → retest) and the scorer are in this repo and reproduce the paper's numbers. |
| **Limit** | Needs the buyer's own human answers to the same items for E0 and, ideally, a retest wave for the ceiling |

**Workflow.**

| Step | What happens | Output |
|---|---|---|
| 1. Input | the vendor's predictions and real answers for a held-out set of people | – |
| 2. Baselines | E0 fitted on the buyer's training people; retest if a repeat wave exists | the reference ladder |
| 3. Score | the same scorer for every row, paired by person | Δ vs E0 with CI, gap closed, per task |
| 4. Decision | buy or build only if the twin beats E0 with a CI excluding 0 on the tasks that matter | a documented go / no-go |

### 2.2 Secondary: questionnaire triage

| | |
|---|---|
| **Who** | Survey designers running the same instrument repeatedly |
| **Job** | Decide which items carry person-specific information and which are answered like the crowd |
| **Evidence** | On framing, Linda, base rate and probability matching the population answer is as accurate as the person's own earlier answer (retest − E0 between −2.7 and −0.1 points). On Allais, pricing and denominator neglect the person's earlier answer is 8.7–13.5 points better. |
| **Limit** | Measured on this panel and these 17 tasks only. Dropping an item from a survey also drops its use for other analyses; this is an input to the decision, not the decision. |

### 2.3 Secondary: filling skipped answers, labelled

| | |
|---|---|
| **Who** | Panel operators with partially completed questionnaires |
| **Job** | Fill a skipped trained item with a predicted answer so a record is not discarded |
| **Evidence** | E3 and E2-All are 0.67–0.78 points more accurate than E0 per answer; E0 itself is 77.3% |
| **Limit** | The filled value must be flagged as predicted. How filling changes later group statistics was not measured. |

---

## 3. What it must not be used for

| Do not use it to | Why (evidence or principle) |
|---|---|
| **Decide anything about a specific person** (credit, insurance, hiring, healthcare, legal, eligibility, price paid) | Individual accuracy is barely above the population median (≤ 17% of the gap closed); a wrong answer has real cost for that person; the training panel was not collected for such uses |
| **Personalised pricing** (charge different people different prices) | On pricing, the best model gains about 1 point over the population answer (73.6% vs 72.6%). The ethical and legal risks (price discrimination) fall on people who did not consent. |
| **Replace human participants in research and report the result as a finding about people** | No model here comes close to a person's own answer; the generalisation claim covers trained items only |
| **Answer survey questions with a zero-shot LLM twin** | The main published twin is below E0 on 14 of 17 tasks (anchoring −12.3, Allais −11.6, sunk cost −9.4 points), equal on 2, and above only on "less is more" (+0.6) |
| **Predict new questions, new products, new countries or later years** | Not tested; the generalisation claim is "new people, trained items, the same panel" |
| **Speak for a named real person** ("what would customer X say") | Low individual accuracy, and it creates a persona of a real person without their consent |
| **Political persuasion or microtargeting** | The dataset includes political attitudes; using a twin to tailor persuasive messages to individuals is manipulation, and individual predictions are weak anyway |
| **Present predicted output without a label** | Readers will treat it as data about people |

---

## 4. Guardrails

### 4.1 Built into the product

| Guardrail | How it works |
|---|---|
| **Scope check** | Refuse items, products or price levels outside the trained set, and populations outside the panel's frame (US online adults). Return "out of scope", not a guess. |
| **Uncertainty with every number** | Report bootstrap intervals and the number of people. |
| **Reference row** | Every report shows E0 (the population answer) next to the model, so users see how much the persona actually changed the prediction. |
| **Label** | "Predicted: model output, not survey data" on every chart and export. |
| **No raw persona out** | Outputs are scores or flagged predictions; the persona data of training panelists is never returned. |

### 4.2 Process

| Guardrail | How it works |
|---|---|
| **Validation before a decision** | Any number that drives a business decision is checked against a fresh human sample of the same population (the size set by the decision's cost) |
| **Restricted individual-level access** | Individual predictions are for research, model evaluation and flagged filling only; access is logged and reviewed against §3 |
| **Use review** | New use cases are approved against §3 by a named owner, with the evidence that supports them |
| **Consent and data rights** | Twins are built only from panelists who consented to modelling; deletion requests are honoured by retraining (the procedure is in [05_maintenance.md](05_maintenance.md)) |
| **Subgroup audit per release** | Accuracy by sex, age, education and race; a group more than 3 points below overall blocks release until explained. Several groups have fewer than 20 test people, so a larger sample is needed for this check to be reliable. |

---

## 5. Limits of this assessment

- One online US panel (over-represents graduates, the non-religious and Democrats), two waves about two weeks apart, stated rather than revealed purchases.
- No use case was piloted with a real business decision; §2 says what the evidence supports, not that the use pays off.
- The effect of filled answers on later analyses (§2.3) was not measured.
- Claims about other work come from abstracts, not full texts.

## Sources in this repo

| Claim | File |
|---|---|
| Accuracy, paired difference from E0, gap closed | `results/metrics/leaderboard.md` |
| Per-task scores (retest, E0, models, LLM twins) | `results/metrics/leaderboard_by_task.csv` |
| Persona shuffle | `results/metrics/persona_shuffle.md` |
| Sample composition | [01_data_exploration.md](01_data_exploration.md) §9 |
