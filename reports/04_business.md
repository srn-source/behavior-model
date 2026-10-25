# 4. Business Applications — Guide Research, Not Judge People

## What we can responsibly offer now

The strongest application is **evaluating digital-twin products against real human answers**. The work also supports low-stakes survey planning. It does **not** support decisions about individuals or replacing human participants.

**Evidence:** the population baseline E0 scores **77.3%**; the best personalized model **78.1%** (+0.78 points). All 13 published LLM twins score below E0 (68.3–72.8%), although they were trained differently. Before buying a behavioral twin, ask whether it beats what is already predictable from the crowd.

## Three use cases

| Who | Job | Deliverable | Boundary |
|---|---|---|---|
| **Market-research agency or survey-platform buyer** *(primary)* | Decide whether a digital-twin system adds value | Score, E0 baseline, paired CI, retest reference, per-task breakdown | Needs real held-out human answers to the same items |
| **Questionnaire designer** | Find questions where people differ from the crowd | Per-task gap between retest and E0 | A small gap does not mean a question is safe to remove |
| **Panel operator** *(experimental)* | Fill a skipped answer on a known question | An answer clearly labelled **predicted**, with an audit trail | Never shown as an observed answer; effect on later statistics untested |

Per task, the person's own earlier answer beats the population answer by up to **13.5 points** (Allais), **11.4** (pricing) and **8.7** (denominator neglect), and not at all on framing, Linda, base rate and probability matching. That is where personal information can matter.

### Example: should we buy a digital-twin service?

An agency gives a vendor's predictions for 300 consented held-out respondents and their real answers. The evaluation:

1. Fits E0 on **training respondents only**, per question and condition.
2. Scores the vendor and E0 on the same people and items.
3. Reports the paired gain, its 95% CI, per-task differences and a retest reference if repeat answers exist.
4. Recommends more human validation, not purchase, unless the gain is practically meaningful on the tasks the buyer cares about.

**Value:** avoid paying for a complex twin that does not beat the crowd answer. Cost savings were not measured.

## Guardrails

| Guardrail | Product behavior |
|---|---|
| **Consent and privacy** | Use authorized survey data only; never return raw personas |
| **Scope** | Refuse unsupported populations, questions, products and price ranges |
| **Reference row** | Show E0 next to every model, with participant counts and CIs |
| **Labelling** | Mark outputs "model prediction, not an observed human answer" |
| **Human oversight** | Validate with real participants before consequential decisions |
| **Fairness** | Report error by subgroup; too few people means "unknown", not "fair" |

## Not supported

Hiring, credit, insurance, eligibility or other high-impact individual decisions; personalized pricing; predicting or impersonating a named person; political microtargeting; replacing representative human research with generated answers. Stated purchase intent is not observed sales.

**Product position:** a human-supervised benchmark and survey research assistant, not a digital replica of a person.

*Sources: `results/metrics/leaderboard.md`, `leaderboard_by_task.csv`.*
