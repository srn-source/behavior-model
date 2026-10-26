# 4. Business Applications — Validate Digital Twins Before Trusting Them

## Core idea

The value of this work today is **not** a twin that predicts individuals. It is a way to check whether a digital twin adds anything over simple population statistics, before anyone pays for it or trusts it.

**Example.** A vendor says: *"Our AI twin predicts human answers with 73% accuracy."* Is that good? On this dataset:

| Predictor | Score (311 held-out people) | Knows the person? |
|---|---:|---|
| The person's own earlier answer (retest) | 81.8% | yes |
| **E0: the typical answer of training people** | **77.3%** | **no** |
| Best published LLM twin (GPT-4.1) | 72.8% | yes (full persona) |

A rule that knows nobody beats every published twin. 73% sounds high but is **below** the free baseline, so the right question is never "how accurate is it?" but **"how much better than E0 is it?"** The best model built here (E3) adds only +0.78 points.

*E0 learned from other people's answers to the same questions; the zero-shot twins did not. So this compares two regimes, not two architectures. And 77.3% is not a universal bar: E0 must be rebuilt on each client's own data and items.*

## Three use cases

### 1. Benchmark digital-twin vendors *(primary)*

| | |
|---|---|
| **Who** | Market-research agencies, survey-panel companies, product teams buying "synthetic respondents" |
| **Job** | Decide whether to buy or build a twin |
| **Deliverable** | Score, E0 baseline, paired gain over E0 with 95% CI, per-task table, retest reference if available |
| **Boundary** | Needs real human answers from held-out people on the same items |

**Example.** An agency has survey answers from 1,000 customers. It holds out 300, sends the vendor their personas, and gets predictions back.
1. Fit E0 on the other 700 (median answer per question and condition).
2. Score the vendor and E0 on the same 300 people with the same scorer.
3. Result: vendor 75%, E0 77%, gain −2 points with a CI below zero.
4. Decision: don't buy. The twin costs money and loses to a median.

If the vendor scored 79% with a CI above zero, check *which* tasks it wins on before buying. Cost savings were not measured.

### 2. Questionnaire triage

| | |
|---|---|
| **Who** | Survey designers who run the same questionnaire repeatedly |
| **Job** | Find questions where people really differ from the crowd |
| **Deliverable** | Per task: retest score minus E0 score |
| **Boundary** | A small gap does not make a question useless; it may serve other analyses |

**Example** (test split, retest − E0):

| Task | Gap | Reading |
|---|---:|---|
| Allais | +13.5 | A person's own earlier answer says much more than the crowd's |
| Pricing | +11.4 | Real individual differences: worth personal modelling |
| Denominator neglect | +8.7 | Same |
| Framing, Linda, base rate | −0.2 to −0.1 | The crowd's answer predicts as well as the person's own |
| Probability matching | −2.7 | The crowd's answer predicts *better*: answers are noisy |

A researcher studying individual differences should start with Allais and pricing, not framing.

### 3. Fill skipped answers, labelled *(experimental)*

| | |
|---|---|
| **Who** | Panel operators with partly completed questionnaires |
| **Job** | Keep a respondent who skipped a few questions instead of dropping them |
| **Deliverable** | A predicted answer, flagged **predicted**, with an audit trail |
| **Boundary** | Never shown as a real answer; effect on group statistics untested |

**Example.** A respondent answers 99 of 100 questions. E3 fills question 100 from the answers of the 100 most similar training people, and the file marks it `predicted`. E3 is right more often than E0, but only by 0.78 points, so this is a convenience, not new information.

## Guardrails

Limitations say what the model *cannot* do; guardrails are the rules that *stop* misuse.

| Guardrail | Product behavior | Example |
|---|---|---|
| **Scope** | Refuse unsupported populations, questions, products and prices | "Will Thai teenagers buy this?" → *out of scope* (US online adult panel only) |
| **Reference row** | Show E0, CI and sample size next to every model | Never "73% accurate" alone; always "73% vs E0 77%, n = 311" |
| **Labelling** | Mark outputs as predictions | Filled answers carry a `predicted` flag in every export |
| **Consent and privacy** | Authorized data only; never return raw personas | A client gets scores, not other respondents' answers |
| **Human oversight** | Validate with real people before consequential decisions | A launch decision needs a fresh human sample |
| **Fairness** | Report error by subgroup; small groups are "unknown" | A group with 15 test people gets no fairness claim |

## Not supported

| Use | Why not |
|---|---|
| Hiring, credit, insurance, healthcare, eligibility | +0.78 points over a rule that knows nobody is far too little to judge one person |
| Personalized pricing | Charging people differently on a weak individual signal is price discrimination |
| Replacing human samples and reporting the result as a survey | Twins score below the crowd's answer; output is not data about people |
| Speaking for a named person, political microtargeting | No consent, low individual accuracy, manipulation risk |

Stated purchase intent ("would you buy?") is also not observed sales.

**Product position:** a human-supervised benchmark and survey research assistant, not a digital replica of a person.

*Sources: `results/metrics/leaderboard.md`, `leaderboard_by_task.csv`. Vendor and respondent examples are illustrative.*
