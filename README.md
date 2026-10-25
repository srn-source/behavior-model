# Large Behavior Model on Twin-2K-500

**Task.** Given a person's answers from survey waves 1–3 (their persona), predict how the same person answers the wave-4 questions. Evaluate against the real wave-4 answers and against human test–retest. Dataset: Twin-2K-500 (2,058 US adults, four waves).

**Short answer.** On this dataset, average accuracy mostly rewards predicting each question's population answer.
- The median answer of training people, with no personal information, scores **77.3%**. Human retest scores **81.8%**.
- Every person-level model built here lands at 77.4–78.1%. The best closes **17%** of the gap between the two.
- The paper's zero-shot LLM twins score below the population median.
- A fine-tuned 0.5B model beats those twins but not the median.
- On that same 0.5B model, prompting and retrieval without training stay at chance (56.6–59.4%). Fine-tuning is what makes the model use the context it is given: fine-tuned with the answers of 20 similar people in the prompt, outside pricing it is not separable from the median. On pricing it still fails, apparently because it answers one item after another and copies its own earlier answers.

---

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Data exploration | [reports/01_data_exploration.md](reports/01_data_exploration.md) · figures in `reports/figures/` · code: `scripts/eda.py`, `scripts/audit_leakage.py`, `scripts/reproduce_retest.py` |
| 2 | Plan to build the model | [reports/02_model_plan.md](reports/02_model_plan.md) |
| 3 | Evaluation strategy | [reports/03_evaluation.md](reports/03_evaluation.md) |
| 4 | Business applications and guardrails | [reports/04_business.md](reports/04_business.md) |
| 5 | Long-run maintenance | [reports/05_maintenance.md](reports/05_maintenance.md) |
| 6 | Bonus: fine-tune a < 0.5B model | Qwen2.5-0.5B + LoRA: code in `scripts/ft_*.py`, `src/ft_format.py`; design and results in [reports/02_model_plan.md](reports/02_model_plan.md) §6.2–6.3 (prompting vs retrieval vs fine-tuning: [results/metrics/llm_ablation.md](results/metrics/llm_ablation.md)); run instructions [below](#bonus-fine-tune-a-05b-model-gpu) |

The numbers in the reports come from files in [results/metrics/](results/metrics/), written by the scripts below.

---

## Results (test split, 311 held-out people, wave-4 answers as truth)

| Method | Regime | Score [95% CI] | Δ vs E0 [95% CI] | E0 → retest gap closed |
|---|---|---|---|---:|
| Human retest: the person's own answer from the other wave | reference | 81.8% [81.0, 82.6] | +4.55 [+3.69, +5.39] | 100% |
| **E3** k nearest neighbours on raw persona (k = 100) | supervised | **78.1%** [77.3, 78.8] | +0.78 [+0.49, +1.06] | **17.1%** |
| **E2-All** demographics + all persona answers, regularized regression | supervised | 78.0% [77.2, 78.7] | +0.67 [+0.28, +1.09] | 14.8% |
| E2-P demographics + Personality block | supervised | 77.8% [77.0, 78.5] | +0.50 [+0.18, +0.82] | 11.0% |
| E1 demographics only | supervised | 77.7% [77.0, 78.5] | +0.44 [+0.12, +0.77] | 9.7% |
| **E0** median of training people, per item and condition (pricing per product × price) | supervised | **77.3%** [76.5, 78.1] | – | 0% |
| C2-knn Qwen2.5-0.5B + LoRA, answers of 20 similar training people in the prompt | supervised | 76.1% [75.3, 76.9] | −1.17 [−1.66, −0.70] | −25.7% |
| E6 Qwen2.5-0.5B-Instruct + LoRA, trained on training people | supervised | 75.8% [75.0, 76.6] | −1.48 [−1.70, −1.26] | −32.5% |
| C2-prior Qwen2.5-0.5B + LoRA, training people's typical answer in the prompt | supervised | 75.5% [74.7, 76.3] | −1.82 [−2.05, −1.60] | −40.1% |
| Best published LLM twin (GPT-4.1, JSON persona) | zero-shot | 72.8% [72.0, 73.5] | −4.48 | −98.5% |
| Main published LLM twin (GPT-4.1-mini, text persona) | zero-shot | 72.4% [71.6, 73.2] | −4.86 [−5.61, −4.18] | −107.0% |
| E5 the same 0.5B model without training | zero-shot | 59.2% [58.4, 60.0] | −18.13 | – |
| A0 / A2 / B1 the same untrained model: questions only / typical answer in the prompt / 20 similar people in the prompt | zero-shot / in-context | 58.4% / 59.4% / 56.6% | −18.86 / −17.87 / −20.74 | – |
| Uniform random answer | reference | 59.4% | | |

Notes:
- The metric is the paper's: `1 − |prediction − truth| / range`, averaged within each of 17 tasks, then over tasks and people. Our scorer reproduces the paper's retest (81.73%) and its main LLM result (71.71%).
- "Supervised" methods learned from other people's answers to the same items. The published twins did not. So E0 vs the published twins is a comparison across regimes, not a controlled experiment. E6 vs E5 (+16.65 points) is controlled.
- Full table, both targets and validation: [results/metrics/leaderboard.md](results/metrics/leaderboard.md).

**Thesis.** On this dataset the population answer is a strong floor: it earns 94% of the retest score without knowing anyone. Every model, tabular or LLM, has to be judged by how much of the remaining gap to retest it closes.

**Part 1 · Population vs. personalization (E0–E3; the published LLM twins as reference).**
- Personal information adds +0.4 to +0.8 points over the population median and closes at most 17% of the gap to retest.
- It is used, not ignored. Given another test person's persona (same fitted model, own items and prices kept), every tabular model loses 0.9–1.7 points and falls *below* E0 ([persona_shuffle.md](results/metrics/persona_shuffle.md)). The predictions depend on the right person's data, but the benefit is small.

**Part 2 · Can a small language model learn behavior? (E5–E6, the bonus).**
- Supervised fine-tuning moves Qwen2.5-0.5B from chance (59.2%) to 75.8% (+16.65 [+15.70, +17.58], controlled).
- That is above the larger zero-shot LLM twins, although model size and supervised adaptation were not independently controlled.
- It stays below E0 (−1.48 [−1.70, −1.26]). Pricing accounts for −1.08 of that gap: the model predicts "do not buy" for almost every product, even free ones. The remaining −0.40 [−0.59, −0.21] comes mostly from anchoring estimates.
- **Prompting vs retrieval vs fine-tuning, same model** ([llm_ablation.md](results/metrics/llm_ablation.md)). Untrained, no prompt helps: questions only 58.4%, persona 59.2%, typical answer of training people 59.4%, answers of 20 similar people 56.6%. The model does not copy the hints it is given. Fine-tuned with the same contexts:
  - similar people beat the typical answer, +0.65 [+0.20, +1.10];
  - the best (C2-knn, 76.1%) is not separable from E6 (+0.31 [−0.16, +0.76]);
  - all three lose the same −1.08 on pricing. Outside pricing, C2-knn is not separable from E0 (−0.09 [−0.56, +0.37]).

  So the remaining bottleneck is how the model answers, not what it is told.
- Its predictions depend on the persona only a little. With another person's persona its score drops 0.39 points [+0.19, +0.57] (one shuffle), against 0.9–1.7 for the tabular models. Fine-tuning taught it the typical answer, not how to benefit from the persona.

These findings replicate, on this dataset and with supervised baselines, results reported for LLM survey respondents elsewhere. This repo adds paired comparisons against E0 on one held-out split, and a controlled comparison of prompting, retrieval and fine-tuning on the same 0.5B model.

## The leakage trap

Wave 4 asks no new questions: it repeats earlier ones.
- The dataset's `full_persona` config is, by design, the wave 1–3 persona **with the wave-4 answers** to the repeated questions. Where a person's two answers differ (41,242 items), it holds the wave-4 answer in 100% of cases. Using it as input puts the answer key in the prompt, with no error. This repo uses only `wave_split`, which separates persona, earlier answers and wave-4 answers.
- A quieter version: `wave1_3_response.csv` contains the earlier answers to every target item, so "use all columns" turns persona-only prediction into copying the previous answer. Model inputs come only from item-registry columns with `role == "persona"`.
- Pricing slots `QID9_1..40` are randomized products. Treating a slot as a product costs 17.5 points on pricing.

All three are audited (`scripts/audit_leakage.py` → [leakage_audit.md](results/metrics/leakage_audit.md)) and enforced by tests.

---

## Run it

Python 3.11, CPU only, ~680 MB download. Everything below reproduces every number in the reports. Splits, bootstrap and models use fixed seeds.

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
.venv/bin/python -m scripts.download_data       # dataset at pinned revision f883165a
.venv/bin/python -m scripts.audit_leakage       # results/metrics/leakage_audit.{json,md}
.venv/bin/python -m scripts.reproduce_retest    # stops if the paper's retest/LLM numbers are not reproduced
.venv/bin/python -m scripts.token_counts        # persona lengths (Qwen2.5-0.5B tokenizer)
.venv/bin/python -m scripts.eda                 # tables and the 3 figures used in report 01
.venv/bin/python -m scripts.run_e0              # E0 population median
.venv/bin/python -m scripts.run_e4              # published LLM simulations, re-keyed to our split
.venv/bin/python -m scripts.run_e1_e2           # E1, E2-P/C/E, E2-All; regularization chosen on validation (~30 min)
.venv/bin/python -m scripts.run_e3              # E3 kNN, k chosen on validation (~1 min)
.venv/bin/python -m scripts.evaluate            # leaderboard (score, Δ vs E0, gap closed) + figure
.venv/bin/python -m scripts.run_e0_cv           # E0 on all 2,058 people vs the paper's own numbers
.venv/bin/python -m scripts.run_persona_shuffle # do the models use the right person's persona? (~4 min)
```

Optional:
- `scripts.plot_bootstrap_example` draws a teaching figure of the paired bootstrap.

### Bonus: fine-tune a < 0.5B model (GPU)

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, 494,032,768 parameters (357,898,112 without the tied embeddings), Apache-2.0. The LoRA adapter is merged into the base weights before prediction, so the evaluated model has 494,032,768 parameters (the merge adds none; `results/metrics/e6_run_config.json` records 494,032,768). |
| Example | one person: wave 1–3 persona as compact Q/A text (from `wave_split`, never `full_persona`), then the wave-4 questions that person saw, then their answers, one line per item (~24k tokens) |
| Training | LoRA r = 16, α = 32 on all linear layers (8.8M trainable); loss on answer lines only; lr 2e-4, cosine; effective batch 8 (1 example × 8 gradient accumulation); max length 26,624 tokens; 2 epochs; training people only |
| Evaluation | test people scored once by the same scorer as everything else (E6 vs E5 vs E0 above) |
| Run record | [results/metrics/e6_run_config.json](results/metrics/e6_run_config.json), [e6_train_log.csv](results/metrics/e6_train_log.csv), [e6_train_manifest.json](results/metrics/e6_train_manifest.json) |

GPU environment (separate from `.venv`, versions in `requirements-ft.txt`). The full run used one 80 GB H100: training 58.5 min, prediction ~45 min per model.

```bash
python -m scripts.ft_build_data --tokenizer Qwen/Qwen2.5-0.5B-Instruct    # data/ft/{train,val}.jsonl (~2 min)
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_train                           # outputs/qwen25_05b_lora
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5_qwen05_zeroshot
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E6_qwen05_lora --adapter outputs/qwen25_05b_lora
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E6_qwen05_lora --adapter outputs/qwen25_05b_lora --persona-shuffle-seed 0   # ablation
.venv/bin/python -m scripts.evaluate                                        # E5/E6 join the leaderboard
.venv/bin/python -m scripts.run_persona_shuffle                             # scores the E6 shuffle too
```

Prompting vs retrieval vs fine-tuning (the prompt variant is chosen on `--split val` first, then the winner is run once on `--split test`):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5a_qwen05_none --prompt none --split test                 # A0
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5b_qwen05_prior_median --prompt prior_median --split test # A2
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5c_qwen05_knn_k20 --prompt knn --k 20 --split test        # B1
python -m scripts.ft_build_data --out data/ft_knn20 --prompt knn --k 20 --tokenizer Qwen/Qwen2.5-0.5B-Instruct       # C2-knn
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_train --train data/ft_knn20/train.jsonl --val data/ft_knn20/val.jsonl --out outputs/qwen25_05b_lora_knn20
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E6k_qwen05_lora_knn20 --adapter outputs/qwen25_05b_lora_knn20 --prompt knn --k 20 --split test
.venv/bin/python -m scripts.evaluate
.venv/bin/python -m scripts.analyze_llm_ablation                            # comparisons, pricing decomposition, figure
```

C2-prior is the same with `--prompt prior_median`, `data/ft_prior`, `outputs/qwen25_05b_lora_prior` and `--exp E6p_qwen05_lora_prior`.

**Smoke test** (minutes; for a smaller GPU such as a free Colab T4, which I have not tested):

```bash
python -m scripts.ft_build_data --limit 16 --out /tmp/ft
python -m scripts.ft_train --train /tmp/ft/train.jsonl --val /tmp/ft/val.jsonl --limit 8 --max-steps 4 --out /tmp/ft_out
python -m scripts.ft_predict --exp E5_qwen05_zeroshot --limit 3
```

At 24k tokens per example, a full run needs about 80 GB of GPU memory, even with gradient checkpointing (FlashAttention-2 if installed). [02 §5](reports/02_model_plan.md) proposes per-item retrieval of persona answers to cut that by an order of magnitude.

**What the bonus shows:**
- Training moves the model from chance (59.2%) to 75.8%, above every published GPT-4.1 twin.
- It stays below E0. On pricing it says "buy" almost never: it appears to copy its own earlier "no" answers (exposure bias; consistent with the evidence, not isolated).
- Prompting or retrieval cannot replace training at this size. Retrieval helps only once the model is trained, and pricing fails the same way whatever context is in the prompt.
- A fix is proposed in [02 §6.2](reports/02_model_plan.md).

---

## Repository layout

```text
reports/                 deliverables 1-5 (English) and figures
src/
  data.py                pinned dataset loaders
  item_registry.py       column roles (persona / target), domains, conditions, tasks: the leakage barrier
  split.py               frozen participant-disjoint 70/15/15 split -> data/splits.json
  metrics.py             paper metric (reproduces published numbers), bootstrap, ratio bootstrap
  eval.py                one scorer for every experiment: score, paired delta, gap closed
  pricing.py             (person, slot) -> (product, price) reconstruction
  baselines.py           E0 population median (pricing by product x price)
  features.py            demographics and raw persona features, fitted on train only
  models.py              E1/E2 per-column regularized regression (median decoding) + pricing logit with E0 offset
  knn.py                 E3 nearest neighbours (pricing = E0 + neighbours' product preference)
  ablation.py            persona shuffle: give each person another person's persona
  ft_format.py           bonus: one person -> chat example (prompt modes: persona, none, prior, knn)
  llm_context.py         per-item context for the prompt: population prior (A2) and similar people (B1), training people only
scripts/                 one script per step (see "Run it")
configs/experiment.yaml  dataset revision, split, metric, bootstrap, bonus settings
results/metrics/         aggregate outputs (participant-level predictions are not committed)
```

## Notes on how this was built

- Built with an AI coding assistant. Every number in the reports was re-checked against the result files.
- Some steps did not hold up and were dropped or fixed:
  - a model on the dataset's undocumented trait scores (dropped);
  - a PCA step (removed);
  - a pricing bug that was found and fixed.

## Data and license

Twin-2K-500 (Toubia et al., 2025), CC BY 4.0: <https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500>. The raw data is not committed; `scripts/download_data` fetches it at the pinned revision.
