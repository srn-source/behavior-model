# Large Behavior Model on Twin-2K-500

**Task.** From a person's survey answers in waves 1–3 (their persona), predict how they answer the repeated questions in wave 4. Score against their real wave-4 answers and against human test–retest. Data: Twin-2K-500, 2,058 US adults, four waves.

**Short answer.** Most of the accuracy comes from knowing the typical answer, not the person.

- The training people's median answer (E0), with no personal information, scores **77.3%**. A person's own earlier answer (retest) scores **81.8%**.
- The best personal model (E3, kNN) scores **78.1%**, a small gain of **+0.78** points.
- The paper's zero-shot LLM twins score **68.3–72.8%**, all below E0.
- A fine-tuned 0.5B LLM scores **75.8%**: above the twins, still below E0, mostly because it almost never says "buy" on pricing.

## Reports

| # | Report | Question it answers |
|---|---|---|
| 1 | [Data exploration](reports/01_data_exploration.md) | What is in the data, how reliable are people, what leaks the answer? |
| 2 | [Model plan](reports/02_model_plan.md) | What to build first, and why (includes the bonus LLM) |
| 3 | [Evaluation](reports/03_evaluation.md) | Does a model beat the right baseline? |
| 4 | [Business use](reports/04_business.md) | Who can use this, and what must it never do? |
| 5 | [Maintenance](reports/05_maintenance.md) | How does the model stay trustworthy over time? |

Every number in the reports comes from a file in [results/metrics/](results/metrics/).

## Results

Test split: 311 held-out people. Score = the paper's metric (higher is better).

| Model | What it uses | Score [95% CI] | vs E0 [95% CI] |
|---|---|---|---|
| Human retest | the person's own earlier answer | 81.8% [81.0, 82.6] | +4.55 |
| **E3** kNN | answers of the 100 most similar training people | **78.1%** [77.3, 78.8] | **+0.78** [+0.49, +1.06] |
| E2-All | demographics + all persona answers, regularized regression | 78.0% [77.2, 78.7] | +0.67 [+0.28, +1.09] |
| E1 | demographics only | 77.7% [77.0, 78.5] | +0.44 [+0.12, +0.77] |
| **E0** | median answer of training people, per question and condition | **77.3%** [76.5, 78.1] | – |
| C2-knn | fine-tuned Qwen2.5-0.5B + 20 similar people in the prompt | 76.1% [75.3, 76.9] | −1.17 [−1.66, −0.70] |
| E6 | fine-tuned Qwen2.5-0.5B (LoRA) | 75.8% [75.0, 76.6] | −1.48 [−1.70, −1.26] |
| E4 | best published LLM twin (GPT-4.1, zero-shot) | 72.8% [72.0, 73.5] | −4.48 |
| E5 | the same 0.5B model, untrained | 59.2% [58.4, 60.0] | −18.13 |
| Random | uniform random answer | 59.4% | |

- E0, E1–E3 and E6 learned from other people's wave-4 answers; the published twins did not. E0 vs twins compares regimes, not architectures.
- Swapping in another person's persona drops E2-All and E3 below E0, so the models do use the right person's data; it just carries little signal ([persona_shuffle.md](results/metrics/persona_shuffle.md)).
- Full tables: [leaderboard.md](results/metrics/leaderboard.md), [llm_ablation.md](results/metrics/llm_ablation.md).

## Three data traps

| Trap | Effect | Guard |
|---|---|---|
| `full_persona` holds the wave-4 answers | 100% match on the 41,242 answers that changed between waves | Use only `wave_split` |
| `wave1_3_response.csv` holds earlier answers to the target questions | Prediction becomes copying | Inputs only from registry columns with `role == "persona"` |
| Pricing slots `QID9_1..40` are randomized products | Scoring by slot costs 17.5 points on pricing | Score per product × price |

Checked by `scripts/audit_leakage.py` → [leakage_audit.md](results/metrics/leakage_audit.md).

## Run it

Python 3.11, CPU only, about 680 MB download. Seeds are fixed.

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
.venv/bin/python -m scripts.download_data        # dataset at pinned revision f883165a
.venv/bin/python -m scripts.audit_leakage        # leakage checks
.venv/bin/python -m scripts.reproduce_retest     # checks the paper's retest and LLM numbers
.venv/bin/python -m scripts.token_counts         # persona lengths
.venv/bin/python -m scripts.eda                  # tables and figures for report 01
.venv/bin/python -m scripts.run_e0               # E0
.venv/bin/python -m scripts.run_e4               # published LLM twins on our split
.venv/bin/python -m scripts.run_e1_e2            # E1, E2 (~30 min)
.venv/bin/python -m scripts.run_e3               # E3
.venv/bin/python -m scripts.evaluate             # leaderboard
.venv/bin/python -m scripts.run_e0_cv            # E0 on all people vs the paper
.venv/bin/python -m scripts.run_persona_shuffle  # persona-shuffle check (~4 min)
```

### Bonus: fine-tune a model under 0.5B (GPU)

`Qwen/Qwen2.5-0.5B-Instruct`, **494,032,768 parameters** (Apache-2.0). LoRA r 16, α 32, all linear layers, loss on answer lines only, 2 epochs, training people only. The adapter is merged before prediction, so the evaluated model still has 494M parameters ([e6_run_config.json](results/metrics/e6_run_config.json)). One example is one person: compact persona, then their wave-4 questions, then their answers (~24k tokens), so a full run needs one 80 GB GPU. GPU packages are in `requirements-ft.txt`.

```bash
python -m scripts.ft_build_data --tokenizer Qwen/Qwen2.5-0.5B-Instruct
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_train
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5_qwen05_zeroshot
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E6_qwen05_lora --adapter outputs/qwen25_05b_lora
.venv/bin/python -m scripts.evaluate
```

Prompting vs retrieval vs fine-tuning on the same model (RQ5):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5a_qwen05_none --prompt none --split test
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5b_qwen05_prior_median --prompt prior_median --split test
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E5c_qwen05_knn_k20 --prompt knn --k 20 --split test
python -m scripts.ft_build_data --out data/ft_knn20 --prompt knn --k 20 --tokenizer Qwen/Qwen2.5-0.5B-Instruct
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_train --train data/ft_knn20/train.jsonl --val data/ft_knn20/val.jsonl --out outputs/qwen25_05b_lora_knn20
CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_predict --exp E6k_qwen05_lora_knn20 --adapter outputs/qwen25_05b_lora_knn20 --prompt knn --k 20 --split test
.venv/bin/python -m scripts.evaluate
.venv/bin/python -m scripts.analyze_llm_ablation
```

C2-prior is the same with `--prompt prior_median`, `data/ft_prior`, `outputs/qwen25_05b_lora_prior` and `--exp E6p_qwen05_lora_prior`.

## Layout

```text
reports/            the five reports and their figures
src/                data loading, item registry (leakage barrier), split, metric, models, LLM prompt formats
scripts/            one script per step (see "Run it")
configs/            dataset revision, split, metric and bonus settings
data/splits.json    frozen participant split (1,440 / 307 / 311)
results/metrics/    aggregate results (per-person predictions are not published)
```

Built with an AI coding assistant; every reported number was re-checked against the result files.

## Data and license

Twin-2K-500 (Toubia et al., 2025), CC BY 4.0: <https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500>. Raw data is not committed; `scripts.download_data` fetches it at the pinned revision.
