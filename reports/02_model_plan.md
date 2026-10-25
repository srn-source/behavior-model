# 2. Model Plan — Start with the Prior, Earn Personalization

**Goal.** Predict a person's Wave-4 answer from their leakage-safe Wave 1–3 persona, the question and its experimental condition (for pricing: product and price).

## Recommendation in one paragraph

**Start with a condition-aware population prior (E0), then learn a regularized person-specific adjustment.** Keep small-LLM fine-tuning as a research track. E0 already scores **77.3%**; the best personalized model (E3, kNN) scores **78.1%**, only **+0.78 points**. The fine-tuned 0.5B LLM (E6) scores **75.8%**, below E0.

## Evidence that shapes the design

| Approach | Score (311 held-out people) | Lesson |
|---|---:|---|
| E0: population median per item and condition | **77.3%** | Hard to beat |
| E1: demographics | 77.7% | Simple personal information helps slightly |
| E2-All: regularized model on all persona answers | 78.0% | More persona data, modest gain |
| E3: kNN, 100 most similar people | **78.1%** | Best observed |
| E4: the paper's best LLM twin (zero-shot) | 72.8% | Below E0 (different supervision; reference only) |
| E5: Qwen2.5-0.5B, untrained | 59.2% | Prompting a small model is near chance |
| E6: same model, LoRA fine-tuned | 75.8% | Training helps a lot, still below E0 |
| C2-knn: E6 + similar people's answers in the prompt | 76.1% | Context helps once trained, not enough |

![Prompting vs retrieval vs fine-tuning](figures/llm_ablation.png)

## Architecture

```
Waves 1–3 persona ──► features (tabular) or compact text (LLM) ──┐
                                                                 ├──► answer distribution ──► median answer
Question + condition ──► population prior E0 (training people) ──┘
```

Every model starts from E0 and learns only how a person differs from it. With the strongest regularization it returns E0 exactly, so personal information can only move a prediction when the data supports it. The answer is the **median** of the predicted distribution, because the metric is an absolute error.

## Pipeline

1. **Split people first:** 1,440 train / 307 validation / 311 test, participant-disjoint.
2. **Typed item registry:** each column is persona (input) or target, with task, condition and scale. Pricing slots are mapped to product × price.
3. **Inputs:** persona columns only (`wave_split`); Wave-4 answers are labels, never inputs.
4. **E0 on training people:** median per item and condition; buy rate per product × price.
5. **Personalization:** demographics first, then persona blocks; regularization chosen on validation. kNN as a non-linear alternative.
6. **Select:** paired difference vs E0, per-task scores, persona shuffle. Add complexity only when it adds measurable value.

## Small-LLM track (bonus, implemented)

**Setup.** Qwen2.5-0.5B-Instruct (**494,032,768 parameters**), LoRA on all linear layers (**r 16, α 32, dropout 0.05, lr 2e-4, 2 epochs**), loss on answer tokens only. One training example = one person: compact persona (~16.9k tokens) + their Wave-4 questions → their Wave-4 answers. The adapter (~8.8M parameters) is merged before prediction, so the evaluated model has 494M parameters.

![Persona length](figures/persona_tokens.png)

The text persona does not fit next to all Wave-4 questions in a 32k context; the compact Q/A form (question + chosen answer, option lists dropped) fits for everyone.

**Observed failure.** Training adds **+16.65 points** over the untrained model, but E6 stays **1.48 points below E0**, mostly from pricing (54.3% vs E0 72.6%): it says "buy" for 0.0% of pricing rows. Its buy probability falls from 0.42 in the first slot to 0.10 on average later, consistent with copying its own earlier "no" answers. The cause is not isolated.

**Prompting, retrieval, preference tuning.**
- **Prompting** without training: near chance (56.6–59.4%) whatever the context.
- **Retrieval** (similar people's answers in the prompt) helps only after training (C2-knn 76.1%), still below E0.
- **GRPO / preference tuning: not run.** Its optimum under a per-item reward is again the median answer, which E0 already gives. Deferred until an SFT model beats E0.

**Next experiment (not run):** score each question on its own (no dependence on earlier generated answers), with E0 as a prior on the answer options, and compare against E0 and E3 before scaling the model.

## Risks and decision gates

| Risk | Mitigation |
|---|---|
| Answer leakage (`full_persona`, earlier target answers) | Registry allowlist and leakage audit |
| Long context | Compact Q/A persona; measure token length |
| A high score only reflects the typical answer | Always report the paired difference vs E0 and persona shuffle |
| Overfitting 1,440 people | Regularization chosen on validation; simpler fallback |
| LLM pricing collapse | Per-task checks; independent-item scoring before new training methods |
| New questions or populations | Not tested; needs its own evaluation |

**Final choice today:** E0 as the default and reference; E2-All / E3 as low-complexity personalized challengers. The LLM is not production-ready until it beats them on fresh data.

*Sources: `results/metrics/leaderboard.md`, `llm_ablation.md`, `e6_run_config.json`, `token_counts.json`.*
