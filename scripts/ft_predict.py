"""Bonus, E5 / E6 (RQ4) and the prompt variants of RQ5: predict wave-4 answers with Qwen2.5-0.5B-Instruct, untrained (E5) or with the LoRA adapter (E6).

Same prompt as training (src/ft_format.build_example, without answers). The answer lines are written one item at a
time with the item id forced ("QID158: "), so every item gets an answer and the order matches training:
  choice items   probability of every option number (each option string scored token by token, so "1" and "10"
                 are told apart), answer = median of that distribution, as in E1-E2-All (the paper metric scores
                 distance; with argmax E1 falls below E0). Probabilities are saved too.
  number items   greedy decoding until a newline (at most 8 tokens), parsed as a number; unparseable = missing.
The chosen answer is appended before the next item, so the model sees its own earlier answers, as in training.

Outputs (scored by scripts.evaluate like every other experiment):
  results/predictions/{exp_id}__wave4.parquet / .json         raw answers, pid x scored column
  results/predictions/option_probs/{exp_id}__wave4.parquet     pid, column, option, prob (choice items)

Run (GPU environment from requirements-ft.txt, one GPU):
  python -m scripts.ft_predict --exp E5_qwen05_zeroshot
  python -m scripts.ft_predict --exp E6_qwen05_lora --adapter outputs/qwen25_05b_lora
  add --limit 5 for a smoke test

Prompting vs retrieval (zero-shot, no adapter), see src/ft_format.py and src/llm_context.py:
  --prompt none                  A0  questions only
  --prompt persona               A1  persona + questions (= E5)
  --prompt prior_median|prior_dist  A2  questions + how training people answered each item (E0 in the prompt)
  --prompt knn --k K             B1  questions + answers of the K most similar training people (E3's neighbours)
  python -m scripts.ft_predict --exp E5b_qwen05_prior_median --prompt prior_median --split val
Running the same --exp on --split val and then --split test merges both into one prediction file (scripts.evaluate
scores each split separately). Choose prompt variants on val; run only the chosen one on test.

C2, fine-tuned with context in the prompt (data from scripts.ft_build_data --prompt ..., adapter from scripts.ft_train):
predict with the same --prompt / --k the adapter was trained with (checked against its run_config.json), e.g.
  python -m scripts.ft_predict --exp E6p_qwen05_lora_prior --adapter outputs/qwen25_05b_lora_prior --prompt prior_median

Persona-shuffle ablation (src/ablation.py): every test person gets another test person's persona, with
their own wave-4 questions. Saved under results/predictions/ablation/ (not on the leaderboard) and scored by
scripts/run_persona_shuffle.py:
  python -m scripts.ft_predict --exp E6_qwen05_lora --adapter outputs/qwen25_05b_lora --persona-shuffle-seed 0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd
import torch

from src import data
from src import metrics as M
from src.ablation import donor_map
from src.eval import PRED_DIR, save_predictions
from src.ft_format import PROMPTS, build_example
from src.llm_context import NeighbourContext, PopulationPrior
from src.models import _ordered_median
from src.split import load_split

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
MAX_NUMBER_TOKENS = 8


class Session:
    """One person's prompt in the KV cache; answer text is appended and the cache kept in sync with re-tokenisation.
    The prompt ends with the template's "<|im_start|>assistant\n", so prompt and answer tokenise independently
    (checked: identical to tokenising the concatenation); only the short answer text is re-tokenised."""

    def __init__(self, model, tok, prompt: str):
        self.model, self.tok = model, tok
        self.prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        self.ids: list[int] = []
        self.cache = None
        self.logits = None
        self.extend("")

    @torch.no_grad()
    def _forward(self, new_ids: list[int], keep: int = 1):
        """Run new tokens through the model; return logits of the last `keep` positions only (memory)."""
        x = torch.tensor([new_ids], device=self.model.device)
        out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True, logits_to_keep=keep)
        self.cache = out.past_key_values
        return out.logits[0].float()

    def extend(self, assistant_text: str):
        """Make the cache hold prompt + assistant_text; return log-probs of the next token."""
        ids = self.prompt_ids + (self.tok(assistant_text, add_special_tokens=False)["input_ids"] if assistant_text else [])
        k = 0
        while k < min(len(ids), len(self.ids)) and ids[k] == self.ids[k]:
            k += 1
        if k == len(ids):  # nothing new (only possible for the very first call on an identical prompt)
            k -= 1
        if self.cache is not None and k < len(self.ids):
            self.cache.crop(k)
        self.logits = self._forward(ids[k:])[-1]
        self.ids = ids
        return torch.log_softmax(self.logits, -1)

    def score(self, continuation_ids: list[int]) -> float:
        """log P(continuation | current text), then roll the cache back."""
        n = len(self.ids)
        logp = float(torch.log_softmax(self.logits, -1)[continuation_ids[0]])
        if len(continuation_ids) > 1:
            lg = torch.log_softmax(self._forward(continuation_ids[:-1], keep=len(continuation_ids) - 1), -1)
            logp += float(sum(lg[i, t] for i, t in enumerate(continuation_ids[1:])))
            self.cache.crop(n)
        return logp


def option_ids(tok, n: int) -> list[list[int]]:
    """Token ids of "k" followed by a newline, for k = 1..n (the newline marks the end of the number)."""
    return [tok(f"{k}\n", add_special_tokens=False)["input_ids"] for k in range(1, n + 1)]


def predict_person(model, tok, pid: int, persona_pid: int | None = None, prompt: str = "persona", hint=None):
    ex = build_example(pid, persona_pid=persona_pid, prompt=prompt, hint=hint)
    prompt = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
    s = Session(model, tok, prompt)
    text, answers, probs = "", {}, []
    for j, it in enumerate(ex["items"]):
        text += ("\n" if j else "") + f"{it['column']}: "
        s.extend(text)
        if it["kind"] == "choice":
            lp = np.array([s.score(ids) for ids in option_ids(tok, it["n_options"])])
            p = np.exp(lp - lp.max())
            p /= p.sum()
            classes = np.arange(1, it["n_options"] + 1, dtype=float)
            ans = float(_ordered_median(classes, p[None, :])[0])
            probs += [dict(pid=pid, column=it["column"], option=int(k), prob=float(q)) for k, q in zip(classes, p)]
            text += str(int(ans))
        else:
            out = ""
            for _ in range(MAX_NUMBER_TOKENS):
                nxt = int(torch.argmax(s.logits))
                piece = tok.decode([nxt])
                if nxt == tok.eos_token_id or "\n" in piece or "<|im_end|>" in piece:
                    break
                out += piece
                s.extend(text + out)
            m = NUMBER.search(out)
            ans = float(m.group()) if m else np.nan
            text += m.group() if m else out.strip()
        answers[it["column"]] = ans
    return answers, probs


PROMPT_INFO = {  # prompt -> (inputs, regime)
    "persona": ("persona text (wave_split wave1_3_persona_json) + wave-4 questions", "item-zero-shot"),
    "none": ("wave-4 questions only (no information about the person)", "item-zero-shot"),
    "prior_median": ("wave-4 questions + training people's typical answer per item (E0 in the prompt)", "in-context-on-items"),
    "prior_dist": ("wave-4 questions + share of training people choosing each option (E0's distribution in the prompt)",
                   "in-context-on-items"),
    "knn": ("wave-4 questions + answers of the k most similar training people (E3's neighbours, persona cosine)",
            "in-context-on-items"),
}


def context_for(prompt: str, k: int | None, pids):
    """Per-person hint function for the prompting modes (training people only), or None."""
    train = load_split()["train"]
    if prompt in ("prior_median", "prior_dist"):
        prior = PopulationPrior(train, prompt.split("_")[1])
        return lambda pid: prior.hint
    if prompt == "knn":
        nb = NeighbourContext(train, k).prepare(pids)
        return nb.hint_for
    return lambda pid: None


def _merge_existing(path, pred: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, list[str]]:
    """Keep rows of other splits already saved under this exp id (same prompt and adapter), replace the rest."""
    splits = [meta["split"]]
    if path.exists():
        old_meta = json.loads(path.with_suffix(".json").read_text())
        for key in ("prompt", "k", "adapter"):
            if old_meta.get(key, "persona" if key == "prompt" else None) != meta.get(key):
                raise SystemExit(f"{path.name} was made with {key}={old_meta.get(key)!r}, not {meta.get(key)!r}: use another --exp")
        old = pd.read_parquet(path)
        pred = pd.concat([old.drop(index=pred.index, errors="ignore"), pred])
        splits = sorted(set(old_meta.get("splits", [old_meta.get("split", "test")])) | set(splits))
    return pred, splits


def main(exp: str, adapter: str | None, split_name: str, limit: int | None, device: str, shuffle_seed: int | None = None,
         prompt: str = "persona", k: int | None = None):
    if prompt not in PROMPTS:
        raise SystemExit(f"--prompt must be one of {PROMPTS}")
    if (prompt == "knn") != (k is not None):
        raise SystemExit("--k is required with --prompt knn and only then")
    if adapter:
        cfg = pathlib.Path(adapter) / "run_config.json"
        trained = json.loads(cfg.read_text()) if cfg.exists() else {}
        trained = (trained.get("prompt", "persona"), trained.get("k"))
        if trained != (prompt, k):
            raise SystemExit(f"the adapter was trained with prompt={trained[0]!r}, k={trained[1]!r}: predict with the same")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    all_pids = load_split()[split_name]
    pids = all_pids[:limit] if limit else all_pids
    donors = donor_map(all_pids, shuffle_seed) if shuffle_seed is not None else {}  # over the whole split
    hint_for = context_for(prompt, k, pids)
    rows, probs, t0 = {}, [], time.time()
    for i, pid in enumerate(pids, 1):
        rows[pid], p = predict_person(model, tok, pid, donors.get(int(pid)), prompt, hint_for(int(pid)))
        probs += p
        if i % 10 == 0 or i == len(pids):
            print(f"{i}/{len(pids)} people  {time.time() - t0:.0f}s", flush=True)
    pred = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=M.SCORED_COLUMNS)
    inputs, regime = PROMPT_INFO[prompt]
    tag = {"persona": "", "none": " · questions only", "prior_median": " · prompt: population median",
           "prior_dist": " · prompt: population distribution", "knn": f" · prompt: {k} similar people"}[prompt]
    if adapter and prompt != "persona":
        tag += " (trained with this prompt)"
    meta = dict(label=f"{exp}: Qwen2.5-0.5B-Instruct" + (" + LoRA (train people, wave-4 answers)" if adapter else " zero-shot") + tag,
                regime="supervised-on-items" if adapter else regime, setting="A",
                inputs=inputs, prompt=prompt, k=k, base_model=BASE,
                parameters=int(n_params), adapter=adapter, split=split_name, people=len(pids),
                decoding="choice: median of option probabilities; number: greedy",
                caveat="" if adapter else ("same prompt and decoding as E6, no training" if prompt == "persona" else
                                           "same model and decoding as E5/E6, no training; only the prompt differs"))
    if shuffle_seed is not None:  # ablation: kept off the leaderboard, scored by scripts.run_persona_shuffle
        exp = f"{exp}_persona_shuffle_s{shuffle_seed}"
        meta.update(label=meta["label"] + f" · persona shuffled (seed {shuffle_seed})", ablation="persona_shuffle",
                    shuffle_seed=shuffle_seed, exp_id=exp, target="wave4", n_pids=len(pred),
                    dataset_revision=data.DATASET_REVISION)
        out = PRED_DIR / "ablation"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{exp}__wave4.parquet"
        pred.index = pred.index.astype(int)
        pred.index.name = "pid"
        pred.apply(pd.to_numeric, errors="coerce").to_parquet(path)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"saved {path}  ({time.time() - t0:.0f}s)")
        return
    pred.index = pred.index.astype(int)
    pred, meta["splits"] = _merge_existing(PRED_DIR / f"{exp}__wave4.parquet", pred, meta)
    meta["people"] = len(pred)
    path = save_predictions(exp, "wave4", pred, meta)
    (PRED_DIR / "option_probs").mkdir(parents=True, exist_ok=True)
    probs_path = PRED_DIR / "option_probs" / f"{exp}__wave4.parquet"
    probs = pd.DataFrame(probs)
    if probs_path.exists():
        old = pd.read_parquet(probs_path)
        probs = pd.concat([old[~old.pid.isin(probs.pid.unique())], probs], ignore_index=True)
    probs.to_parquet(probs_path)
    print(f"saved {path}  ({n_params:,} parameters, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="E5_qwen05_zeroshot or E6_qwen05_lora")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir written by scripts.ft_train (omit for zero-shot E5)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--persona-shuffle-seed", type=int, default=None,
                    help="ablation: give each person another person's persona (src/ablation.py)")
    ap.add_argument("--prompt", default="persona", choices=PROMPTS, help="prompting mode (module docstring)")
    ap.add_argument("--k", type=int, default=None, help="neighbours for --prompt knn")
    a = ap.parse_args()
    main(a.exp, a.adapter, a.split, a.limit, a.device, a.persona_shuffle_seed, a.prompt, a.k)
