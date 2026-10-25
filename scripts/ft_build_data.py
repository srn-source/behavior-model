"""Bonus, E6 (RQ4): write the chat-format training data for scripts.ft_train.

One example per person = waves 1-3 persona (compact text, D4 wave_split `wave1_3_persona_json`) +
the wave-4 questions they were shown (D4 `wave4_Q_wave4_A`, answers removed) -> their wave-4 answers (D3).
Train and val people only, from the frozen split; test people are never written here (their prompts are
rebuilt at prediction time without answers). See src/ft_format.py.

Outputs (default data/ft/, git-ignored):
  train.jsonl   {"pid", "messages": [system, user, assistant]}   1,440 people
  val.jsonl     same for the 307 val people (validation loss and epoch choice only)
  manifest.json counts, token lengths (with --tokenizer), split sizes, dataset revision

C2 (fine-tuning with context in the prompt, see src/llm_context.py):
  --prompt prior_median            under each item, training people's typical answer / buy rate (E0 in the prompt)
  --prompt knn --k 20              under each item, the answers of the 20 most similar training people
For a training person the context leaves that person out (leave-one-out statistics; never their own neighbour),
so training sees the same conditions as prediction. Validation people get the ordinary context.

Run:  python -m scripts.ft_build_data [--out DIR] [--limit N] [--tokenizer Qwen/Qwen2.5-0.5B-Instruct]
                                      [--prompt persona|prior_median|prior_dist|knn] [--k K]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from src import data
from src.ft_format import PROMPTS, build_example
from src.llm_context import NeighbourContext, PopulationPrior
from src.split import load_split

OUT = data.REPO_ROOT / "data" / "ft"


def context(prompt: str, k: int | None, split):
    """pid -> hint function (None for the persona prompt); training people are left out of their own context."""
    if prompt in ("prior_median", "prior_dist"):
        prior = PopulationPrior(split["train"], prompt.split("_")[1])
        return prior.hint_for
    if prompt == "knn":
        nb = NeighbourContext(split["train"], k).prepare(list(split["train"]) + list(split["val"]))
        return nb.hint_for
    return lambda pid: None


def main(out: pathlib.Path, limit: int | None, tokenizer: str | None, prompt: str = "persona", k: int | None = None):
    if (prompt == "knn") != (k is not None):
        raise SystemExit("--k is required with --prompt knn and only then")
    out.mkdir(parents=True, exist_ok=True)
    split = load_split()
    hint_for = context(prompt, k, split)
    w4 = data.load_responses("wave4")  # D3: wave-4 answers = targets
    tok = None
    if tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
    manifest = dict(dataset_revision=data.DATASET_REVISION, prompt=prompt, k=k, splits={})
    for name in ("train", "val"):
        pids = split[name][:limit] if limit else split[name]
        lengths = []
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for pid in pids:
                ex = build_example(pid, w4.loc[pid], prompt=prompt, hint=hint_for(int(pid)))
                f.write(json.dumps({"pid": ex["pid"], "messages": ex["messages"]}, ensure_ascii=False) + "\n")
                if tok is not None:
                    ids = tok.apply_chat_template(ex["messages"], tokenize=True, return_dict=True)["input_ids"]
                    lengths.append(len(ids))
        info = dict(people=len(pids))
        if lengths:
            info.update(tokens_min=int(min(lengths)), tokens_median=int(np.median(lengths)), tokens_max=int(max(lengths)))
        manifest["splits"][name] = info
        print(f"{name:5s} {len(pids):5d} people -> {out / (name + '.jsonl')}  {info}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--limit", type=int, default=None, help="only the first N people per split (smoke test)")
    ap.add_argument("--tokenizer", default=None, help="also report token lengths with this tokenizer")
    ap.add_argument("--prompt", default="persona", choices=PROMPTS)
    ap.add_argument("--k", type=int, default=None, help="neighbours for --prompt knn")
    a = ap.parse_args()
    main(a.out, a.limit, a.tokenizer, a.prompt, a.k)
