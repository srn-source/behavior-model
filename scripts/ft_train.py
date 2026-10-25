"""Bonus, E6 (RQ4): LoRA fine-tune of Qwen2.5-0.5B-Instruct on the persona -> wave-4 answers task.

Plain transformers + peft (no trainer framework), one GPU:
  data     data/ft/train.jsonl (1,440 train people) from scripts/ft_build_data.py; data/ft/val.jsonl for eval loss
  loss     next-token cross-entropy on the assistant turn only (the "<column>: <answer>" lines + end-of-turn);
           the system prompt, persona and questions are masked (-100)
  model    Qwen/Qwen2.5-0.5B-Instruct, bf16, LoRA on every linear layer of the transformer blocks
  schedule AdamW, linear warm-up then cosine decay, gradient accumulation, gradient checkpointing
  output   adapter after every epoch (epoch_k/), the epoch with the lowest val loss copied to the output dir,
           train_log.csv (step, train_loss, val_loss, lr), run_config.json

Run (GPU environment from requirements-ft.txt):  CUDA_VISIBLE_DEVICES=0 python -m scripts.ft_train
Smoke test:              python -m scripts.ft_train --limit 8 --max-steps 4 --out /tmp/ft_smoke
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import shutil
import time

import torch

from src import data

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = data.REPO_ROOT / "outputs" / "qwen25_05b_lora"
FT = data.REPO_ROOT / "data" / "ft"


def encode(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """Token ids of the whole chat and labels with everything before the assistant answer masked."""
    full = tok.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
    prompt = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
    assert full[:len(prompt)] == prompt, "chat template: prompt is not a prefix of the full chat"
    if len(full) > max_len:
        raise ValueError(f"example of {len(full)} tokens exceeds max_len {max_len}")
    return full, [-100] * len(prompt) + full[len(prompt):]


def load(path: pathlib.Path, tok, max_len: int, limit: int | None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = rows[:limit] if limit else rows
    return [encode(tok, r["messages"], max_len) for r in rows]


def loss_of(model, ids, labels, device) -> torch.Tensor:
    x = torch.tensor([ids], device=device)
    y = torch.tensor([labels], device=device)
    return model(input_ids=x, labels=y).loss  # mean over answer tokens (shifted inside the model)


@torch.no_grad()
def eval_loss(model, examples, device) -> float:
    model.eval()
    total = sum(float(loss_of(model, ids, lab, device)) for ids, lab in examples)
    model.train()
    return total / len(examples)


def main(a):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    train = load(pathlib.Path(a.train), tok, a.max_len, a.limit)
    val = load(pathlib.Path(a.val), tok, a.max_len, a.limit)
    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation=attn).to(device)
    n_base = sum(p.numel() for p in model.parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train()

    steps_per_epoch = math.ceil(len(train) / a.grad_accum)
    total = min(a.max_steps, steps_per_epoch * a.epochs) if a.max_steps else steps_per_epoch * a.epochs
    warmup = max(1, int(a.warmup * total))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warmup) / max(1, total - warmup)))))
    manifest = pathlib.Path(a.train).parent / "manifest.json"  # written by scripts.ft_build_data
    data_info = json.loads(manifest.read_text()) if manifest.exists() else {}
    run = dict(vars(a), prompt=data_info.get("prompt", "persona"), k=data_info.get("k"),
               base_model=BASE, base_parameters=n_base, trainable_parameters=n_train, attn=attn,
               train_examples=len(train), val_examples=len(val), optimizer_steps=total,
               train_tokens_median=sorted(len(i) for i, _ in train)[len(train) // 2])
    (out / "run_config.json").write_text(json.dumps(run, indent=2, default=str))
    print(json.dumps(run, indent=2, default=str), flush=True)

    log, step, best, t0 = [], 0, (math.inf, None), time.time()
    val0 = eval_loss(model, val, device)
    log.append(dict(step=0, epoch=0, train_loss=None, val_loss=val0, lr=0.0))
    print(f"step 0  val_loss {val0:.4f}", flush=True)
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        running, n_in = 0.0, 0
        for i, k in enumerate(order, 1):
            ids, lab = train[k]
            loss = loss_of(model, ids, lab, device)
            (loss / a.grad_accum).backward()
            running += float(loss.detach())
            n_in += 1
            if n_in == a.grad_accum or i == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.append(dict(step=step, epoch=epoch, train_loss=running / n_in, val_loss=None, lr=sched.get_last_lr()[0]))
                if step % a.log_every == 0:
                    print(f"epoch {epoch} step {step}/{total}  train_loss {running / n_in:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
                running, n_in = 0.0, 0
                if a.max_steps and step >= a.max_steps:
                    break
        vl = eval_loss(model, val, device)
        log.append(dict(step=step, epoch=epoch, train_loss=None, val_loss=vl, lr=sched.get_last_lr()[0]))
        print(f"epoch {epoch} done: val_loss {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        model.save_pretrained(out / f"epoch_{epoch}")
        if vl < best[0]:
            best = (vl, epoch)
        if a.max_steps and step >= a.max_steps:
            break

    import pandas as pd
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    for f in (out / f"epoch_{best[1]}").iterdir():  # best epoch -> output dir (what ft_predict loads)
        shutil.copy(f, out / f.name)
    run.update(best_epoch=best[1], best_val_loss=best[0], val_loss_before_training=val0, seconds=round(time.time() - t0))
    (out / "run_config.json").write_text(json.dumps(run, indent=2, default=str))
    print(f"best epoch {best[1]} (val loss {best[0]:.4f}, before training {val0:.4f}) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(FT / "train.jsonl"))
    ap.add_argument("--val", default=str(FT / "val.jsonl"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=26624)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="first N examples of train and val (smoke test)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke test)")
    main(ap.parse_args())
