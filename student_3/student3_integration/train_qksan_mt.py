"""Train and evaluate the QKSAN-integrated translation model.

Run from the repository root after installing requirements:
  python -m student3_integration.train_qksan_mt --epochs 3 --train_limit 2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "student1_baseline"
sys.path.insert(0, str(BASELINE))

from preprocessing import read_split
from tokenizer import Tokenizer
from dataset import make_loader
from evaluate import translate, bleu_score, chrf_score
from transformer_baseline import count_params
from student3_integration.qksan_transformer import build_qksan_transformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["src"], batch["tgt_in"],
                       batch["src_pad_mask"], batch["tgt_pad_mask"])
        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         batch["tgt_out"].reshape(-1))
        count = batch["tgt_out"].ne(0).sum().item()
        total_loss += loss.item() * count
        total_tokens += count
    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/qksan")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--encoding", choices=["angle", "amplitude"], default="angle")
    parser.add_argument("--ansatz", choices=["hardware_efficient", "qaoa"],
                        default="hardware_efficient")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--replace_all", action="store_true",
                        help="Replace all encoder/decoder attention layers (advanced experiment).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_limit", type=int, default=1000,
                        help="Small default because exact quantum simulation is expensive; use 0 for all.")
    parser.add_argument("--eval_limit", type=int, default=200,
                        help="Number of test sentences to translate; use 0 for all.")
    parser.add_argument("--comet", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = read_split(f"{args.data}/train", "train")
    val = read_split(f"{args.data}/validation", "validation")
    test = read_split(f"{args.data}/test", "test")
    if args.train_limit > 0:
        train = train[:args.train_limit]
    if args.eval_limit > 0:
        test = test[:args.eval_limit]

    tokenizer_path = f"{args.data}/spm_joint.model"
    tokenizer = Tokenizer(tokenizer_path)
    train_loader = make_loader(train, tokenizer, args.batch_size,
                               args.max_len, shuffle=True)
    val_loader = make_loader(val[:max(args.batch_size, min(len(val), 200))],
                             tokenizer, args.batch_size, args.max_len)

    model = build_qksan_transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=tokenizer.pad_id,
        n_qubits=args.n_qubits,
        encoding=args.encoding,
        ansatz=args.ansatz,
        depth=args.depth,
        replace_all=args.replace_all,
    ).to(device)

    print(f"device={device} | train_pairs={len(train)} | "
          f"val_pairs={len(val)} | test_pairs={len(test)}")
    print(f"trainable_parameters={count_params(model):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.98), eps=1e-9)
    train_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id,
                                          label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    history, best_val = [], float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, steps = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["src"], batch["tgt_in"],
                           batch["src_pad_mask"], batch["tgt_pad_mask"])
            loss = train_criterion(logits.reshape(-1, logits.size(-1)),
                                   batch["tgt_out"].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
            steps += 1

        val_loss = evaluate_loss(model, val_loader, eval_criterion, device)
        train_loss = loss_sum / max(1, steps)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})
        print(f"epoch {epoch:02d}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       f"{args.out}/best.pt")

    train_time = time.time() - start
    checkpoint = torch.load(f"{args.out}/best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    sources = [source for source, _ in test]
    references = [reference for _, reference in test]
    hypotheses, inference_sec = translate(
        model, tokenizer, sources, device, args.batch_size, args.max_len
    )
    metrics = {
        "model": "Transformer + QKSAN attention",
        "bleu": bleu_score(hypotheses, references),
        "chrf": chrf_score(hypotheses, references),
        "best_val_loss": best_val,
        "train_time_sec": train_time,
        "inference_sec_per_sentence": inference_sec,
        "params": count_params(model),
        "train_examples": len(train),
        "test_examples": len(test),
        "config": vars(args),
        "history": history,
    }
    if args.comet:
        from evaluate import comet_score
        metrics["comet"] = comet_score(sources, hypotheses, references)
    with open(f"{args.out}/results.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    with open(f"{args.out}/test_translations.tsv", "w", encoding="utf-8") as file:
        for source, hypothesis, reference in zip(sources, hypotheses, references):
            file.write(f"{source}\t{hypothesis}\t{reference}\n")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in {"history", "config"}}, indent=2))


if __name__ == "__main__":
    main()
