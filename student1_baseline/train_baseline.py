"""Train + evaluate the classical baseline.

Usage:
  python preprocessing.py --n 50000            # creates data/{train,validation,test}
  python train_baseline.py --epochs 20
"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn as nn

from preprocessing import read_split
from tokenizer import Tokenizer, train_tokenizer
from dataset import make_loader
from transformer_baseline import TransformerMT, count_params
import evaluate as ev


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def eval_loss(model, loader, crit, device):
    model.eval(); tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(b["src"], b["tgt_in"], b["src_pad_mask"], b["tgt_pad_mask"])
        loss = crit(logits.reshape(-1, logits.size(-1)), b["tgt_out"].reshape(-1))
        ntok = b["tgt_out"].ne(0).sum().item()
        tot += loss.item() * ntok; n += ntok
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--comet", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = (read_split(f"{a.data}/{s}", s) for s in ("train", "validation", "test"))

    spm_path = f"{a.data}/spm_joint.model"
    if not os.path.exists(spm_path):
        train_tokenizer(train, f"{a.data}/spm_joint", a.vocab)
    tok = Tokenizer(spm_path)

    tl = make_loader(train, tok, a.batch_size, a.max_len, shuffle=True)
    vl = make_loader(val, tok, a.batch_size, a.max_len)
    model = TransformerMT(tok.vocab_size, a.d_model, a.heads, a.layers, a.d_ff, a.dropout).to(device)
    print("params:", count_params(model), "| device:", device)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.98), eps=1e-9)
    crit = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    plain = nn.CrossEntropyLoss(ignore_index=0)       # un-smoothed, for comparable val loss
    log, best, t_train = [], float("inf"), time.time()

    for ep in range(1, a.epochs + 1):
        model.train(); tot, n = 0.0, 0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(b["src"], b["tgt_in"], b["src_pad_mask"], b["tgt_pad_mask"])   # teacher forcing
            loss = crit(logits.reshape(-1, logits.size(-1)), b["tgt_out"].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        vloss = eval_loss(model, vl, plain, device)
        log.append({"epoch": ep, "train_loss": tot / n, "val_loss": vloss})
        print(f"epoch {ep:02d}  train {tot/n:.4f}  val {vloss:.4f}")
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "args": vars(a)}, f"{a.out}/best.pt")
    train_time = time.time() - t_train

    # ---- final evaluation on the held-out test set with the best checkpoint ----
    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=device)["model"])
    srcs, refs = [e for e, _ in test], [h for _, h in test]
    hyps, sec_per_sent = ev.translate(model, tok, srcs, device, a.batch_size, a.max_len)
    res = {"bleu": ev.bleu_score(hyps, refs), "chrf": ev.chrf_score(hyps, refs),
           "best_val_loss": best, "train_time_sec": train_time,
           "inference_sec_per_sentence": sec_per_sent, "params": count_params(model),
           "config": vars(a), "history": log}
    if a.comet:
        res["comet"] = ev.comet_score(srcs, hyps, refs)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2, ensure_ascii=False)
    with open(f"{a.out}/test_translations.tsv", "w", encoding="utf-8") as f:
        for s, h, r in zip(srcs, hyps, refs):
            f.write(f"{s}\t{h}\t{r}\n")
    print({k: v for k, v in res.items() if k not in ("history", "config")})


if __name__ == "__main__":
    main()
