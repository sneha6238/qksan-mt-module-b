"""Compare supplied classical baseline results with QKSAN experiment JSON files."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out", default="results/module_c_analysis")
    args = parser.parse_args()
    root, out = Path(args.results_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in ["baseline", "baseline_lr5e-4", "trial"]:
        path = root / name / "results.json"
        if path.exists():
            item = read_json(path)
            rows.append({
                "model": name,
                "BLEU": item.get("bleu"),
                "chrF": item.get("chrf"),
                "validation_loss": item.get("best_val_loss"),
                "training_seconds": item.get("train_time_sec"),
                "inference_seconds_per_sentence": item.get("inference_sec_per_sentence"),
                "parameters": item.get("params"),
            })
    for path in sorted(root.glob("qksan*/results.json")) + sorted(root.glob("ablation_*/results.json")):
        item = read_json(path)
        rows.append({
            "model": path.parent.name,
            "BLEU": item.get("bleu"),
            "chrF": item.get("chrf"),
            "validation_loss": item.get("best_val_loss"),
            "training_seconds": item.get("train_time_sec"),
            "inference_seconds_per_sentence": item.get("inference_sec_per_sentence"),
            "parameters": item.get("params"),
        })

    if not rows:
        raise SystemExit("No results.json files found.")
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Keep only comparable numeric BLEU/chrF values; metric scales must match.
    labels = [row["model"] for row in rows if row["BLEU"] is not None and row["chrF"] is not None]
    bleu = [row["BLEU"] for row in rows if row["BLEU"] is not None and row["chrF"] is not None]
    chrf = [row["chrF"] for row in rows if row["BLEU"] is not None and row["chrF"] is not None]
    if labels:
        plt.figure(figsize=(max(7, len(labels) * 1.4), 4.5))
        positions = list(range(len(labels)))
        plt.bar([p - 0.18 for p in positions], bleu, width=0.36, label="BLEU")
        plt.bar([p + 0.18 for p in positions], chrf, width=0.36, label="chrF")
        plt.xticks(positions, labels, rotation=25, ha="right")
        plt.ylabel("Score")
        plt.title("Classical Transformer vs QKSAN-MT")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "quality_comparison.png", dpi=180)
        plt.close()

    # Plot QKSAN training histories, if available.
    histories = []
    for path in sorted(root.glob("qksan*/results.json")) + sorted(root.glob("ablation_*/results.json")):
        history = read_json(path).get("history", [])
        if history:
            histories.append((path.parent.name, history))
    if histories:
        plt.figure(figsize=(7, 4.5))
        for name, history in histories:
            plt.plot([x["epoch"] for x in history],
                     [x["val_loss"] for x in history], marker="o", label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")
        plt.title("QKSAN-MT validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "qksan_validation_loss.png", dpi=180)
        plt.close()

    print(f"Saved {out / 'comparison.csv'} and available graphs.")


if __name__ == "__main__":
    main()
