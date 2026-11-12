"""Create or run controlled QKSAN ablations.

Default: print the experiment matrix only.
Add --run to launch the experiments (quantum simulations can be slow).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    {"name": "angle_2q_he_d1", "encoding": "angle", "ansatz": "hardware_efficient", "qubits": 2, "depth": 1},
    {"name": "angle_3q_he_d1", "encoding": "angle", "ansatz": "hardware_efficient", "qubits": 3, "depth": 1},
    {"name": "angle_4q_he_d1", "encoding": "angle", "ansatz": "hardware_efficient", "qubits": 4, "depth": 1},
    {"name": "angle_2q_he_d2", "encoding": "angle", "ansatz": "hardware_efficient", "qubits": 2, "depth": 2},
    {"name": "angle_2q_qaoa_d1", "encoding": "angle", "ansatz": "qaoa", "qubits": 2, "depth": 1},
    {"name": "amplitude_2q_he_d1", "encoding": "amplitude", "ansatz": "hardware_efficient", "qubits": 2, "depth": 1},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_limit", type=int, default=500)
    parser.add_argument("--eval_limit", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--replace_all", action="store_true",
                        help="Replace all attention layers in each ablation run.")
    parser.add_argument("--csv", default="results/ablation_matrix.csv")
    args = parser.parse_args()

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(CONFIGS[0]))
        writer.writeheader()
        writer.writerows(CONFIGS)
    print(f"Saved experiment matrix to {args.csv}")

    if args.run:
        for config in CONFIGS:
            command = [
                sys.executable, "-m", "student3_integration.train_qksan_mt",
                "--out", f"results/ablation_{config['name']}",
                "--epochs", str(args.epochs),
                "--train_limit", str(args.train_limit),
                "--eval_limit", str(args.eval_limit),
                "--batch_size", str(args.batch_size),
                "--encoding", config["encoding"],
                "--ansatz", config["ansatz"],
                "--n_qubits", str(config["qubits"]),
                "--depth", str(config["depth"]),
            ]
            if args.replace_all:
                command.append("--replace_all")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
