"""Run a small, reproducible set of QKSAN-MT configurations.

Example:
  python -m student3_integration.experiments --epochs 2 --train_limit 500
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_limit", type=int, default=500)
    parser.add_argument("--eval_limit", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--replace_all", action="store_true",
                        help="Replace all attention layers in each experiment.")
    args = parser.parse_args()

    configs = [
        ("qksan_angle_he_2q", "angle", "hardware_efficient", 2, 1),
        ("qksan_angle_he_3q", "angle", "hardware_efficient", 3, 1),
        ("qksan_angle_he_2q_d2", "angle", "hardware_efficient", 2, 2),
    ]
    for name, encoding, ansatz, qubits, depth in configs:
        command = [
            sys.executable, "-m", "student3_integration.train_qksan_mt",
            "--out", f"results/{name}",
            "--epochs", str(args.epochs),
            "--train_limit", str(args.train_limit),
            "--eval_limit", str(args.eval_limit),
            "--batch_size", str(args.batch_size),
            "--encoding", encoding,
            "--ansatz", ansatz,
            "--n_qubits", str(qubits),
            "--depth", str(depth),
        ]
        if args.replace_all:
            command.append("--replace_all")
        print("\\nRunning:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
