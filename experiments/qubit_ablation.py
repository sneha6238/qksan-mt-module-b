"""Run a synthetic simulator smoke comparison across small qubit counts."""

import torch

from student2_quantum import QuantumKernel


def main() -> None:
    torch.manual_seed(29)
    for n_qubits in (2, 3, 4):
        features = torch.randn(2, n_qubits, dtype=torch.float64)
        kernel = QuantumKernel(n_qubits=n_qubits, encoding="angle")
        scores = kernel(features, features)
        print(
            f"n_qubits={n_qubits} matrix_shape={tuple(scores.shape)} "
            f"mean_fidelity={scores.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
