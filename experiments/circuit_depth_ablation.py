"""Run a synthetic simulator smoke comparison across circuit depths."""

import torch

from student2_quantum import QuantumKernel


def main() -> None:
    torch.manual_seed(31)
    features = torch.randn(2, 2, dtype=torch.float64)
    for depth in (1, 2, 3):
        kernel = QuantumKernel(
            n_qubits=2,
            encoding="angle",
            ansatz="hardware_efficient",
            depth=depth,
        )
        scores = kernel(features, features)
        print(
            f"depth={depth} matrix_shape={tuple(scores.shape)} "
            f"mean_fidelity={scores.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
