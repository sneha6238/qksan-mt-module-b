"""Compare synthetic angle/amplitude kernel matrices without MT claims."""

import time

import torch

from student2_quantum import QuantumKernel


def main() -> None:
    torch.manual_seed(23)
    features = torch.randn(3, 2, dtype=torch.float64)
    for encoding in ("angle", "amplitude"):
        kernel = QuantumKernel(n_qubits=2, encoding=encoding)
        start = time.perf_counter()
        scores = kernel(features, features)
        elapsed = time.perf_counter() - start
        print(
            f"encoding={encoding} mean_fidelity={scores.mean().item():.6f} "
            f"elapsed_seconds={elapsed:.3f}"
        )


if __name__ == "__main__":
    main()
