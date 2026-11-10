"""Small simulator demo for a trainable quantum fidelity kernel."""

import torch

from student2_quantum import QuantumKernel


torch.manual_seed(17)
kernel = QuantumKernel(n_qubits=2, encoding="angle", ansatz="hardware_efficient")
query = torch.tensor([[0.1, 0.4], [0.7, -0.2]], dtype=torch.float64)
key = torch.tensor([[0.2, 0.3], [-0.5, 0.8]], dtype=torch.float64)
print(kernel(query, key))
