"""Run QKSAN attention on synthetic projected query/key/value tensors."""

import torch

from student2_quantum import QKSANAttention


torch.manual_seed(17)
attention = QKSANAttention(n_qubits=2, encoding="angle", ansatz="hardware_efficient")
query = torch.randn(3, 2)
key = torch.randn(4, 2)
value = torch.randn(4, 6)
output, quantum_scores = attention(query, key, value)
print("Output:", output.shape)
print("Raw fidelity scores:", quantum_scores)
