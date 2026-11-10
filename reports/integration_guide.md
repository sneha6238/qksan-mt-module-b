# Module B integration guide

## API

```python
from student2_quantum import QKSANAttention

attention = QKSANAttention(
    n_qubits=2,
    encoding="angle",  # "angle" or "amplitude"
    ansatz="hardware_efficient",  # or "qaoa"
    depth=1,
)
output, raw_kernel_scores = attention(query, key, value, mask=None)
```

Inputs are already projected Q, K, and V tensors:

- Unbatched: Q `(Lq, dq)`, K `(Lk, dq)`, V `(Lk, dv)`.
- Batched: Q `(B, Lq, dq)`, K `(B, Lk, dq)`, V `(B, Lk, dv)`.
- For angle encoding, `dq <= n_qubits`; for amplitude encoding,
  `dq <= 2**n_qubits`.

Return shapes are output `(Lq, dv)` / `(B, Lq, dv)` and raw fidelity score
matrix `(Lq, Lk)` / `(B, Lq, Lk)`. A boolean `True` masks a position; a
floating mask is additive. Batched key-padding masks `(B, Lk)` and attention
masks broadcastable to the score shape are supported.

Use the module as a PyTorch `nn.Module`; its QAOA or hardware-efficient circuit
parameters appear in `parameters()` and can be passed to standard optimizers.
Run on the `default.qubit` simulator for initial experiments. Keep the
Transformer's Q/K projection small; pairwise state preparation is costly.

## Suggested initial integration

Replace one encoder self-attention layer only, as suggested in the project
outline. Keep the baseline Transformer, tokenizer, data pipeline, decoder,
and training/evaluation loop in Module A/C ownership. Compare one-layer
replacement against the baseline before expanding coverage.

The initial amplitude-state preparation is not generally differentiable
with respect to Q/K input features. This implementation also re-uploads input
features through data-dependent rotations, creating an additional gradient
path; a local gradient check confirmed nonzero Q/K and ansatz gradients for
the tested configuration. Validate gradients for the configuration you use.
Angle encoding remains the simpler starting point for end-to-end projection
training.
