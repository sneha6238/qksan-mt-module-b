# QKSAN-MT — Module B (Quantum Attention)

Independent Module B implementation: quantum feature encoding, trainable
ansätze, fidelity kernel, and PyTorch-compatible attention over projected Q/K/V.

## Implemented

- Amplitude and angle encoding helpers; angle features are not implicitly rescaled.
- Hardware-efficient and QAOA-inspired trainable circuits.
- Exact PennyLane statevector fidelity kernel with trainable data-reuploading.
- Batched/unbatched attention scores and classical value aggregation.
- PyTorch autograd support for angle-encoded Q/K and ansatz parameters.
- Simulator tests, synthetic examples, experiment scripts, and handoff reports.

The original QKSAN work is classification-oriented. This module is an MT
research adaptation, not evidence that the source architecture already
establishes translation results. The value-register/DMP extension is outside
the current implementation. See `reports/mathematical_formulation.md` for
details and limitations.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Run tests

```powershell
python -m pytest -q
```

## Run examples

```powershell
python examples/encoding_smoke_test.py
python examples/kernel_demo.py
python examples/attention_demo.py
```

## Module B interface

`QKSANAttention.forward(query, key, value, mask=None)` accepts already
projected tensors and returns `(output, raw_kernel_scores)`. The shapes,
mask semantics, supported encoding dimensions, and integration caveats are
documented in `reports/integration_guide.md`.

For initial work use 2–4 qubits and `default.qubit`. Exact statevector
simulation is exponential and the per-token pairwise kernel is not intended for
production throughput. The initial amplitude-state preparation is not generally differentiable
through its input features. This implementation also re-uploads features
through data-dependent rotations, which provide an additional gradient path.
Validate gradients for your chosen configuration; angle encoding remains the
simpler option for end-to-end projection training.
