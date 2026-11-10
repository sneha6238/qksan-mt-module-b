# Testing report

## Verified run

- Python: 3.11.9 (project virtual environment)
- PyTorch: 2.15.0+cpu
- PennyLane: 0.43.3
- Device: `default.qubit`
- Command: `python -m pytest -q tests`
- Result: **25 passed**.
- Also executed successfully: encoding smoke test, kernel demo, attention demo,
  encoding comparison, qubit-count smoke ablation (2/3/4 qubits), and depth
  smoke ablation (depths 1/2/3).

The ablation scripts report synthetic fidelities and local elapsed times only;
these values are not translation quality measurements and should not be used
to claim an encoding, qubit count, or depth is superior.

## Reproduce

Run with the project's configured environment:

```powershell
python -m pytest -q
```

Tests cover encoding utilities, both ansatz families, angle and amplitude
kernel paths, masked attention, and gradient/optimizer integration.
