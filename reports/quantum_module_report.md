# Module B — Quantum Module Experimental Report

## Implementation status

Implemented amplitude and angle input preparation, hardware-efficient and
QAOA-inspired ansätze, an exact simulator fidelity kernel, and batched or
unbatched PyTorch attention over projected Q/K/V. The test suite records
correctness checks; no machine-translation quality or speed claims are made.

## Environment and test results

Environment versions and the final pass/fail count are recorded after running
the suite. See [testing_report.md](./testing_report.md) for executable results.
The simulator target is PennyLane `default.qubit`.

## Experiment status

The automated tests exercise encoding and ansatz combinations, kernel matrix
range/shape, attention output shape and masking, and gradient-based optimizer
updates. No translation dataset is included in this standalone Module B
scaffold; comparative BLEU or model-performance experiments remain outside its
scope.

## Known limitations

- Exact statevector overlap scales exponentially and is intended for small
  circuits only.
- The initial amplitude-state preparation is not generally differentiable
  with respect to input features. The implementation also re-uploads features
  through data-dependent rotations, providing an additional gradient path.
  A local test confirmed nonzero Q/K and ansatz gradients for the tested
  configuration.
- The DMP conditional value-register extension is not implemented.
- This is a research adaptation of classification QKSAN to MT, not a result
  demonstrated by the source paper.
