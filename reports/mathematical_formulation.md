# Module B mathematical formulation

## Classical projected interface

The Transformer or its integration layer supplies projected tensors
`Q' in R^(B x Lq x dq)`, `K' in R^(B x Lk x dq)`, and
`V' in R^(B x Lk x dv)`. This module intentionally does not implement the
Transformer projections or encoder/decoder.

## Quantum feature state and kernel

For each projected query/key vector `x`, the simulator prepares a state
`|phi_theta(x)>` by feature encoding interleaved with a trainable ansatz. The
returned quantum kernel is the squared fidelity

`S_ij = |<phi_theta(q_i)|phi_theta(k_j)>|^2`.

The current simulator implementation constructs the statevectors and evaluates
their exact inner products. This is equivalent to an ideal compute-uncompute
fidelity measurement, but does not estimate finite-shot noise or provide a
hardware execution path.

The trainable ansatz is interleaved with data re-uploading. A shared unitary
appended after both inputs would cancel in the state overlap and would not
provide trainable kernel parameters.

## Attention output

The raw scores are normalized across keys:

`A_ij = softmax_j(S_ij + M_ij)`, and `O_i = sum_j A_ij V'_j`,

where `M` is an optional additive or boolean exclusion mask. Kernel scores
already lie in `[0, 1]`; no Transformer `1/sqrt(d)` scaling is applied.

## Research adaptations and limitations

- The QKSAN source architecture is being adapted to sequence attention; this
  implementation must not be presented as a result already established by the
  original classification paper.
- Module B uses quantum fidelity for Q/K similarity and classical weighted
  values. It does not implement the paper's second value register or
  Deferred-Measurement-Principle conditional connection.
- Amplitude encoding prepares a normalized vector, then uses data-dependent
  rotations between ansatz layers because state preparation is valid only at
  the beginning of a PennyLane circuit. Angle encoding uses unscaled RX angles.
- The initial PennyLane amplitude-state preparation is not generally
  differentiable with respect to its input features. This implementation also
  re-uploads features through data-dependent rotations, providing an additional
  gradient path. A local test confirmed nonzero Q/K and ansatz gradients for
  the tested configuration; validate this for other configurations. Angle
  encoding remains the simpler choice for end-to-end Q/K projection training.
- The exact statevector simulator is exponential in qubit count and is intended
  for small prototypes (initially 2–4 qubits), not production MT throughput.
- The professor's pasted project outline was available in this workspace; the
  referenced full PDF was not present in the starter directory.
