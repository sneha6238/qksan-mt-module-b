"""Trainable circuit ansätze used by QKSAN feature maps."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

from .quantum_utils import validate_n_qubits, validate_wires


def apply_hardware_efficient_ansatz(
    weights: torch.Tensor,
    wires: tuple[int, ...] | list[int],
    entangler: str = "cz",
) -> None:
    """Apply trainable rotations followed by linear entanglement.

    ``weights`` has shape ``(depth, n_qubits, 3)`` and parameterizes a
    ``Rot(phi, theta, omega)`` on every wire in every layer.
    """
    wires = validate_wires(wires)
    if len(wires) < 2:
        raise ValueError("The hardware-efficient ansatz requires at least 2 qubits.")
    if entangler not in {"cz", "cnot"}:
        raise ValueError("entangler must be 'cz' or 'cnot'.")
    if (
        weights.ndim != 3
        or weights.shape[0] < 1
        or tuple(weights.shape[1:]) != (len(wires), 3)
    ):
        raise ValueError(
            "Hardware-efficient weights must have shape "
            f"(depth, {len(wires)}, 3); got {tuple(weights.shape)}."
        )

    for layer in range(weights.shape[0]):
        for wire_index, wire in enumerate(wires):
            qml.Rot(*weights[layer, wire_index], wires=wire)
        gate = qml.CZ if entangler == "cz" else qml.CNOT
        for left, right in zip(wires, wires[1:]):
            gate(wires=[left, right])


class HardwareEfficientAnsatz(nn.Module):
    """Trainable hardware-efficient ansatz with a linear entangler layout."""

    def __init__(
        self,
        n_qubits: int,
        depth: int = 1,
        entangler: str = "cz",
    ) -> None:
        super().__init__()
        self.n_qubits = validate_n_qubits(n_qubits)
        if self.n_qubits < 2:
            raise ValueError("The hardware-efficient ansatz requires at least 2 qubits.")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer.")
        if entangler not in {"cz", "cnot"}:
            raise ValueError("entangler must be 'cz' or 'cnot'.")
        self.depth = depth
        self.entangler = entangler
        self.weights = nn.Parameter(
            0.01 * torch.randn(depth, self.n_qubits, 3, dtype=torch.float64)
        )

    def forward(self, wires: tuple[int, ...] | list[int] | None = None) -> None:
        """Queue this ansatz in an active PennyLane QNode."""
        selected_wires = tuple(range(self.n_qubits)) if wires is None else tuple(wires)
        if len(selected_wires) != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} wires, got {len(selected_wires)}.")
        apply_hardware_efficient_ansatz(self.weights, selected_wires, self.entangler)


def apply_qaoa_ansatz(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    wires: tuple[int, ...] | list[int],
    *,
    apply_initial_hadamards: bool = True,
) -> None:
    """Apply QAOA layers for a ring ZZ cost Hamiltonian and X mixer.

    The cost Hamiltonian is ``sum_i Z_i Z_(i+1)`` with periodic boundaries;
    for two qubits, the single unique edge is used. The mixer is ``sum_i X_i``.
    """
    wires = validate_wires(wires)
    if len(wires) < 2:
        raise ValueError("The QAOA ansatz requires at least 2 qubits.")
    if gamma.ndim != 1 or beta.ndim != 1 or gamma.shape != beta.shape:
        raise ValueError("gamma and beta must be 1-D tensors with the same shape.")
    if gamma.numel() < 1:
        raise ValueError("QAOA depth must be at least one.")

    if apply_initial_hadamards:
        for wire in wires:
            qml.Hadamard(wires=wire)

    edges = list(zip(wires, wires[1:]))
    if len(wires) > 2:
        edges.append((wires[-1], wires[0]))
    for layer in range(gamma.shape[0]):
        for left, right in edges:
            qml.IsingZZ(2.0 * gamma[layer], wires=[left, right])
        for wire in wires:
            qml.RX(2.0 * beta[layer], wires=wire)


class QAOAAnsatz(nn.Module):
    """Trainable QAOA-inspired cost/mixer circuit, separate from HEA."""

    def __init__(self, n_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.n_qubits = validate_n_qubits(n_qubits)
        if self.n_qubits < 2:
            raise ValueError("The QAOA ansatz requires at least 2 qubits.")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer.")
        self.depth = depth
        self.gamma = nn.Parameter(0.01 * torch.randn(depth, dtype=torch.float64))
        self.beta = nn.Parameter(0.01 * torch.randn(depth, dtype=torch.float64))

    def forward(
        self,
        wires: tuple[int, ...] | list[int] | None = None,
        *,
        apply_initial_hadamards: bool = True,
    ) -> None:
        """Queue the QAOA-inspired ansatz in an active PennyLane QNode."""
        selected_wires = tuple(range(self.n_qubits)) if wires is None else tuple(wires)
        if len(selected_wires) != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} wires, got {len(selected_wires)}.")
        apply_qaoa_ansatz(
            self.gamma,
            self.beta,
            selected_wires,
            apply_initial_hadamards=apply_initial_hadamards,
        )