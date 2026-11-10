"""Trainable fidelity quantum kernels implemented with PennyLane simulators."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from torch.nn import functional as F

from .quantum_ansatz import apply_hardware_efficient_ansatz
from .quantum_utils import validate_n_qubits


class QuantumKernel(nn.Module):
    """Squared state-overlap kernel with a trainable data-reuploading map.

    The feature map re-encodes each input between ansatz layers and once after
    the final layer. A shared trainable unitary appended only at the end of
    both states would cancel from their fidelity, so trainable layers are
    interleaved with data encoding.

    Input tensors have shape ``(items, features)``. The output is the fidelity
    matrix ``|<phi(q)|phi(k)>|**2`` with shape ``(n_query, n_key)``.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        *,
        encoding: str = "angle",
        ansatz: str = "hardware_efficient",
        depth: int = 1,
        entangler: str = "cz",
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = validate_n_qubits(n_qubits)
        if self.n_qubits < 2:
            raise ValueError("QuantumKernel requires at least 2 qubits.")
        if encoding not in {"angle", "amplitude"}:
            raise ValueError("encoding must be 'angle' or 'amplitude'.")
        if ansatz not in {"hardware_efficient", "qaoa"}:
            raise ValueError("ansatz must be 'hardware_efficient' or 'qaoa'.")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer.")
        if entangler not in {"cz", "cnot"}:
            raise ValueError("entangler must be 'cz' or 'cnot'.")
        self.encoding = encoding
        self.ansatz_name = ansatz
        self.depth = depth
        self.entangler = entangler

        if ansatz == "hardware_efficient":
            self.weights = nn.Parameter(
                0.01 * torch.randn(depth, self.n_qubits, 3, dtype=torch.float64)
            )
        else:
            self.gamma = nn.Parameter(0.01 * torch.randn(depth, dtype=torch.float64))
            self.beta = nn.Parameter(0.01 * torch.randn(depth, dtype=torch.float64))

        device = qml.device(device_name, wires=self.n_qubits)
        wires = tuple(range(self.n_qubits))

        def encode(features: torch.Tensor) -> None:
            if self.encoding == "angle":
                angles = F.pad(features, (0, self.n_qubits - features.shape[0]))
                for angle, wire in zip(angles, wires):
                    qml.RX(angle, wires=wire)
            else:
                angles = F.pad(features[: self.n_qubits], (0, max(0, self.n_qubits - features.shape[0])))
                for angle, wire in zip(angles, wires):
                    qml.RY(angle, wires=wire)

        def apply_layer(layer: int) -> None:
            if self.ansatz_name == "hardware_efficient":
                apply_hardware_efficient_ansatz(
                    self.weights[layer : layer + 1], wires, self.entangler
                )
            else:
                edges = list(zip(wires, wires[1:]))
                if self.n_qubits > 2:
                    edges.append((wires[-1], wires[0]))
                for left, right in edges:
                    qml.IsingZZ(2.0 * self.gamma[layer], wires=[left, right])
                for wire in wires:
                    qml.RX(2.0 * self.beta[layer], wires=wire)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def state_circuit(features: torch.Tensor) -> torch.Tensor:
            if self.encoding == "amplitude":
                state = F.pad(features, (0, 2**self.n_qubits - features.shape[0]))
                state = state / torch.linalg.vector_norm(state)
                qml.AmplitudeEmbedding(state, wires=wires, normalize=False)
            if self.ansatz_name == "qaoa":
                for wire in wires:
                    qml.Hadamard(wires=wire)
            for layer in range(self.depth):
                encode(features)
                apply_layer(layer)
            encode(features)
            return qml.state()

        self._state_circuit = state_circuit

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float64)
        if features.ndim != 2:
            raise ValueError(
                f"Expected a 2-D feature tensor (items, features), got {tuple(features.shape)}."
            )
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("Feature tensors must have non-empty item and feature dimensions.")
        capacity = self.n_qubits if self.encoding == "angle" else 2**self.n_qubits
        if features.shape[1] > capacity:
            raise ValueError(
                f"{features.shape[1]} features exceed {self.encoding} encoding capacity "
                f"{capacity}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("Features must contain only finite values.")
        features = features.to(dtype=torch.float64)
        if self.encoding == "amplitude" and torch.any(
            torch.linalg.vector_norm(features, dim=-1) == 0
        ):
            raise ValueError("Amplitude encoding is undefined for a zero vector.")
        return features

    def state(self, features: torch.Tensor) -> torch.Tensor:
        """Return the encoded state for each row in a 2-D feature tensor."""
        prepared = self._prepare_features(features)
        return torch.stack([self._state_circuit(row) for row in prepared])

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared state-overlap similarities."""
        if query.ndim != 2 or key.ndim != 2:
            raise ValueError("query and key must be 2-D tensors (items, features).")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("Query and key feature dimensions must match.")
        query_states = self.state(query)
        key_states = self.state(key)
        overlaps = query_states.conj() @ key_states.transpose(0, 1)
        return overlaps.abs().square().clamp(min=0.0, max=1.0)


__all__ = ["QuantumKernel"]