"""Input preparation and PennyLane operations for quantum feature encoding.

Amplitude encoding maps a real vector to a normalized statevector. For n
qubits, the statevector has 2**n amplitudes; shorter inputs are zero-padded.

Angle encoding maps one scalar feature to each wire using an RX rotation.
The input is not automatically rescaled: callers should choose/record a
consistent feature-scaling policy for experiments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml

from .quantum_utils import validate_wires


def _as_1d_float_array(features: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert input to a finite, non-empty 1-D float array."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1-D feature vector, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError("Feature vector must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Feature vector must contain only finite values.")
    return values


def amplitude_encode(
    features: Sequence[float] | np.ndarray,
    n_qubits: int | None = None,
) -> np.ndarray:
    """Pad and L2-normalize features for amplitude encoding.

    Args:
        features: A finite, non-empty, one-dimensional real vector.
        n_qubits: Number of qubits. If omitted, the smallest number capable
            of holding the vector is selected.

    Returns:
        A normalized NumPy vector of length ``2**n_qubits``.

    Raises:
        ValueError: If the vector is zero, too long, or n_qubits is invalid.
    """
    values = _as_1d_float_array(features)

    if n_qubits is None:
        n_qubits = max(1, int(np.ceil(np.log2(values.size))))
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")

    state_size = 2**n_qubits
    if values.size > state_size:
        raise ValueError(
            f"{values.size} features do not fit in {n_qubits} qubits "
            f"(capacity {state_size})."
        )

    padded = np.zeros(state_size, dtype=np.float64)
    padded[: values.size] = values

    norm = np.linalg.norm(padded)
    if np.isclose(norm, 0.0):
        raise ValueError("Amplitude encoding is undefined for a zero vector.")

    return padded / norm


def angle_encode(
    features: Sequence[float] | np.ndarray,
    n_qubits: int | None = None,
) -> np.ndarray:
    """Prepare one rotation angle per qubit, padding unused wires with zero.

    This helper deliberately does not normalize or rescale the angles.
    """
    values = _as_1d_float_array(features)

    if n_qubits is None:
        n_qubits = max(1, values.size)
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")
    if values.size > n_qubits:
        raise ValueError(
            f"{values.size} features do not fit on {n_qubits} qubits."
        )

    angles = np.zeros(n_qubits, dtype=np.float64)
    angles[: values.size] = values
    return angles


def apply_amplitude_encoding(
    features: Sequence[float] | np.ndarray,
    wires: Sequence[int],
) -> None:
    """Apply PennyLane amplitude state preparation on the supplied wires."""
    wires = validate_wires(wires)
    state = amplitude_encode(features, n_qubits=len(wires))
    qml.AmplitudeEmbedding(state, wires=wires, normalize=False)


def apply_angle_encoding(
    features: Sequence[float] | np.ndarray,
    wires: Sequence[int],
) -> None:
    """Apply RX angle encoding, one feature per wire."""
    wires = validate_wires(wires)
    angles = angle_encode(features, n_qubits=len(wires))
    for angle, wire in zip(angles, wires):
        qml.RX(float(angle), wires=wire)
