"""Shared validation helpers for the quantum attention module."""

from __future__ import annotations

from collections.abc import Sequence


def validate_n_qubits(n_qubits: int) -> int:
    """Validate a positive qubit count and return it unchanged."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")
    return n_qubits


def validate_wires(wires: Sequence[int]) -> tuple[int, ...]:
    """Return unique, non-empty wire labels as a tuple."""
    result = tuple(wires)
    if not result:
        raise ValueError("At least one wire is required.")
    if len(set(result)) != len(result):
        raise ValueError("Wire labels must be unique.")
    return result