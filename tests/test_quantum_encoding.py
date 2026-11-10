import numpy as np
import pytest

from student2_quantum.quantum_encoding import amplitude_encode, angle_encode


def test_amplitude_encoding_normalizes_and_pads():
    state = amplitude_encode([3.0, 4.0], n_qubits=2)
    assert state.shape == (4,)
    np.testing.assert_allclose(state, [0.6, 0.8, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_amplitude_encoding_auto_selects_qubits():
    state = amplitude_encode([1.0, 2.0, 3.0])
    assert state.shape == (4,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_amplitude_encoding_rejects_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        amplitude_encode([0.0, 0.0], n_qubits=1)


def test_amplitude_encoding_rejects_oversized_input():
    with pytest.raises(ValueError, match="do not fit"):
        amplitude_encode([1, 2, 3, 4, 5], n_qubits=2)


def test_encoding_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        amplitude_encode([1.0, np.nan])


def test_angle_encoding_pads_without_rescaling():
    angles = angle_encode([0.2, -0.5], n_qubits=4)
    np.testing.assert_allclose(angles, [0.2, -0.5, 0.0, 0.0])


def test_angle_encoding_rejects_too_many_features():
    with pytest.raises(ValueError, match="do not fit"):
        angle_encode([0.1, 0.2, 0.3], n_qubits=2)
