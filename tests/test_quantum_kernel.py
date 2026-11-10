import pytest
import torch

from student2_quantum.quantum_kernel import QuantumKernel


@pytest.mark.parametrize("encoding", ["angle", "amplitude"])
@pytest.mark.parametrize("ansatz", ["hardware_efficient", "qaoa"])
def test_kernel_matrix_shape_range_and_diagonal(encoding, ansatz):
    torch.manual_seed(7)
    kernel = QuantumKernel(n_qubits=2, encoding=encoding, ansatz=ansatz, depth=1)
    features = torch.tensor([[0.2, 0.4], [0.7, -0.3]], dtype=torch.float64)

    matrix = kernel(features, features)

    assert matrix.shape == (2, 2)
    assert torch.all((matrix >= 0.0) & (matrix <= 1.0))
    torch.testing.assert_close(torch.diagonal(matrix), torch.ones(2, dtype=matrix.dtype))
    torch.testing.assert_close(matrix, matrix.T, atol=1e-8, rtol=1e-8)


def test_amplitude_kernel_rejects_zero_vector():
    kernel = QuantumKernel(2, encoding="amplitude")
    with pytest.raises(ValueError, match="zero vector"):
        kernel(torch.zeros(1, 2), torch.ones(1, 2))


def test_kernel_rejects_input_over_capacity():
    kernel = QuantumKernel(2, encoding="angle")
    with pytest.raises(ValueError, match="capacity"):
        kernel(torch.ones(1, 3), torch.ones(1, 3))


def test_kernel_rejects_mismatched_query_and_key_dimensions():
    kernel = QuantumKernel(2, encoding="angle")
    with pytest.raises(ValueError, match="dimensions must match"):
        kernel(torch.ones(1, 1), torch.ones(1, 2))
