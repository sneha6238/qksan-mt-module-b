import torch

from student2_quantum.qksan_attention import QKSANAttention
from student2_quantum.quantum_kernel import QuantumKernel


def test_angle_kernel_propagates_gradients_to_features_and_ansatz():
    torch.manual_seed(11)
    kernel = QuantumKernel(n_qubits=2, encoding="angle", depth=2)
    query = torch.tensor([[0.21, 0.43]], dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[0.8, -0.2]], dtype=torch.float64, requires_grad=True)

    loss = kernel(query, key).sum()
    loss.backward()

    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert kernel.weights.grad is not None
    assert torch.isfinite(kernel.weights.grad).all()
    assert kernel.weights.grad.norm() > 0


def test_attention_value_and_quantum_parameters_are_trainable():
    torch.manual_seed(5)
    attention = QKSANAttention(n_qubits=2, depth=2)
    query = torch.randn(2, 2, requires_grad=True)
    key = torch.randn(3, 2, requires_grad=True)
    value = torch.randn(3, 4, requires_grad=True)

    output, _ = attention(query, key, value)
    output.square().sum().backward()

    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert attention.kernel.weights.grad is not None
    assert attention.kernel.weights.grad.norm() > 0


def test_qaoa_kernel_parameters_receive_gradients():
    torch.manual_seed(19)
    kernel = QuantumKernel(n_qubits=2, ansatz="qaoa", depth=2)
    query = torch.tensor([[0.2, 0.6]], dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[-0.4, 0.7]], dtype=torch.float64)

    kernel(query, key).sum().backward()

    assert kernel.gamma.grad is not None and torch.isfinite(kernel.gamma.grad).all()
    assert kernel.beta.grad is not None and torch.isfinite(kernel.beta.grad).all()
    assert kernel.gamma.grad.norm() + kernel.beta.grad.norm() > 0
