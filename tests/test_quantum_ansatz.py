import pennylane as qml
import pytest
import torch

from student2_quantum.quantum_ansatz import HardwareEfficientAnsatz, QAOAAnsatz


@pytest.mark.parametrize(
    ("ansatz_type", "expected_parameters"),
    [(HardwareEfficientAnsatz, 1), (QAOAAnsatz, 2)],
)
def test_ansatz_executes_and_exposes_trainable_parameters(ansatz_type, expected_parameters):
    ansatz = ansatz_type(n_qubits=2, depth=2)
    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit():
        ansatz()
        return qml.state()

    state = circuit()
    assert state.shape == (4,)
    torch.testing.assert_close(torch.linalg.vector_norm(state), torch.tensor(1.0, dtype=state.real.dtype))
    assert len(list(ansatz.parameters())) == expected_parameters


def test_ansatz_rejects_invalid_depth():
    with pytest.raises(ValueError, match="depth"):
        HardwareEfficientAnsatz(n_qubits=2, depth=0)
    with pytest.raises(ValueError, match="depth"):
        QAOAAnsatz(n_qubits=2, depth=0)
