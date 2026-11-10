import pennylane as qml

from student2_quantum.quantum_encoding import (
    amplitude_encode,
    apply_amplitude_encoding,
    apply_angle_encoding,
)

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def amplitude_circuit():
    apply_amplitude_encoding([0.3, 0.4, 0.0], wires=range(n_qubits))
    return qml.state()

@qml.qnode(dev)
def angle_circuit():
    apply_angle_encoding([0.2, -0.5], wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

print("Amplitude-prepared state:", amplitude_circuit())
print("Amplitude norm:", abs((amplitude_circuit().conj() @ amplitude_circuit())) ** 0.5)
print("Angle-circuit Z expectations:", angle_circuit())
print("Expected amplitude helper output:", amplitude_encode([0.3, 0.4, 0.0], n_qubits=2))
