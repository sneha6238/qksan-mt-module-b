import torch

from student2_quantum import QKSANAttention


def test_attention_can_be_optimized_with_a_torch_optimizer():
    torch.manual_seed(13)
    module = QKSANAttention(n_qubits=2, encoding="angle", depth=1)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    query = torch.randn(2, 2)
    key = torch.randn(3, 2)
    value = torch.randn(3, 4)

    before = [parameter.detach().clone() for parameter in module.parameters()]
    output, scores = module(query, key, value)
    loss = output.square().mean() + scores.mean()
    loss.backward()
    optimizer.step()

    assert output.shape == (2, 4)
    assert scores.shape == (2, 3)
    assert any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))
