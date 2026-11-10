import pytest
import torch

from student2_quantum.qksan_attention import QKSANAttention


def test_unbatched_attention_returns_output_and_raw_kernel_scores():
    torch.manual_seed(3)
    attention = QKSANAttention(n_qubits=2)
    query = torch.randn(3, 2)
    key = torch.randn(4, 2)
    value = torch.randn(4, 5)

    output, scores = attention(query, key, value)

    assert output.shape == (3, 5)
    assert scores.shape == (3, 4)
    assert torch.isfinite(output).all()
    assert torch.isfinite(scores).all()


def test_batched_attention_supports_boolean_key_mask():
    attention = QKSANAttention(n_qubits=2)
    query = torch.randn(2, 3, 2)
    key = torch.randn(2, 4, 2)
    value = torch.randn(2, 4, 3)
    mask = torch.tensor([[False, False, True, True]])

    output, scores = attention(query, key, value, mask=mask)

    assert output.shape == (2, 3, 3)
    assert scores.shape == (2, 3, 4)
    assert torch.isneginf(scores[..., 2:]).all()


def test_batched_attention_accepts_standard_batch_key_padding_mask():
    attention = QKSANAttention(n_qubits=2)
    query = torch.randn(2, 2, 2)
    key = torch.randn(2, 3, 2)
    value = torch.randn(2, 3, 2)
    mask = torch.tensor([[False, False, True], [False, True, True]])

    _, scores = attention(query, key, value, mask=mask)

    assert torch.isneginf(scores[0, :, 2]).all()
    assert torch.isneginf(scores[1, :, 1:]).all()


def test_attention_rejects_all_masked_query_and_shape_mismatch():
    attention = QKSANAttention(n_qubits=2)
    query = torch.randn(2, 2)
    key = torch.randn(2, 2)
    value = torch.randn(2, 3)
    with pytest.raises(ValueError, match="at least one unmasked"):
        attention(query, key, value, torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="lengths"):
        attention(query, key, torch.randn(3, 3))
