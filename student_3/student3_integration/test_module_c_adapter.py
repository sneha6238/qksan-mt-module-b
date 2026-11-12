"""Small simulator tests for Module C's Transformer adapter."""
import torch

from student3_integration.integration import QKSANMultiHeadAttention
from student3_integration.qksan_transformer import build_qksan_transformer


def test_adapter_matches_baseline_attention_interface():
    torch.manual_seed(7)
    module = QKSANMultiHeadAttention(
        d_model=8, n_heads=2, dropout=0.0, n_qubits=2, depth=1
    )
    query = torch.randn(1, 3, 8)
    key_value = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 1, 1, 4, dtype=torch.bool)
    output = module(query, key_value, mask)
    assert output.shape == (1, 3, 8)
    assert torch.isfinite(output).all()


def test_qksan_model_replaces_only_first_encoder_attention_by_default():
    model = build_qksan_transformer(
        vocab_size=20, d_model=8, n_heads=2, n_layers=2,
        d_ff=16, dropout=0.0, n_qubits=2,
    )
    assert isinstance(model.enc[0].attn, QKSANMultiHeadAttention)
    assert not isinstance(model.enc[1].attn, QKSANMultiHeadAttention)
    assert not isinstance(model.dec[0].self_attn, QKSANMultiHeadAttention)


def test_full_replacement_option():
    model = build_qksan_transformer(
        vocab_size=20, d_model=8, n_heads=2, n_layers=1,
        d_ff=16, dropout=0.0, n_qubits=2, replace_all=True,
    )
    assert isinstance(model.enc[0].attn, QKSANMultiHeadAttention)
    assert isinstance(model.dec[0].self_attn, QKSANMultiHeadAttention)
    assert isinstance(model.dec[0].cross_attn, QKSANMultiHeadAttention)
