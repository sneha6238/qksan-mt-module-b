"""Build a staged QKSAN-MT model using the supplied Module A architecture.

By default only the first encoder self-attention is replaced, matching the
research plan's recommended first integration stage. Set replace_all=True
for the later full encoder/decoder self- and cross-attention experiment.
"""
from pathlib import Path
import sys

# Module A is kept in its own folder; make its module importable when Module C
# is imported directly (for example, by pytest from the repository root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_DIR = _REPO_ROOT / "student1_baseline"
if str(_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_DIR))

from transformer_baseline import TransformerMT
from .integration import QKSANMultiHeadAttention


def build_qksan_transformer(
    vocab_size: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 512,
    dropout: float = 0.1,
    max_len: int = 512,
    pad_id: int = 0,
    n_qubits: int = 2,
    encoding: str = "angle",
    ansatz: str = "hardware_efficient",
    depth: int = 1,
    replace_all: bool = False,
) -> TransformerMT:
    model = TransformerMT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pad_id=pad_id,
    )

    def new_quantum_attention():
        return QKSANMultiHeadAttention(
            d_model, n_heads, dropout,
            n_qubits=n_qubits,
            encoding=encoding,
            ansatz=ansatz,
            depth=depth,
        )

    if replace_all:
        for layer in model.enc:
            layer.attn = new_quantum_attention()
        for layer in model.dec:
            layer.self_attn = new_quantum_attention()
            layer.cross_attn = new_quantum_attention()
    else:
        # Stage 1: quantum attention in encoder layer 0 only.
        model.enc[0].attn = new_quantum_attention()
    return model
