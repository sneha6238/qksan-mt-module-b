"""Adapter that plugs Module B QKSAN attention into Surbhi's Transformer API.

The baseline expects attention modules with:
    __init__(d_model, n_heads, dropout)
    forward(query, key_value, attn_mask) -> [batch, query_len, d_model]

Module B expects projected Q/K/V. This adapter performs per-head classical
projections, compresses Q/K to n_qubits, calls QKSANAttention, and combines
the classical value outputs.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from student2_quantum import QKSANAttention


class QKSANMultiHeadAttention(nn.Module):
    """Multi-head wrapper around the supplied Module B QKSANAttention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        *,
        n_qubits: int = 2,
        encoding: str = "angle",
        ansatz: str = "hardware_efficient",
        depth: int = 1,
        entangler: str = "cz",
        device_name: str = "default.qubit",
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if encoding == "angle" and n_qubits < 2:
            raise ValueError("Module B requires at least 2 qubits.")
        if encoding not in {"angle", "amplitude"}:
            raise ValueError("encoding must be 'angle' or 'amplitude'.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_qubits = n_qubits

        # Q/K are compressed to a small quantum feature vector.
        self.wq = nn.Linear(d_model, n_heads * n_qubits)
        self.wk = nn.Linear(d_model, n_heads * n_qubits)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Independent trainable quantum kernel parameters for each head.
        self.quantum_heads = nn.ModuleList([
            QKSANAttention(
                n_qubits=n_qubits,
                encoding=encoding,
                ansatz=ansatz,
                depth=depth,
                entangler=entangler,
                device_name=device_name,
            )
            for _ in range(n_heads)
        ])

    def _head_mask(self, mask: torch.Tensor | None, head: int):
        if mask is None:
            return None
        # Baseline masks are normally [B, 1, 1|Lq, Lk].
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                return mask[:, 0]
            return mask[:, head]
        if mask.ndim == 3:
            return mask
        if mask.ndim == 2:
            return mask
        raise ValueError(f"Unsupported attention mask rank: {mask.ndim}")

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("query and key_value must have shape [B, L, d_model].")
        batch, q_len, _ = query.shape
        k_len = key_value.shape[1]

        q = self.wq(query).view(batch, q_len, self.n_heads, self.n_qubits)
        k = self.wk(key_value).view(batch, k_len, self.n_heads, self.n_qubits)
        v = self.wv(key_value).view(batch, k_len, self.n_heads, self.head_dim)

        # Bound angle inputs; rotations are periodic and this stabilizes scale.
        q = math.pi * torch.tanh(q)
        k = math.pi * torch.tanh(k)

        head_outputs = []
        for h, quantum_attention in enumerate(self.quantum_heads):
            qh = q[:, :, h, :]
            kh = k[:, :, h, :]
            vh = v[:, :, h, :]
            out, _raw_scores = quantum_attention(
                qh, kh, vh, mask=self._head_mask(attn_mask, h)
            )
            head_outputs.append(out)

        merged = torch.cat(head_outputs, dim=-1)
        return self.wo(self.dropout(merged))
