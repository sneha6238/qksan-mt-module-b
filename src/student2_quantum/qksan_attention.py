"""PyTorch-compatible quantum-kernel self-attention for Module B integration."""

from __future__ import annotations

import torch
from torch import nn

from .quantum_kernel import QuantumKernel
from .quantum_utils import validate_n_qubits


class QKSANAttention(nn.Module):
    """Attention with quantum fidelity scores and classical values.

    The caller supplies already projected query, key, and value tensors. Inputs
    may be unbatched ``(length, features)`` or batched ``(batch, length,
    features)``. The return is always ``(output, raw_kernel_scores)``.

    Fidelity scores are normalized with softmax over keys; unlike scaled
    dot-product attention, they are bounded in ``[0, 1]`` and are not divided by
    ``sqrt(d)``. Values remain classical in this Module B implementation.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        *,
        encoding: str = "angle",
        ansatz: str = "hardware_efficient",
        depth: int = 1,
        entangler: str = "cz",
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = validate_n_qubits(n_qubits)
        self.kernel = QuantumKernel(
            n_qubits=self.n_qubits,
            encoding=encoding,
            ansatz=ansatz,
            depth=depth,
            entangler=entangler,
            device_name=device_name,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted classical values and unnormalized kernel scores.

        Boolean masks use ``True`` for excluded query-key positions, matching
        PyTorch's Transformer convention. A floating mask is added to raw
        kernel scores before softmax.
        """
        if query.ndim not in {2, 3} or key.ndim != query.ndim or value.ndim != query.ndim:
            raise ValueError("query, key, and value must all be rank 2 or all be rank 3.")
        if query.ndim == 3 and query.shape[0] == 0:
            raise ValueError("Batch size must be non-empty.")
        if query.shape[-1] == 0 or key.shape[-1] == 0 or value.shape[-1] == 0:
            raise ValueError("Query, key, and value feature dimensions must be non-empty.")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("Query and key feature dimensions must match.")
        if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
            raise ValueError("Batch dimensions of query, key, and value must match.")
        if query.shape[-2] == 0 or key.shape[-2] == 0 or value.shape[-2] == 0:
            raise ValueError("Query, key, and value sequence lengths must be non-empty.")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("Key and value sequence lengths must match.")
        if not query.is_floating_point() or not key.is_floating_point() or not value.is_floating_point():
            raise ValueError("Query, key, and value tensors must use floating-point dtypes.")

        unbatched = query.ndim == 2
        if unbatched:
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)

        scores = torch.stack([self.kernel(q, k) for q, k in zip(query, key)])
        scores = scores.to(dtype=value.dtype)
        if mask is not None:
            mask = torch.as_tensor(mask, device=scores.device)
            if (
                scores.ndim == 3
                and mask.ndim == 2
                and mask.shape == (scores.shape[0], scores.shape[-1])
            ):
                mask = mask.unsqueeze(1)
            try:
                mask = torch.broadcast_to(mask, scores.shape)
            except RuntimeError as error:
                raise ValueError(
                    f"mask with shape {tuple(mask.shape)} is not broadcastable to "
                    f"attention scores {tuple(scores.shape)}."
                ) from error
            if mask.dtype == torch.bool:
                if mask.all(dim=-1).any():
                    raise ValueError("Every query must retain at least one unmasked key.")
                scores = scores.masked_fill(mask, -torch.inf)
            else:
                if not mask.is_floating_point():
                    raise ValueError("Masks must be boolean or floating-point tensors.")
                if torch.isnan(mask).any() or torch.isposinf(mask).any():
                    raise ValueError("Floating masks must contain finite values or -inf.")
                scores = scores + mask.to(dtype=scores.dtype)
                if torch.isneginf(scores).all(dim=-1).any():
                    raise ValueError("Every query must retain at least one unmasked key.")

        attention = torch.softmax(scores, dim=-1)
        output = attention @ value
        if unbatched:
            return output.squeeze(0), scores.squeeze(0)
        return output, scores


__all__ = ["QKSANAttention"]