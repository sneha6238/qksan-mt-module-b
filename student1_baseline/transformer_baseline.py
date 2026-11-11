"""Classical encoder-decoder Transformer (baseline).

The attention block is a separate class (MultiHeadAttention) so Yanaparti can
swap it for QKSANAttention later without touching anything else.
"""
import math, torch, torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(T, device=None):
    """True above the diagonal = positions that must NOT be attended (causal)."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    """Interface: forward(query, key_value, attn_mask) -> [B, Lq, d_model]
    attn_mask: BoolTensor broadcastable to [B, 1, Lq, Lk]; True = blocked."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d_model, d_model) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None):
        B, Lq, _ = query.shape
        Lk = key_value.size(1)
        q = self.wq(query).view(B, Lq, self.h, self.dk).transpose(1, 2)
        k = self.wk(key_value).view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = self.wv(key_value).view(B, Lk, self.h, self.dk).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)       # [B,h,Lq,Lk]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d, h, ff, p, attn_cls=MultiHeadAttention):
        super().__init__()
        self.attn, self.ff = attn_cls(d, h, p), FeedForward(d, ff, p)
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, x, mask):
        x = self.n1(x + self.drop(self.attn(x, x, mask)))
        return self.n2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, h, ff, p, attn_cls=MultiHeadAttention):
        super().__init__()
        self.self_attn, self.cross_attn = attn_cls(d, h, p), attn_cls(d, h, p)
        self.ff = FeedForward(d, ff, p)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x, mem, self_mask, cross_mask):
        x = self.n1(x + self.drop(self.self_attn(x, x, self_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, mem, cross_mask)))
        return self.n3(x + self.drop(self.ff(x)))


class TransformerMT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=512,
                 dropout=0.1, max_len=512, pad_id=0, attn_cls=MultiHeadAttention):
        super().__init__()
        self.d_model, self.pad_id = d_model, pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # shared src/tgt/output
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.enc = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, attn_cls) for _ in range(n_layers)])
        self.dec = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout, attn_cls) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.emb.weight                                  # weight tying
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, ids):
        return self.pos(self.emb(ids) * math.sqrt(self.d_model))

    def encode(self, src, src_pad_mask):
        mask = src_pad_mask[:, None, None, :]                              # [B,1,1,S]
        x = self._embed(src)
        for layer in self.enc:
            x = layer(x, mask)
        return x, mask

    def decode(self, tgt_in, tgt_pad_mask, mem, cross_mask):
        T = tgt_in.size(1)
        self_mask = subsequent_mask(T, tgt_in.device)[None, None] | tgt_pad_mask[:, None, None, :]
        x = self._embed(tgt_in)
        for layer in self.dec:
            x = layer(x, mem, self_mask, cross_mask)
        return self.out(x)                                                 # logits [B,T,V]

    def forward(self, src, tgt_in, src_pad_mask, tgt_pad_mask):
        mem, cross_mask = self.encode(src, src_pad_mask)
        return self.decode(tgt_in, tgt_pad_mask, mem, cross_mask)

    @torch.no_grad()
    def greedy_decode(self, src, src_pad_mask, bos_id=2, eos_id=3, max_len=64):
        self.eval()
        mem, cross_mask = self.encode(src, src_pad_mask)
        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, ys.eq(self.pad_id), mem, cross_mask)
            nxt = logits[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, self.pad_id)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt.eq(eos_id)
            if done.all():
                break
        return ys[:, 1:]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
