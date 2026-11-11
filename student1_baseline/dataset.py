"""PyTorch Dataset + collate function. Outputs token IDs, masks, batches.

Batch format (agreed interface for Sneha / Yanaparti):
  src          LongTensor [B, S]   source ids (ends with EOS, PAD=0)
  tgt_in       LongTensor [B, T]   BOS + target ids          (decoder input)
  tgt_out      LongTensor [B, T]   target ids + EOS          (labels)
  src_pad_mask BoolTensor [B, S]   True where PAD
  tgt_pad_mask BoolTensor [B, T]   True where PAD
(The causal mask is built inside the model: see subsequent_mask().)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import PAD, BOS, EOS


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=64):
        self.data = []
        for en, hi in pairs:
            s = tokenizer.encode(en, add_eos=True, max_len=max_len)
            t = tokenizer.encode(hi, add_eos=False, max_len=max_len - 1)
            self.data.append((s, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def _pad(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs], dtype=torch.long)


def collate(batch):
    src = _pad([s for s, _ in batch])
    tgt_in = _pad([[BOS] + t for _, t in batch])
    tgt_out = _pad([t + [EOS] for _, t in batch])
    return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out,
            "src_pad_mask": src.eq(PAD), "tgt_pad_mask": tgt_in.eq(PAD)}


def make_loader(pairs, tokenizer, batch_size=32, max_len=64, shuffle=False):
    ds = TranslationDataset(pairs, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
