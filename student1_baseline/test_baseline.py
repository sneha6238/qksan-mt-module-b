"""Unit tests: run with  python -m pytest -q  (from student1_baseline/)"""
import torch, pytest
from preprocessing import clean_pairs, normalize, split_pairs
from transformer_baseline import TransformerMT, subsequent_mask
from dataset import collate


def test_cleaning_removes_empty_dupes_long():
    pairs = [("hello", "नमस्ते"), ("hello", "नमस्ते"), ("", "x"), ("a " * 100, "b " * 100)]
    assert clean_pairs(pairs) == [("hello", "नमस्ते")]


def test_normalize_whitespace():
    assert normalize("  a   b \n") == "a b"


def test_split_is_deterministic_and_disjoint():
    pairs = [(str(i), str(i)) for i in range(1000)]
    a, b = split_pairs(pairs), split_pairs(pairs)
    assert a == b
    tr, va, te = a
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_collate_shapes_and_masks():
    batch = [([5, 6, 3], [7, 8]), ([5, 3], [9])]
    b = collate(batch)
    assert b["src"].shape == (2, 3) and b["tgt_in"].shape == b["tgt_out"].shape == (2, 3)
    assert b["src_pad_mask"][1].tolist() == [False, False, True]
    assert b["tgt_in"][0, 0].item() == 2 and b["tgt_out"][0, -1].item() == 3


def test_causal_mask():
    m = subsequent_mask(3)
    assert m.tolist() == [[False, True, True], [False, False, True], [False, False, False]]


def test_model_forward_and_decode():
    m = TransformerMT(100, 32, 4, 1, 64)
    b = collate([([5, 6, 3], [7, 8]), ([5, 3], [9])])
    out = m(b["src"], b["tgt_in"], b["src_pad_mask"], b["tgt_pad_mask"])
    assert out.shape == (2, 3, 100)
    assert m.greedy_decode(b["src"], b["src_pad_mask"], max_len=5).shape[0] == 2


def test_decoder_is_causal():
    """Changing a future target token must not change earlier logits."""
    torch.manual_seed(0)
    m = TransformerMT(100, 32, 4, 1, 64, dropout=0.0).eval()
    src = torch.tensor([[5, 6, 3]]); sm = src.eq(0)
    t1, t2 = torch.tensor([[2, 7, 8]]), torch.tensor([[2, 7, 9]])
    l1, l2 = m(src, t1, sm, t1.eq(0)), m(src, t2, sm, t2.eq(0))
    assert torch.allclose(l1[:, :2], l2[:, :2], atol=1e-5)
