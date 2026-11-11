"""Shared SentencePiece tokenizer (one joint vocab for English + Hindi)."""
import os
import sentencepiece as spm

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def train_tokenizer(pairs, model_prefix="data/spm_joint", vocab_size=8000):
    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)
    txt = model_prefix + "_train.txt"
    with open(txt, "w", encoding="utf-8") as f:
        for en, hi in pairs:
            f.write(en + "\n"); f.write(hi + "\n")
    spm.SentencePieceTrainer.train(
        input=txt, model_prefix=model_prefix, vocab_size=vocab_size,
        model_type="bpe", character_coverage=1.0,      # 1.0 keeps all Devanagari chars
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        input_sentence_size=500_000, shuffle_input_sentence=True)
    os.remove(txt)
    return model_prefix + ".model"


class Tokenizer:
    def __init__(self, model_path="data/spm_joint.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.pad_id, self.bos_id, self.eos_id = PAD, BOS, EOS

    def encode(self, text, add_bos=False, add_eos=True, max_len=None):
        ids = self.sp.encode(text)
        if max_len:
            ids = ids[: max_len - int(add_bos) - int(add_eos)]
        return ([BOS] if add_bos else []) + ids + ([EOS] if add_eos else [])

    def decode(self, ids):
        ids = [i for i in ids if i not in (PAD, BOS)]
        if EOS in ids:
            ids = ids[: ids.index(EOS)]
        return self.sp.decode(ids)
