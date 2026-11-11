"""BLEU, chrF (SacreBLEU) and COMET. Deterministic given fixed checkpoint + data."""
import time, torch, sacrebleu


@torch.no_grad()
def translate(model, tokenizer, sources, device, batch_size=32, max_len=64):
    from dataset import _pad
    model.eval()
    hyps, t0 = [], time.time()
    for i in range(0, len(sources), batch_size):
        chunk = sources[i:i + batch_size]
        src = _pad([tokenizer.encode(s, add_eos=True, max_len=max_len) for s in chunk]).to(device)
        out = model.greedy_decode(src, src.eq(tokenizer.pad_id), tokenizer.bos_id, tokenizer.eos_id, max_len)
        hyps += [tokenizer.decode(o.tolist()) for o in out]
    return hyps, (time.time() - t0) / max(1, len(sources))   # sec/sentence


def bleu_score(hyps, refs):
    # tokenize="none" would be wrong for Hindi; "13a" is SacreBLEU's default, fine for
    # whitespace-separated Devanagari. Use "intl" if you want to be stricter.
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def chrf_score(hyps, refs):
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def comet_score(srcs, hyps, refs, model_name="Unbabel/wmt22-comet-da", batch_size=16, gpus=0):
    """Needs: pip install unbabel-comet (downloads a ~2GB model). Optional."""
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return model.predict(data, batch_size=batch_size, gpus=gpus).system_score
