"""Download, clean and split the English-Hindi corpus."""
import os, random, re, unicodedata, argparse

SEED = 42


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_pairs(pairs, max_words=60, min_words=1, ratio=3.0):
    """Remove empty/duplicate/too-long/badly-aligned pairs."""
    seen, out = set(), []
    for en, hi in pairs:
        en, hi = normalize(en), normalize(hi)
        if not en or not hi:
            continue
        ne, nh = len(en.split()), len(hi.split())
        if not (min_words <= ne <= max_words and min_words <= nh <= max_words):
            continue
        if max(ne, nh) / max(1, min(ne, nh)) > ratio:   # length-ratio filter
            continue
        if (en, hi) in seen:
            continue
        seen.add((en, hi))
        out.append((en, hi))
    return out


def load_samanantar_subset(n=50_000, buffer_size=100_000, seed=SEED):
    """Stream Samanantar (hi) and take a shuffled subset of ~n raw pairs."""
    from datasets import load_dataset
    ds = load_dataset("ai4bharat/samanantar", "hi", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    pairs = []
    for i, row in enumerate(ds):
        pairs.append((row["src"], row["tgt"]))   # src=English, tgt=Hindi
        if i + 1 >= n:
            break
    return pairs


def split_pairs(pairs, val=0.05, test=0.05, seed=SEED):
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    nv, nt = int(n * val), int(n * test)
    return pairs[nv + nt:], pairs[:nv], pairs[nv:nv + nt]   # train, val, test


def write_split(pairs, folder, name):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}.en"), "w", encoding="utf-8") as fe, \
         open(os.path.join(folder, f"{name}.hi"), "w", encoding="utf-8") as fh:
        for en, hi in pairs:
            fe.write(en + "\n"); fh.write(hi + "\n")


def read_split(folder, name):
    with open(os.path.join(folder, f"{name}.en"), encoding="utf-8") as fe, \
         open(os.path.join(folder, f"{name}.hi"), encoding="utf-8") as fh:
        return [(e.rstrip("\n"), h.rstrip("\n")) for e, h in zip(fe, fh)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000, help="raw pairs to pull")
    ap.add_argument("--out", default="data")
    ap.add_argument("--local_en"); ap.add_argument("--local_hi")
    a = ap.parse_args()

    if a.local_en and a.local_hi:   # offline option: your own parallel text files
        with open(a.local_en, encoding="utf-8") as f1, open(a.local_hi, encoding="utf-8") as f2:
            raw = list(zip(f1.read().splitlines(), f2.read().splitlines()))
    else:
        raw = load_samanantar_subset(a.n)

    cleaned = clean_pairs(raw)
    print(f"raw={len(raw)}  after cleaning={len(cleaned)}")
    train, val, test = split_pairs(cleaned)
    for name, d in [("train", train), ("validation", val), ("test", test)]:
        write_split(d, os.path.join(a.out, name), name)
        print(name, len(d))
