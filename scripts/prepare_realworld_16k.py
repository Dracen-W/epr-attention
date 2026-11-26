"""
Stream and interleave 3 real-world corpora (FineWeb-Edu 45% + Wikipedia 35% + CNN/DM News 20%)
into a 210M-token binary dataset with a custom 16,384-vocab ByteLevel BPE tokenizer.
Run from repo root: python3 scripts/prepare_realworld_16k.py
"""
import os, pickle, time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

out_dir = "data/realworld_16k"
os.makedirs(out_dir, exist_ok=True)
train_bin = os.path.join(out_dir, "train.bin")
val_bin = os.path.join(out_dir, "val.bin")
tok_path = os.path.join(out_dir, "tokenizer.json")
meta_path = os.path.join(out_dir, "meta.pkl")

if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(tok_path):
    print("Real-world 16K dataset already prepared at", out_dir)
    raise SystemExit(0)

t0 = time.time()
TARGET_VAL_TOKENS = 4_000_000
TARGET_TRAIN_TOKENS = 210_000_000

print("1. Opening 3 real-world streaming datasets (FineWeb-Edu 45% + Wikipedia 35% + CNN/DM News 20%)...")
it_fw   = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
it_wiki = iter(load_dataset("wikimedia/wikipedia", name="20231101.en", split="train", streaming=True))
it_news = iter(load_dataset("abisee/cnn_dailymail", name="3.0.0", split="train", streaming=True))

def next_clean(it, key, min_chars=250):
    while True:
        txt = next(it)[key]
        if txt and len(txt) >= min_chars:
            return txt.strip()

def sample_mixed_docs(n_total):
    n_fw = int(n_total * 0.45)
    n_wiki = int(n_total * 0.35)
    n_news = n_total - n_fw - n_wiki
    docs = (
        [next_clean(it_fw, "text") for _ in range(n_fw)]
        + [next_clean(it_wiki, "text") for _ in range(n_wiki)]
        + [next_clean(it_news, "article") for _ in range(n_news)]
    )
    rng = np.random.RandomState(len(docs) + int(time.time()) % 10000)
    rng.shuffle(docs)
    return docs

print("2. Training 16,384-vocab ByteLevel BPE tokenizer on 24,000 mixed docs...")
tok_docs = sample_mixed_docs(24_000)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(tok_docs, trainer=trainers.BpeTrainer(vocab_size=16384, min_frequency=2, special_tokens=["<|endoftext|>"]))
tok.save(tok_path)
eos_id = tok.token_to_id("<|endoftext|>")
del tok_docs

def stream_write(target_tokens, bin_path, label):
    t1 = time.time()
    total = 0
    with open(bin_path, "wb") as f:
        while total < target_tokens:
            encs = tok.encode_batch(sample_mixed_docs(1500))
            arr = []
            for e in encs:
                arr.extend(e.ids)
                arr.append(eos_id)
            np.asarray(arr, dtype=np.uint16).tofile(f)
            total += len(arr)
            if total % 20_000_000 < len(arr) or total >= target_tokens:
                print(f"   [{label}] {total/1e6:6.2f}M / {target_tokens/1e6:.0f}M tokens ({time.time()-t1:.1f}s)")
    return total

n_val = stream_write(TARGET_VAL_TOKENS, val_bin, "VAL")
n_train = stream_write(TARGET_TRAIN_TOKENS, train_bin, "TRAIN")
with open(meta_path, "wb") as f:
    pickle.dump({"vocab_size": tok.get_vocab_size(), "tokenizer_type": "hf_bpe"}, f)
print(f"Done! Train: {n_train/1e6:.2f}M | Val: {n_val/1e6:.2f}M | Total: {time.time()-t0:.1f}s")
