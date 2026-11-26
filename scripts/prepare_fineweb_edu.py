"""
Stream FineWeb-Edu (sample-10BT) into a 260M-token dataset with 8,192-vocab ByteLevel BPE.
Run from repo root: python3 scripts/prepare_fineweb_edu.py
"""
import os, pickle, time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

out_dir = "data/fineweb_edu_8k"
os.makedirs(out_dir, exist_ok=True)
train_bin, val_bin = os.path.join(out_dir, "train.bin"), os.path.join(out_dir, "val.bin")
tok_path, meta_path = os.path.join(out_dir, "tokenizer.json"), os.path.join(out_dir, "meta.pkl")

if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(tok_path):
    print("FineWeb-Edu 8K dataset already prepared at", out_dir)
    raise SystemExit(0)

t0 = time.time()
it = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
tok_docs = [next(it)["text"] for _ in range(25_000)]
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(tok_docs, trainer=trainers.BpeTrainer(vocab_size=8192, min_frequency=2, special_tokens=["<|endoftext|>"]))
tok.save(tok_path)
eos_id = tok.token_to_id("<|endoftext|>")
del tok_docs

def stream_write(target_tokens, bin_path, label):
    total = 0
    with open(bin_path, "wb") as f:
        while total < target_tokens:
            encs = tok.encode_batch([next(it)["text"] for _ in range(2000)])
            arr = []
            for e in encs:
                arr.extend(e.ids)
                arr.append(eos_id)
            np.asarray(arr, dtype=np.uint16).tofile(f)
            total += len(arr)
            if total % 20_000_000 < len(arr) or total >= target_tokens:
                print(f"   [{label}] {total/1e6:6.2f}M / {target_tokens/1e6:.0f}M tokens")
    return total

stream_write(4_000_000, val_bin, "VAL")
stream_write(260_000_000, train_bin, "TRAIN")
with open(meta_path, "wb") as f:
    pickle.dump({"vocab_size": tok.get_vocab_size(), "tokenizer_type": "hf_bpe"}, f)
print(f"Done in {time.time()-t0:.1f}s")
