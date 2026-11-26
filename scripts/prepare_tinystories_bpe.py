"""
Build TinyStories 110M-token dataset with 4,096-vocab ByteLevel BPE.
Run from repo root: python3 scripts/prepare_tinystories_bpe.py
"""
import os, pickle, time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

out_dir = "data/tinystories_bpe"
os.makedirs(out_dir, exist_ok=True)
train_bin, val_bin = os.path.join(out_dir, "train.bin"), os.path.join(out_dir, "val.bin")
tok_path, meta_path = os.path.join(out_dir, "tokenizer.json"), os.path.join(out_dir, "meta.pkl")

if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(tok_path):
    print("TinyStories BPE dataset already prepared at", out_dir)
    raise SystemExit(0)

t0 = time.time()
ds_train = load_dataset("roneneldan/TinyStories", split="train")
ds_val = load_dataset("roneneldan/TinyStories", split="validation")
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator((ds_train[i]["text"] for i in range(50_000)), trainer=trainers.BpeTrainer(vocab_size=4096, min_frequency=2, special_tokens=["<|endoftext|>"]))
tok.save(tok_path)
eos_id = tok.token_to_id("<|endoftext|>")

def encode_write(ds, limit, path):
    with open(path, "wb") as f:
        for s in range(0, limit, 20_000):
            encs = tok.encode_batch(ds[s:min(s+20_000, limit)]["text"])
            arr = []
            for e in encs:
                arr.extend(e.ids)
                arr.append(eos_id)
            np.asarray(arr, dtype=np.uint16).tofile(f)

encode_write(ds_val, len(ds_val), val_bin)
encode_write(ds_train, min(500_000, len(ds_train)), train_bin)
with open(meta_path, "wb") as f:
    pickle.dump({"vocab_size": tok.get_vocab_size(), "tokenizer_type": "hf_bpe"}, f)
print(f"Done in {time.time()-t0:.1f}s")
