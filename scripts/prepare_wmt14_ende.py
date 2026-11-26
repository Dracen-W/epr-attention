"""
Prepare 300K WMT14 En-De sentence pairs with an 8,192-vocab shared ByteLevel BPE tokenizer
plus 200 deterministic test pairs from official newstest2014 for sacreBLEU evaluation.

Run from repository root:
  python3 scripts/prepare_wmt14_ende.py
"""
import json, os, pickle, time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

out_dir = "data/wmt14_ende_8k"
os.makedirs(out_dir, exist_ok=True)
train_bin = os.path.join(out_dir, "train.bin")
val_bin = os.path.join(out_dir, "val.bin")
tok_path = os.path.join(out_dir, "tokenizer.json")
meta_path = os.path.join(out_dir, "meta.pkl")
test_json = os.path.join(out_dir, "test_pairs.json")

if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(test_json):
    print("WMT14 En-De 8K dataset already prepared at", out_dir)
    raise SystemExit(0)

t0 = time.time()
NUM_TRAIN_PAIRS = 300_000
print("1. Streaming wmt/wmt14 (de-en) from HuggingFace...")
ds_train = load_dataset("wmt/wmt14", "de-en", split="train", streaming=True)
ds_val   = load_dataset("wmt/wmt14", "de-en", split="validation")  # newstest2013
ds_test  = load_dataset("wmt/wmt14", "de-en", split="test")        # newstest2014

def fmt_pair(en, de):
    return f"<|en|> {en.strip()} <|de|> {de.strip()} <|endoftext|>"

it_tr = iter(ds_train)
train_texts = []
print(f"2. Collecting {NUM_TRAIN_PAIRS:,} clean En-De pairs...")
while len(train_texts) < NUM_TRAIN_PAIRS:
    ex = next(it_tr)["translation"]
    en, de = ex["en"].strip(), ex["de"].strip()
    if 10 <= len(en) <= 300 and 10 <= len(de) <= 300:
        train_texts.append(fmt_pair(en, de))
        if len(train_texts) % 100_000 == 0:
            print(f"   collected {len(train_texts):,} pairs ({time.time()-t0:.1f}s)")

print("3. Training 8,192-vocab shared En-De ByteLevel BPE tokenizer...")
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=8192,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|en|>", "<|de|>"],
)
tok.train_from_iterator(train_texts[:50_000], trainer=trainer)
tok.save(tok_path)

def encode_list(texts, bin_path, label):
    total = 0
    with open(bin_path, "wb") as f:
        for i in range(0, len(texts), 20_000):
            encs = tok.encode_batch(texts[i : i + 20_000])
            arr = []
            for e in encs:
                arr.extend(e.ids)
            np.asarray(arr, dtype=np.uint16).tofile(f)
            total += len(arr)
    print(f"   [{label}] {len(texts):,} pairs -> {total/1e6:.2f}M tokens")

val_texts = [fmt_pair(x["translation"]["en"], x["translation"]["de"]) for x in ds_val]
encode_list(val_texts, val_bin, "VAL (newstest2013)")
encode_list(train_texts, train_bin, "TRAIN")

test_pairs = [{"en": ds_test[i]["translation"]["en"].strip(), "de": ds_test[i]["translation"]["de"].strip()} for i in range(200)]
with open(test_json, "w", encoding="utf-8") as f:
    json.dump(test_pairs, f, ensure_ascii=False, indent=2)

with open(meta_path, "wb") as f:
    pickle.dump({"vocab_size": tok.get_vocab_size(), "tokenizer_type": "hf_bpe"}, f)

print(f"\nDone! Saved WMT14 En-De dataset & newstest2014 eval set in {time.time()-t0:.1f}s")
