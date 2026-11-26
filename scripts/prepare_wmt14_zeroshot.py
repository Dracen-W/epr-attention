"""
Prepare Trilingual Pivot WMT14 dataset (150K En<->De + 150K En<->Fr = 600K bidirectional examples,
STRICTLY ZERO De<->Fr pairs) with a 12,288-vocab shared ByteLevel BPE tokenizer, and extract exact
3-way parallel (En, De, Fr) test pairs from newstest2013 for Zero-Shot De<->Fr evaluation.

Run from repository root:
  python3 scripts/prepare_wmt14_zeroshot.py
"""
import json, os, pickle, time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

out_dir = "data/wmt14_trilingual_12k"
os.makedirs(out_dir, exist_ok=True)
train_bin = os.path.join(out_dir, "train.bin")
val_bin   = os.path.join(out_dir, "val.bin")
tok_path  = os.path.join(out_dir, "tokenizer.json")
meta_path = os.path.join(out_dir, "meta.pkl")
test_json = os.path.join(out_dir, "eval_trilingual.json")

if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(test_json):
    print("Trilingual WMT14 dataset already prepared at", out_dir)
    raise SystemExit(0)

t0 = time.time()
PAIRS_PER_LANG = 150_000

print("1. Streaming WMT14 de-en & fr-en from HuggingFace...")
ds_de_tr = iter(load_dataset("wmt/wmt14", "de-en", split="train", streaming=True))
ds_fr_tr = iter(load_dataset("wmt/wmt14", "fr-en", split="train", streaming=True))

def collect_bidir(it, lang_code, limit):
    tag = f"<|{lang_code}|>"
    texts, cnt = [], 0
    while cnt < limit:
        ex = next(it)["translation"]
        en, tgt = ex["en"].strip(), ex[lang_code].strip()
        if 10 <= len(en) <= 260 and 10 <= len(tgt) <= 260:
            texts.append(f"<|en|> {en} {tag} {tgt} <|endoftext|>")
            texts.append(f"{tag} {tgt} <|en|> {en} <|endoftext|>")
            cnt += 1
    return texts

print(f"2. Collecting {PAIRS_PER_LANG:,} En<->De and {PAIRS_PER_LANG:,} En<->Fr bidirectional pairs...")
train_texts = collect_bidir(ds_de_tr, "de", PAIRS_PER_LANG) + collect_bidir(ds_fr_tr, "fr", PAIRS_PER_LANG)
np.random.RandomState(42).shuffle(train_texts)

print("3. Training 12,288-vocab shared En/De/Fr ByteLevel BPE tokenizer...")
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=12288,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|en|>", "<|de|>", "<|fr|>"],
)
tok.train_from_iterator(train_texts[:60_000], trainer=trainer)
tok.save(tok_path)

def encode_write(texts, bin_path, label):
    total = 0
    with open(bin_path, "wb") as f:
        for i in range(0, len(texts), 20_000):
            encs = tok.encode_batch(texts[i : i + 20_000])
            arr = []
            for e in encs:
                arr.extend(e.ids)
            np.asarray(arr, dtype=np.uint16).tofile(f)
            total += len(arr)
    print(f"   [{label}] {len(texts):,} examples -> {total/1e6:.2f}M tokens")

ds_de_val = load_dataset("wmt/wmt14", "de-en", split="validation")
ds_fr_val = load_dataset("wmt/wmt14", "fr-en", split="validation")
en2de = {x["translation"]["en"].strip(): x["translation"]["de"].strip() for x in ds_de_val}

tri_pairs, val_texts = [], []
for x in ds_fr_val:
    en, fr = x["translation"]["en"].strip(), x["translation"]["fr"].strip()
    if en in en2de:
        de = en2de[en]
        tri_pairs.append({"en": en, "de": de, "fr": fr})
        val_texts.append(f"<|en|> {en} <|de|> {de} <|endoftext|>")
        val_texts.append(f"<|en|> {en} <|fr|> {fr} <|endoftext|>")

print(f"4. Extracted {len(tri_pairs):,} exact 3-way parallel (En, De, Fr) sentences from newstest2013!")
encode_write(val_texts, val_bin, "VAL")
encode_write(train_texts, train_bin, "TRAIN")

with open(test_json, "w", encoding="utf-8") as f:
    json.dump(tri_pairs[:200], f, ensure_ascii=False, indent=2)
with open(meta_path, "wb") as f:
    pickle.dump({"vocab_size": tok.get_vocab_size(), "tokenizer_type": "hf_bpe"}, f)

print(f"\nDone in {time.time()-t0:.1f}s!")
