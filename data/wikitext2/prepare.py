import os
import pickle
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

out_dir = os.path.dirname(os.path.abspath(__file__))
VOCAB_SIZE = 4096

print("1. 正在拉取 WikiText-2 (raw)...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

def clean_lines(split):
    # 过滤纯空行，保留维基词条标题 (= Title =) 与段落换行结构
    return [t.strip() + "\n" for t in ds[split]["text"] if t.strip()]

train_lines = clean_lines("train")
val_lines = clean_lines("validation")

print(f"2. 正在训练 {VOCAB_SIZE} 词表 Byte-Level BPE 分词器（约 1~2 秒）...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<eos>"],
)
tokenizer.train_from_iterator(train_lines, trainer=trainer)

tok_path = os.path.join(out_dir, "tokenizer.json")
tokenizer.save(tok_path)

def encode_lines(lines):
    ids = []
    for enc in tokenizer.encode_batch(lines):
        ids.extend(enc.ids)
    return np.array(ids, dtype=np.uint16)

print("3. 正在编码并导出 train.bin / val.bin / meta.pkl...")
train_ids = encode_lines(train_lines)
val_ids = encode_lines(val_lines)

train_ids.tofile(os.path.join(out_dir, "train.bin"))
val_ids.tofile(os.path.join(out_dir, "val.bin"))

meta = {
    "vocab_size": tokenizer.get_vocab_size(),
    "tokenizer_type": "hf_bpe",
}
with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"完成！vocab_size = {meta['vocab_size']}")
print(f"train tokens: {len(train_ids):,} | val tokens: {len(val_ids):,}")
print("分词抽检:", tokenizer.encode("Biology studies the living cell and DNA.").tokens)