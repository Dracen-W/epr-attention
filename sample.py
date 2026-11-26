"""
Text Sampling Script with Damped Pauli Repulsion Decoding.

Usage:
  python3 sample.py --out_dir=out-6m-v6-wormhole \
      --start="In 1945 , the" --temperature=0.7 --top_k=40 \
      --pauli_alpha=1.8 --pauli_halflife=16.0
"""

import os
import pickle
from contextlib import nullcontext

import torch

from model import ModelArgs, Transformer

# -----------------------------------------------------------------------------
out_dir = ""
checkpoint = "out/ckpt.pt"
start = "Biology is the study of"
num_samples = 2
max_new_tokens = 160
temperature = 0.7
top_k = 40
pauli_alpha = 1.8      # Damped Pauli Repulsion strength (0.0 disables repulsion)
pauli_halflife = 16.0  # Repulsion decay half-life in tokens (short-range anti-loop)
tokenizer = ""
seed = 42
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = "float32"
compile = False
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

if out_dir:
    checkpoint = os.path.join(out_dir, "ckpt.pt")

torch.manual_seed(seed)
if "cuda" in device:
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
# Filter legacy keys cleanly if loading older checkpoints
valid_fields = set(ModelArgs.__dataclass_fields__.keys())
clean_args = {k: v for k, v in ckpt["model_args"].items() if k in valid_fields}
gptconf = ModelArgs(**clean_args)
model = Transformer(gptconf)
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(sd, strict=True)
model.eval().to(device)
if compile:
    model = torch.compile(model)

total_params = sum(p.numel() for p in model.parameters())
print(
    f"Loaded {checkpoint} | params: {total_params/1e6:.2f}M | "
    f"gravity={gptconf.use_unified_gravity} | wormhole={gptconf.use_wormhole} | "
    f"pauli_alpha={pauli_alpha} (half-life={pauli_halflife}t)"
)

# Resolve tokenizer (HF BPE / char / Llama 2 SentencePiece)
cfg = ckpt.get("config", {})
dataset = cfg.get("dataset", "wikitext2")
meta_path = os.path.join("data", dataset, "meta.pkl")

if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if meta.get("tokenizer_type") == "hf_bpe":
        from tokenizers import Tokenizer as HFTokenizer
        tok_path = tokenizer if tokenizer else os.path.join(os.path.dirname(meta_path), "tokenizer.json")
        tok = HFTokenizer.from_file(tok_path)
        encode = lambda s: tok.encode(s).ids
        decode = lambda l: tok.decode(l)
    else:
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
else:
    from tokenizer import Tokenizer as SPTokenizer
    from tinystories import get_tokenizer_model_path
    vocab_source = cfg.get("vocab_source", "llama2")
    tok_model = tokenizer if tokenizer else get_tokenizer_model_path(vocab_size=0 if vocab_source == "llama2" else gptconf.vocab_size)
    enc = SPTokenizer(tokenizer_model=tok_model)
    encode = lambda s: enc.encode(s, bos=True, eos=False)
    decode = lambda l: enc.decode(l)

if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()

start_ids = encode(start) or [0]
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad(), ctx:
    for _ in range(num_samples):
        y = model.generate(
            x, max_new_tokens,
            temperature=temperature, top_k=top_k,
            pauli_alpha=pauli_alpha, pauli_halflife=pauli_halflife,
        )
        print(decode(y[0].tolist()))
        print("-" * 60)
