"""
Evaluate WMT14 En->De checkpoints on official newstest2014 (200 pairs) via sacreBLEU.

Usage:
  python3 eval_bleu_wmt14.py out-ende-6m-base out-ende-6m-v6-wormhole
"""
import json, os, sys, time
import torch
import sacrebleu
from tokenizers import Tokenizer as HFTokenizer
from model import ModelArgs, Transformer

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
ckpt_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["out-ende-6m-base", "out-ende-6m-v6-wormhole"]

tok = HFTokenizer.from_file("data/wmt14_ende_8k/tokenizer.json")
eos_id = tok.token_to_id("<|endoftext|>")
with open("data/wmt14_ende_8k/test_pairs.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)

def load_m(path):
    p = os.path.join(path, "ckpt.pt") if os.path.isdir(path) else path
    ck = torch.load(p, map_location=device, weights_only=False)
    valid = set(ModelArgs.__dataclass_fields__.keys())
    m = Transformer(ModelArgs(**{k: v for k, v in ck["model_args"].items() if k in valid}))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=True)
    return m.to(device).eval()

@torch.inference_mode()
def translate_greedy(model, en_text, max_new=80):
    ids = tok.encode(f"<|en|> {en_text} <|de|>").ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    for _ in range(max_new):
        logits = model(idx[:, -model.params.max_seq_len:])[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        if nxt.item() == eos_id:
            break
        out_ids.append(nxt.item())
        idx = torch.cat([idx, nxt], dim=1)
    return tok.decode(out_ids).strip()

refs = [p["de"] for p in pairs]
results = {}
for path in ckpt_dirs:
    label = os.path.basename(path.rstrip("/"))
    m = load_m(path)
    t0 = time.time()
    hyps = [translate_greedy(m, p["en"]) for p in pairs]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    results[label] = (bleu.score, hyps)
    print(f"[{label:24s}] newstest2014 ({len(pairs)} pairs) | sacreBLEU = {bleu.score:5.2f} | elapsed {time.time()-t0:.1f}s")

print("\n=== Side-by-Side Translation Samples (newstest2014) ===")
for i in range(5):
    print(f"\n--- Sample {i+1} ---")
    print(f"EN  : {pairs[i]['en']}")
    print(f"REF : {pairs[i]['de']}")
    for label in results:
        print(f"{label:24s}: {results[label][1][i]}")
