"""
Evaluate Trilingual Pivot checkpoints on Supervised (En->De, En->Fr) and
ZERO-SHOT (De->Fr, Fr->De) translation via sacreBLEU on newstest2013 (150 3-way pairs).

Usage:
  python3 eval_zeroshot_defr.py out-tri-7m-base out-tri-7m-v6-wormhole
"""
import json, os, sys
import torch
import sacrebleu
from tokenizers import Tokenizer as HFTokenizer
from model import ModelArgs, Transformer

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
ckpt_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["out-tri-7m-base", "out-tri-7m-v6-wormhole"]

tok = HFTokenizer.from_file("data/wmt14_trilingual_12k/tokenizer.json")
eos_id = tok.token_to_id("<|endoftext|>")
with open("data/wmt14_trilingual_12k/eval_trilingual.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)[:150]

def load_m(path):
    p = os.path.join(path, "ckpt.pt") if os.path.isdir(path) else path
    ck = torch.load(p, map_location=device, weights_only=False)
    valid = set(ModelArgs.__dataclass_fields__.keys())
    m = Transformer(ModelArgs(**{k: v for k, v in ck["model_args"].items() if k in valid}))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=True)
    return m.to(device).eval()

@torch.inference_mode()
def translate_greedy(model, src_lang, tgt_lang, text, max_new=75):
    ids = tok.encode(f"<|{src_lang}|> {text} <|{tgt_lang}|>").ids
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

models = [(os.path.basename(p.rstrip("/")), load_m(p)) for p in ckpt_dirs]
directions = [
    ("en", "de", "Supervised En->De"),
    ("en", "fr", "Supervised En->Fr"),
    ("de", "fr", "ZERO-SHOT De->Fr "),
    ("fr", "de", "ZERO-SHOT Fr->De "),
]

sample_outputs = {d[2]: {} for d in directions}
header = f"{'Direction':20s} | " + " | ".join(f"{name:>20s}" for name, _ in models)
if len(models) == 2:
    header += f" | {'Gain':>14s}"
print(header)
print("-" * len(header))

for src, tgt, label in directions:
    refs = [p[tgt] for p in pairs]
    scores = []
    for name, m in models:
        hyps = [translate_greedy(m, src, tgt, p[src]) for p in pairs]
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        scores.append(bleu)
        sample_outputs[label][name] = hyps
    row = f"{label:20s} | " + " | ".join(f"{s:20.2f}" for s in scores)
    if len(scores) == 2:
        diff = scores[1] - scores[0]
        pct = (diff / max(scores[0], 1e-6)) * 100.0
        row += f" | {diff:+5.2f} ({pct:+.1f}%)"
    print(row)

print("\n=== ZERO-SHOT De -> Fr Samples (Never Seen During Training!) ===")
for i in range(4):
    print(f"\n--- Sample {i+1} ---")
    print(f"DE (Input) : {pairs[i]['de']}")
    print(f"FR (Ref)   : {pairs[i]['fr']}")
    for name, _ in models:
        print(f"{name:20s}: {sample_outputs['ZERO-SHOT De->Fr '][name][i]}")
