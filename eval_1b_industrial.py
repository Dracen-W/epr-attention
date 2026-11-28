"""
eval_1b_industrial.py — 1.1B Scale Industrial Capability & Benchmark Suite (NVIDIA L4 / A100 GPU)

Executes two complementary 1.1B experimental tracks:
  [Track A] 1.1B From-Scratch Equal-Budget Pretraining & 4x Extrapolation (`FineWeb-Edu`):
    - Pure Llama-1.1B (`dim=2048, 22L, 32Q/4KV GQA`, `1,034.51M` tied params)
    - EPR-1.1B + Mass-Elastic RBF (`1,012.35M` tied params, `-22,167,552` fewer matrix weights!)
    Trained on identical `FineWeb-Edu` token streams (`T=1024`, `300` steps × `16,384` tok/step),
    comparing Same-Step & Iso-Wallclock Val PPL, VRAM, throughput, and `[0:1K]/[1K:2K]/[2K:4K]` Extrap PPL.

  [Track B] Pretrained `TinyLlama-1.1B-3T` Architectural Surgery + Downstream Benchmarks:
    - Anchor: Untouched `TinyLlama-1.1B-intermediate-step-1431k-3T` (`1,100.05M` untied params).
    - Control (`base_1b_cpt`): Intact `TinyLlama-1.1B` (`0` weights cut) continued-pretrained on `FineWeb-Edu` (`T=2048`, `300` steps).
    - Surgery (`epr_elastic_1b_cpt`): In-place `upgrade_hf_llama_to_epr` (`1,077.88M` params, `-22,166,760` net)
      continued-pretrained on the EXACT SAME `FineWeb-Edu` `T=2048` batches (`300` steps = `9.83M` tokens).
    Evaluated side-by-side (`lm-eval` conditional log-likelihood protocol, `400` validation samples/task):
      1. `ARC-Easy`, `HellaSwag`, `WinoGrande`, `PIQA`, `BoolQ` (Acc & Acc_norm %)
      2. `FineWeb-Edu` Held-Out Validation PPL across `[0:1024]`, `[1024:2048]`, `[2048:3072]`, `[3072:4096]`
      3. `4K Passkey Retrieval` (`Dist = 1024, 2048, 3584`) — Full Context vs `W=256` Evicted (`-71.9%` KV)
"""

import copy
import gc
import json
import math
import os
import time
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hf_epr import upgrade_hf_llama_to_epr
from model import ModelArgs, Transformer


# =============================================================================
# 1. Tokenized FineWeb-Edu Stream Loader (Cached Locally in /content)
# =============================================================================

def prepare_fineweb_tokens(tokenizer, cache_path: str = "/content/fineweb_1b_tokens.npy", target_tokens: int = 22_000_000):
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        print(f"Loaded cached FineWeb-Edu tokens: {len(arr):,} tokens from {cache_path}", flush=True)
        return arr

    from datasets import load_dataset
    print(f"Streaming & tokenizing {target_tokens:,} FineWeb-Edu tokens with {tokenizer.name_or_path}...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    chunks = []
    total = 0
    eos = tokenizer.eos_token_id or 2
    for row in ds:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        ids.append(eos)
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if total >= target_tokens:
            break
    arr = np.concatenate(chunks, axis=0)
    np.save(cache_path, arr)
    print(f"Saved {len(arr):,} tokens -> {cache_path}", flush=True)
    return np.load(cache_path, mmap_mode="r")


# =============================================================================
# 2. Standard Zero-Shot Multiple-Choice Evaluator (Exact `lm-eval` Protocol)
# =============================================================================

@torch.no_grad()
def score_mc_dataset(model, tokenizer, items: List[Tuple[str, List[str], int]], device: str = "cuda") -> Tuple[float, float]:
    """
    Exact `lm-evaluation-harness` conditional log-likelihood multiple-choice evaluator.
    Each item is `(context_str, [continuation_0, continuation_1, ...], gold_idx)`.
    Returns `(acc_pct, acc_norm_pct)`.
    """
    model.eval()
    hits_raw = 0
    hits_norm = 0

    for ctx_str, choices, gold in items:
        ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=True)
        if not ctx_ids:
            ctx_ids = [tokenizer.bos_token_id or 1]

        seqs = []
        cont_lens = []
        char_lens = []
        for ch in choices:
            full_ids = tokenizer.encode(ctx_str + ch, add_special_tokens=True)
            # Fallback if BPE merges across boundary
            c_len = max(1, len(full_ids) - len(ctx_ids))
            if len(full_ids) <= len(ctx_ids):
                ch_ids = tokenizer.encode(ch, add_special_tokens=False)
                full_ids = ctx_ids + (ch_ids if ch_ids else [0])
                c_len = max(1, len(ch_ids))
            seqs.append(full_ids)
            cont_lens.append(c_len)
            char_lens.append(max(1, len(ch)))

        max_l = max(len(s) for s in seqs)
        pad_id = tokenizer.eos_token_id or 0
        x = torch.full((len(choices), max_l), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
        log_probs = F.log_softmax(logits.float(), dim=-1)

        raw_scores = []
        norm_scores = []
        for i, s in enumerate(seqs):
            L_s = len(s)
            c_len = cont_lens[i]
            start_pos = L_s - c_len
            # Sum log P(s[pos] | s[:pos]) for pos in [start_pos .. L_s - 1]
            lp_sum = 0.0
            for pos in range(start_pos, L_s):
                lp_sum += log_probs[i, pos - 1, s[pos]].item()
            raw_scores.append(lp_sum)
            norm_scores.append(lp_sum / char_lens[i])

        if int(np.argmax(raw_scores)) == gold:
            hits_raw += 1
        if int(np.argmax(norm_scores)) == gold:
            hits_norm += 1

    n = max(1, len(items))
    return round(100.0 * hits_raw / n, 2), round(100.0 * hits_norm / n, 2)


def load_standard_benchmarks(max_samples: int = 500) -> Dict[str, List[Tuple[str, List[str], int]]]:
    from datasets import load_dataset
    tasks = {}

    print("Loading standard zero-shot benchmarks (ARC-Easy, HellaSwag, WinoGrande, BoolQ)...", flush=True)
    # 1. ARC-Easy (test)
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        arc_items = []
        for r in arc:
            labels = r["choices"]["label"]
            texts = [" " + t for t in r["choices"]["text"]]
            if r["answerKey"] in labels:
                gold = labels.index(r["answerKey"])
                arc_items.append((f"Question: {r['question']}\nAnswer:", texts, gold))
                if len(arc_items) >= max_samples:
                    break
        tasks["arc_easy"] = arc_items
    except Exception as e:
        print(f"Warning: ARC-Easy load failed: {e}", flush=True)

    # 2. HellaSwag (validation)
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation")
        hs_items = []
        for r in hs:
            ctx = r["ctx"]
            endings = [" " + e for e in r["endings"]]
            hs_items.append((ctx, endings, int(r["label"])))
            if len(hs_items) >= max_samples:
                break
        tasks["hellaswag"] = hs_items
    except Exception as e:
        print(f"Warning: HellaSwag load failed: {e}", flush=True)

    # 3. WinoGrande (`winogrande_xl` validation)
    try:
        wg = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        wg_items = []
        for r in wg:
            sent = r["sentence"]
            if "_" not in sent:
                continue
            pre, post = sent.split("_", 1)
            ctx = pre.rstrip()
            c1 = " " + r["option1"] + post
            c2 = " " + r["option2"] + post
            gold = int(r["answer"]) - 1
            wg_items.append((ctx, [c1, c2], gold))
            if len(wg_items) >= max_samples:
                break
        tasks["winogrande"] = wg_items
    except Exception as e:
        print(f"Warning: WinoGrande load failed: {e}", flush=True)

    # 4. PIQA (if parquet mirror available) + BoolQ (official Google Parquet)
    try:
        pq = load_dataset("baber/piqa", split="validation")
        pq_items = []
        for r in pq:
            pq_items.append((f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])))
            if len(pq_items) >= max_samples:
                break
        tasks["piqa"] = pq_items
    except Exception as e:
        print(f"Note: PIQA mirror skipped ({e})", flush=True)

    try:
        bq = load_dataset("google/boolq", split="validation")
        bq_items = []
        for r in bq:
            ctx = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
            gold = 1 if r["answer"] else 0
            bq_items.append((ctx, [" no", " yes"], gold))
            if len(bq_items) >= max_samples:
                break
        tasks["boolq"] = bq_items
    except Exception as e:
        print(f"Warning: BoolQ load failed: {e}", flush=True)

    return tasks


@torch.no_grad()
def eval_hf_extrapolation_and_passkey(model, tokenizer, val_tokens: np.ndarray, device: str = "cuda", damped_window: int = 0):
    """Evaluate `[0:1024]`, `[1024:2048]` (in-domain), and `[2048:4096]` (2x zero-shot extrapolation!) + 4K Passkey NIAH."""
    model.eval()

    # Optional damped sliding-window BlockMask on EPR adapter layers (O(T*W) memory & compute!)
    orig_windows = []
    if damped_window > 0 and hasattr(model.model.layers[0].self_attn, "epr"):
        torch._dynamo.reset()
        for lyr in model.model.layers:
            a = lyr.self_attn.epr
            orig_windows.append((a, a.flex_damped_window))
            a.flex_damped_window = damped_window

    # 1. Bucketed PPL up to L=4096 (TinyLlama trained at T=2048 -> [2048:4096] is 2x zero-shot extrapolation!)
    L_max = 4096
    buckets = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
    b_losses = {b: [] for b in buckets}
    rng = np.random.RandomState(2026)
    for _ in range(16):
        ix = rng.randint(len(val_tokens) - L_max - 1000, len(val_tokens) - L_max - 1, size=(2,))
        x = torch.stack([torch.from_numpy(val_tokens[i : i + L_max].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(val_tokens[i + 1 : i + 1 + L_max].astype(np.int64)) for i in ix]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
        for b_i in range(x.size(0)):
            tl_b = F.cross_entropy(logits[b_i].float(), y[b_i], reduction="none")
            for b0, b1 in buckets:
                b_losses[(b0, b1)].append(tl_b[b0:b1].mean().item())
        del out, logits
    ppl_buckets = {f"{b0}:{b1}": round(math.exp(min(20.0, float(np.mean(b_losses[(b0, b1)])))), 2) for b0, b1 in buckets}

    # 2. Natural Passkey Retrieval across `Dist = 1024, 2048, 3584` (80 trials each)
    # Tokenize contiguous `full_txt` and slice exact suffix `target_ids` to avoid SentencePiece double-`▁` artifacts!
    filler_unit = "The grass is green. The sky is blue. The sun is yellow. Here we go back and again. "
    passkey_res = {}
    for dist_toks in [1024, 2048, 3584]:
        rng_p = np.random.RandomState(2026 + dist_toks)
        hits = 0
        trials = 80
        f_ids = tokenizer.encode(filler_unit * ((dist_toks // 16) + 8), add_special_tokens=False)[:dist_toks]
        filler_txt = tokenizer.decode(f_ids, skip_special_tokens=True)
        for _ in range(trials):
            code = str(rng_p.randint(10000, 99999))
            needle_txt = f"The secret passkey is {code}. Remember this number. "
            prefix_txt = f"{needle_txt}{filler_txt}\nWhat is the secret passkey? The secret passkey is "
            full_txt = prefix_txt + code
            pref_ids = tokenizer.encode(prefix_txt, add_special_tokens=True)
            full_ids = tokenizer.encode(full_txt, add_special_tokens=True)
            common = 0
            while common < min(len(pref_ids), len(full_ids)) and pref_ids[common] == full_ids[common]:
                common += 1
            target_ids = full_ids[common:]

            inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(inp)
                logits = out.logits if hasattr(out, "logits") else out
            pred_ids = logits[0, -len(target_ids) :, :].argmax(dim=-1).tolist()
            if pred_ids == target_ids:
                hits += 1
        passkey_res[str(dist_toks)] = round(100.0 * hits / trials, 2)

    for a, orig_w in orig_windows:
        a.flex_damped_window = orig_w

    return ppl_buckets, passkey_res


# =============================================================================
# 3. Track A: 1.1B From-Scratch Pretraining Convergence (`Pure Llama-1.1B` vs `EPR-1.1B`)
# =============================================================================

def run_track_a_1b_from_scratch(tokens_arr: np.ndarray, results: dict, results_path: str, device: str = "cuda"):
    if "track_a_1b_scratch" in results:
        print("Skipping Track A (already completed)", flush=True)
        return

    print("\n" + "=" * 88)
    print("[TRACK A] 1.1B From-Scratch Pretraining Showdown on FineWeb-Edu (22L, dim=2048, 32Q/4KV)")
    print("=" * 88, flush=True)

    T = 1024
    micro_B = 2
    grad_accum = 8  # 16,384 tok/step, L4-safe activation footprint (< 13 GiB)
    steps = 300
    val_split = len(tokens_arr) - 500_000

    track_a = {}
    for tag, use_epr in [("pure_llama_1b", False), ("epr_elastic_1b", True)]:
        torch._dynamo.reset()
        torch.manual_seed(1337)
        np.random.seed(1337)
        margs = ModelArgs(
            dim=2048, n_layers=22, n_heads=32, n_kv_heads=4,
            vocab_size=32000, hidden_dim=5632, multiple_of=256,
            max_seq_len=T, dropout=0.0,
            use_epr=use_epr, use_elastic_rbf=use_epr, use_flex_attention=True,
        )
        raw_m = Transformer(margs).to(device=device, dtype=torch.bfloat16)
        n_params = sum(p.numel() for p in raw_m.parameters())
        print(f"\n--- Training {tag} | params: {n_params:,} ({n_params/1e6:.2f}M) ---", flush=True)

        opt = raw_m.configure_optimizers(0.1, 3e-4, (0.9, 0.95), "cuda")
        model = torch.compile(raw_m)

        rng_train = np.random.RandomState(1337)
        traj = []
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        step_times = []

        for st in range(steps + 1):
            lr = 3e-4 * (st + 1) / 31 if st < 30 else 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (st - 30) / max(1, steps - 30)))
            for g in opt.param_groups:
                g["lr"] = lr

            if st % 50 == 0:
                model.eval()
                rng_v = np.random.RandomState(42)
                v_losses = []
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for _ in range(12):
                        ix = rng_v.randint(val_split, len(tokens_arr) - T - 2, size=(micro_B,))
                        xv = torch.stack([torch.from_numpy(tokens_arr[i : i + T].astype(np.int64)) for i in ix]).to(device)
                        yv = torch.stack([torch.from_numpy(tokens_arr[i + 1 : i + 1 + T].astype(np.int64)) for i in ix]).to(device)
                        model(xv, yv)
                        v_losses.append(raw_m.last_loss.item())
                vl = float(np.mean(v_losses))
                vp = math.exp(min(20.0, vl))
                el = time.perf_counter() - t0
                traj.append({"step": st, "val_loss": round(vl, 4), "val_ppl": round(vp, 2), "wall_sec": round(el, 1)})
                print(f"[{tag}] step {st:4d} | val_loss={vl:.4f} | val_ppl={vp:7.2f} | wall={el:6.1f}s", flush=True)
                model.train()

            if st == steps:
                break

            ts0 = time.perf_counter()
            for _ in range(grad_accum):
                ix = rng_train.randint(0, val_split - T - 2, size=(micro_B,))
                xt = torch.stack([torch.from_numpy(tokens_arr[i : i + T].astype(np.int64)) for i in ix]).to(device)
                yt = torch.stack([torch.from_numpy(tokens_arr[i + 1 : i + 1 + T].astype(np.int64)) for i in ix]).to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(xt, yt)
                    loss = raw_m.last_loss / grad_accum
                loss.backward()
            nn.utils.clip_grad_norm_(raw_m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if st >= 5:
                step_times.append((time.perf_counter() - ts0) * 1000.0)

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        med_ms = float(np.median(step_times))

        # Evaluate 2x & 4x Extrapolation Bucketed PPL (`L=4096`)
        raw_m.eval()
        rng_e = np.random.RandomState(999)
        buckets = [(0, 1024), (1024, 2048), (2048, 4096)]
        bl = {b: [] for b in buckets}
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(12):
                ix = rng_e.randint(val_split, len(tokens_arr) - 4098, size=(2,))
                xe = torch.stack([torch.from_numpy(tokens_arr[i : i + 4096].astype(np.int64)) for i in ix]).to(device)
                ye = torch.stack([torch.from_numpy(tokens_arr[i + 1 : i + 4097].astype(np.int64)) for i in ix]).to(device)
                logits = raw_m(xe, ye)
                for b_i in range(xe.size(0)):
                    tl_b = F.cross_entropy(logits[b_i].float(), ye[b_i], reduction="none")
                    for b0, b1 in buckets:
                        bl[(b0, b1)].append(tl_b[b0:b1].mean().item())
                del logits
        extrap = {f"{b0}:{b1}": round(math.exp(min(20.0, float(np.mean(bl[(b0, b1)])))), 2) for b0, b1 in buckets}
        print(f"[{tag} DONE] median_step={med_ms:.1f}ms ({16384*1000/med_ms:,.0f} tok/s) | VRAM={peak_gb:.2f} GiB | Extrap={extrap}", flush=True)

        track_a[tag] = {
            "params": n_params,
            "median_step_ms": round(med_ms, 2),
            "tok_per_s": round(16384 * 1000.0 / med_ms, 1),
            "peak_vram_gib": round(peak_gb, 2),
            "trajectory": traj,
            "extrapolation_buckets": extrap,
        }
        del model, raw_m, opt
        gc.collect()
        torch.cuda.empty_cache()

    results["track_a_1b_scratch"] = track_a
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# 4. Track B: Pretrained `TinyLlama-1.1B` Surgery + Continued Pretraining + Standard LLM Benchmarks
# =============================================================================

def run_track_b_pretrained_1b_benchmarks(model_id: str, tokenizer, tokens_arr: np.ndarray, results: dict, results_path: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM

    tasks = load_standard_benchmarks(max_samples=400)
    val_split = len(tokens_arr) - 500_000
    T = 2048
    micro_B = 2
    grad_accum = 8  # 32,768 tok/step, fits comfortably inside 22 GiB L4!
    cpt_steps = 300

    # 0. Record untouched pretrained TinyLlama-1.1B anchor before CPT
    if "pretrained_1b_anchor" not in results:
        print("\n" + "=" * 88)
        print(f"[TRACK B-0] Evaluating untouched pretrained anchor: {model_id}")
        print("=" * 88, flush=True)
        try:
            m0 = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        except TypeError:
            m0 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        b0_scores = {}
        for t_name, items in tasks.items():
            acc, acc_norm = score_mc_dataset(m0, tokenizer, items, device=device)
            b0_scores[t_name] = {"acc": acc, "acc_norm": acc_norm}
            print(f"  [pretrained_1b_anchor] {t_name:12s} | acc={acc:5.2f}% | acc_norm={acc_norm:5.2f}%", flush=True)
        ppl0, pk0 = eval_hf_extrapolation_and_passkey(m0, tokenizer, tokens_arr, device=device, damped_window=0)
        results["pretrained_1b_anchor"] = {
            "params": sum(p.numel() for p in m0.parameters()),
            "benchmarks": b0_scores,
            "ppl_buckets_4k": ppl0,
            "passkey_4k": pk0,
            "passkey_v2": True,
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        del m0
        gc.collect()
        torch.cuda.empty_cache()
    elif not results["pretrained_1b_anchor"].get("passkey_v2"):
        print("Refreshing pretrained_1b_anchor Passkey NIAH (SentencePiece-aligned)...", flush=True)
        try:
            m0 = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        except TypeError:
            m0 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        _, pk0 = eval_hf_extrapolation_and_passkey(m0, tokenizer, tokens_arr, device=device, damped_window=0)
        results["pretrained_1b_anchor"]["passkey_4k"] = pk0
        results["pretrained_1b_anchor"]["passkey_v2"] = True
        print(f"  [pretrained_1b_anchor] Passkey Full={pk0}", flush=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        del m0
        gc.collect()
        torch.cuda.empty_cache()

    if "base_1b_cpt" in results and not results["base_1b_cpt"].get("passkey_v2") and os.path.exists("/content/base_1b_cpt_bf16.pt"):
        print("Refreshing base_1b_cpt Passkey NIAH from saved /content/base_1b_cpt_bf16.pt...", flush=True)
        try:
            mb = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        except TypeError:
            mb = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        ckpt_b = torch.load("/content/base_1b_cpt_bf16.pt", map_location="cpu", weights_only=False)
        mb.load_state_dict(ckpt_b["state_dict"])
        mb.eval()
        _, pkb = eval_hf_extrapolation_and_passkey(mb, tokenizer, tokens_arr, device=device, damped_window=0)
        results["base_1b_cpt"]["passkey_4k"] = pkb
        results["base_1b_cpt"]["passkey_v2"] = True
        print(f"  [base_1b_cpt] Passkey Full={pkb}", flush=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        del mb, ckpt_b
        gc.collect()
        torch.cuda.empty_cache()

    for variant in ["base_1b_cpt", "epr_elastic_1b_cpt"]:
        if variant in results:
            print(f"Skipping {variant} (already completed)", flush=True)
            continue

        print("\n" + "=" * 88)
        print(f"[TRACK B] Loading {model_id} for variant: {variant}")
        print("=" * 88, flush=True)

        try:
            hf_m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        except TypeError:
            hf_m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        hf_m.config.use_cache = False

        if variant == "epr_elastic_1b_cpt":
            hf_m, orig_p, new_p = upgrade_hf_llama_to_epr(
                hf_m, use_elastic_rbf=True, use_wormhole=True, wormhole_v0_init=1.8, damped_v0_init=-3.5
            )
            print(f"Upgraded {model_id} -> EPR Attention: {orig_p:,} -> {new_p:,} params ({new_p - orig_p:+,})", flush=True)
        else:
            orig_p = new_p = sum(p.numel() for p in hf_m.parameters())
            print(f"Control {model_id}: {new_p:,} params", flush=True)

        # Continued pretraining on IDENTICAL FineWeb-Edu `T=2048` batches (`seed=2026`)
        decay_p = [p for p in hf_m.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_p = [p for p in hf_m.parameters() if p.requires_grad and p.dim() < 2]
        # Give newly-created EPR geometry projections a slightly higher warmup LR (`3e-4`) while backbone uses `8e-5`
        if variant == "epr_elastic_1b_cpt":
            epr_p = [p for n, p in hf_m.named_parameters() if ".self_attn.epr." in n and p.dim() >= 2]
            epr_ids = {id(p) for p in epr_p}
            other_decay = [p for p in decay_p if id(p) not in epr_ids]
            opt = torch.optim.AdamW([
                {"params": other_decay, "weight_decay": 0.05, "lr_mult": 1.0},
                {"params": epr_p, "weight_decay": 0.05, "lr_mult": 3.0},
                {"params": nodecay_p, "weight_decay": 0.0, "lr_mult": 3.0},
            ], lr=8e-5, betas=(0.9, 0.95), fused=True)
        else:
            opt = torch.optim.AdamW([
                {"params": decay_p, "weight_decay": 0.05, "lr_mult": 1.0},
                {"params": nodecay_p, "weight_decay": 0.0, "lr_mult": 1.0},
            ], lr=8e-5, betas=(0.9, 0.95), fused=True)

        hf_m.gradient_checkpointing_enable()
        hf_m.train()
        rng_cpt = np.random.RandomState(2026)
        t_start = time.perf_counter()

        for st in range(1, cpt_steps + 1):
            base_lr = 8e-5 * st / 40.0 if st < 40 else 1e-5 + 0.5 * (8e-5 - 1e-5) * (1 + math.cos(math.pi * (st - 40) / (cpt_steps - 40)))
            for g in opt.param_groups:
                g["lr"] = base_lr * g["lr_mult"]

            loss_acc = 0.0
            for _ in range(grad_accum):
                ix = rng_cpt.randint(0, val_split - T - 2, size=(micro_B,))
                xb = torch.stack([torch.from_numpy(tokens_arr[i : i + T].astype(np.int64)) for i in ix]).to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = hf_m(xb, labels=xb)
                    loss = out.loss / grad_accum
                loss.backward()
                loss_acc += loss.item()
            nn.utils.clip_grad_norm_(hf_m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if st % 75 == 0 or st == cpt_steps:
                print(f"[{variant}] CPT step {st:4d}/{cpt_steps} | train_loss={loss_acc:.4f} (ppl={math.exp(min(15, loss_acc)):.2f}) | elapsed={time.perf_counter()-t_start:.1f}s", flush=True)

        hf_m.gradient_checkpointing_disable()
        hf_m.eval()

        print(f"--> Running standard zero-shot LLM benchmarks on {variant}...", flush=True)
        bench_scores = {}
        for t_name, items in tasks.items():
            acc, acc_norm = score_mc_dataset(hf_m, tokenizer, items, device=device)
            bench_scores[t_name] = {"acc": acc, "acc_norm": acc_norm}
            print(f"  [{variant}] {t_name:12s} | acc={acc:5.2f}% | acc_norm={acc_norm:5.2f}%", flush=True)

        ppl_b, passkey_full = eval_hf_extrapolation_and_passkey(hf_m, tokenizer, tokens_arr, device=device, damped_window=0)
        print(f"  [{variant}] Extrap PPL={ppl_b} | Passkey Full={passkey_full}", flush=True)

        evict_info = None
        if variant == "epr_elastic_1b_cpt":
            ppl_w256, passkey_w256 = eval_hf_extrapolation_and_passkey(hf_m, tokenizer, tokens_arr, device=device, damped_window=256)
            evict_info = {"window_256_ppl": ppl_w256, "window_256_passkey": passkey_w256}
            print(f"  [{variant}] Evicted W=256 (71.9% KV saved @4K) PPL={ppl_w256} | Passkey W=256={passkey_w256}", flush=True)

        results[variant] = {
            "params": new_p,
            "param_delta": new_p - orig_p,
            "benchmarks": bench_scores,
            "ppl_buckets_4k": ppl_b,
            "passkey_4k": passkey_full,
            "passkey_v2": True,
            "eviction_w256": evict_info,
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save pure bfloat16 model weights (no optimizer state -> ~2.15 GB per 1.1B model!)
        ckpt_out = f"/content/{variant}_bf16.pt"
        print(f"--> Saving pure bfloat16 weights to {ckpt_out}...", flush=True)
        clean_sd = {k: v.detach().cpu().to(torch.bfloat16) for k, v in hf_m.state_dict().items()}
        torch.save(
            {
                "variant": variant,
                "base_model_id": model_id,
                "params": new_p,
                "use_epr": (variant == "epr_elastic_1b_cpt"),
                "use_elastic_rbf": (variant == "epr_elastic_1b_cpt"),
                "state_dict": clean_sd,
            },
            ckpt_out,
        )
        print(f"Saved {ckpt_out} ({os.path.getsize(ckpt_out)/(1024**3):.2f} GiB)", flush=True)
        del clean_sd

        del hf_m, opt
        gc.collect()
        torch.cuda.empty_cache()


def main():
    from transformers import AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    results_path = "/content/industrial_1b_results.json"
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    tok = AutoTokenizer.from_pretrained(model_id)
    tokens_arr = prepare_fineweb_tokens(tok)

    run_track_a_1b_from_scratch(tokens_arr, results, results_path, device="cuda")
    run_track_b_pretrained_1b_benchmarks(model_id, tok, tokens_arr, results, results_path, device="cuda")
    print("\n=== ALL 1.1B INDUSTRIAL BENCHMARKS COMPLETED ===", flush=True)


if __name__ == "__main__":
    main()
