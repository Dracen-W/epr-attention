"""
Training Script for Llama 2 & Unified Fiber-Gravity / ER=EPR Wormhole Attention (v6).

ATTRIBUTION:
  Built upon Andrej Karpathy's `llama2.c` (MIT License, Copyright (c) 2023 Andrej)
  implementing Meta AI's Llama 2 architecture (Touvron et al., 2023).
  See `LICENSE`, `NOTICE`, and `CITATION.bib`.

FEATURES:
  - Seamless support for Google Colab CUDA (L4 / A100 / G4 / T4) and Apple Silicon MPS.
  - Automatic bfloat16 -> float16 (+ GradScaler) fallback on T4 (Turing) GPUs.
  - Exact deterministic validation evaluation (`np.random.RandomState(42)`) across all runs.
  - Supports Pure Llama 2 (`--use_unified_gravity=False`), v6 Wormhole (`--use_unified_gravity=True`),
    and all fine-grained physics ablations via CLI flags.
"""

import math
import os
import pickle
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ModelArgs, Transformer

# -----------------------------------------------------------------------------
# Default Hyperparameters (overridable via config file or --key=val CLI args)
# -----------------------------------------------------------------------------
# I/O & Logging
out_dir = "out"
eval_interval = 250
log_interval = 50
eval_iters = 20
eval_only = False
always_save_checkpoint = False
init_from = "scratch"  # 'scratch' or 'resume'
export_bin = False
wandb_log = False
wandb_project = "epr-attention"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Dataset & Context Window
dataset = "wikitext2"  # 'wikitext2', 'shakespeare_char', or 'tinystories'
batch_size = 32
max_seq_len = 256
vocab_source = "custom"
vocab_size = 4096

# Backbone Architecture (Llama 2 Standard)
dim = 128
n_layers = 4
n_heads = 4
n_kv_heads = 4
multiple_of = 32
dropout = 0.1

# Unified Fiber-Gravity & ER=EPR Wormhole Attention (v6)
use_unified_gravity = False  # False = Pure Llama 2 Baseline; True = Fiber-Gravity / Wormhole
gravity_dim = 0              # 0 = auto power-of-2 packing (d_g=6 for D=32; d_g=14 for D=64)
use_wormhole = True          # ER=EPR zero-decay, NoPE wormhole on head H-1
wormhole_v0_init = 2.5       # High vacuum throat threshold for wormhole silence
use_base_rbf = True          # Low-dim Euclidean RBF capture in base spacetime
use_bilateral_mass = True    # Bilateral mass coupling M_i * M_j
use_time_dilation = True     # Dynamic proper time dtau_{h,t} >= 0
use_nested_time = True       # Solar-system top-down cascade (Earth -> Sat -> Moon)
use_attn_vacuum = True       # Physical vacuum noise floor e^{v0} (Softmax1)
use_medium_field = False     # Optional explicit continuous medium field Phi_{h,t-1}
use_grav_bridge = True       # Spatial bridge dressing (only active if use_medium_field=True)
use_mass_boost = True        # Field-resonance mass feedback (only active if use_medium_field=True)
use_event_horizon = False    # Optional QK-L2Norm bounded resonance cone

# Optimizer & Schedule (AdamW + Cosine Decay)
gradient_accumulation_steps = 1
learning_rate = 1e-3
max_iters = 3000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 150
lr_decay_iters = -1
min_lr = -1.0

# System & Precision
seed = 1337
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = (
    "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else ("float16" if torch.cuda.is_available() else "float32")
)
compile = False
# -----------------------------------------------------------------------------

config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())

if lr_decay_iters <= 0:
    lr_decay_iters = max_iters
if min_lr < 0.0:
    min_lr = learning_rate / 10.0

# Hardware-safe precision fallbacks (T4 lacks native bfloat16; MPS is most stable in float32)
if "cuda" in device and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
    print("NOTICE: GPU lacks native bfloat16 (e.g., T4); switching dtype -> float16 (+ GradScaler).")
    dtype = "float16"
if "mps" in device and dtype != "float32":
    print(f"NOTICE: Overriding dtype={dtype} -> float32 for numerical stability on Apple MPS.")
    dtype = "float32"

config = {k: globals()[k] for k in config_keys}

# Distributed Data Parallel (DDP) setup if launched via torchrun
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    if "cuda" in device and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Hardware: {gpu_name} ({vram_gb:.1f} GB VRAM) | dtype={dtype} | compile={compile}")
    else:
        print(f"Hardware: device={device} | dtype={dtype} | compile={compile}")
    print(f"tokens per iteration: {tokens_per_iter:,} (accum={gradient_accumulation_steps} * B={batch_size} * T={max_seq_len})")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)

device_type = "cuda" if "cuda" in device else "cpu"
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data Loader (TinyStories or Pre-tokenized Binary Memmap)
# -----------------------------------------------------------------------------
if dataset == "tinystories":
    from tinystories import Task
    assert vocab_source in ["llama2", "custom"]
    assert vocab_source == "custom" or vocab_size == 32000, "Meta Llama 2 tokenizer has vocab_size=32000"
    iter_batches = partial(
        Task.iter_batches,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
        device=device,
        num_workers=0,
    )
else:
    data_dir = os.path.join("data", dataset)
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = int(meta["vocab_size"])
        config["vocab_size"] = vocab_size
        if master_process:
            print(f"found vocab_size = {vocab_size} (inside {meta_path})")

    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    assert os.path.exists(train_bin) and os.path.exists(val_bin), (
        f"Missing {train_bin} or {val_bin}. Run data/{dataset}/prepare.py first."
    )

    _train_mmap = np.memmap(train_bin, dtype=np.uint16, mode="r")
    _val_mmap = np.memmap(val_bin, dtype=np.uint16, mode="r")

    def iter_batches(split, eval_rng=None):
        data = _train_mmap if split == "train" else _val_mmap
        max_idx = len(data) - max_seq_len - 1
        while True:
            ix = eval_rng.randint(0, max_idx, size=(batch_size,)) if eval_rng is not None else torch.randint(max_idx, (batch_size,)).numpy()
            x = torch.stack([torch.from_numpy(data[i : i + max_seq_len].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + max_seq_len].astype(np.int64)) for i in ix])
            if device_type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            yield x, y

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    use_unified_gravity=use_unified_gravity,
    gravity_dim=gravity_dim,
    use_wormhole=use_wormhole,
    wormhole_v0_init=wormhole_v0_init,
    use_base_rbf=use_base_rbf,
    use_bilateral_mass=use_bilateral_mass,
    use_time_dilation=use_time_dilation,
    use_nested_time=use_nested_time,
    use_attn_vacuum=use_attn_vacuum,
    use_medium_field=use_medium_field,
    use_grav_bridge=use_grav_bridge,
    use_mass_boost=use_mass_boost,
    use_event_horizon=use_event_horizon,
)

if init_from == "scratch":
    if use_unified_gravity:
        arch_label = (
            f"Llama 2 + UnifiedFiberGravity(wormhole={use_wormhole}[v0={wormhole_v0_init}], "
            f"rbf={use_base_rbf}, bilat={use_bilateral_mass}, dil={use_time_dilation}, "
            f"nest={use_nested_time}, vac={use_attn_vacuum}, med={use_medium_field}, hor={use_event_horizon})"
        )
    else:
        arch_label = "Pure Llama 2 Baseline (Bit-Exact karpathy/llama2.c)"
    print(f"Initializing from scratch: {arch_label}")
    model = Transformer(ModelArgs(**model_args))
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device, weights_only=False)
    for k in list(model_args.keys()):
        if k in checkpoint["model_args"] and k != "dropout":
            model_args[k] = checkpoint["model_args"][k]
    model = Transformer(ModelArgs(**model_args))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

total_params = sum(p.numel() for p in model.parameters())
print(f"number of parameters: {total_params / 1e6:.2f}M ({total_params:,})")
model.to(device)

# GradScaler is needed only for float16 (T4); bfloat16/float32 use a clean no-op wrapper
if device_type == "cuda" and dtype == "float16":
    scaler = torch.amp.GradScaler("cuda", enabled=True)
else:
    class _NoOpScaler:
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
    scaler = _NoOpScaler()

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

if compile:
    print("compiling the model with torch.compile... (takes ~20-50s on step 0)")
    model = torch.compile(model)

if ddp:
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cos", prefix + "freqs_sin"}
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    """Evaluate train and val loss on fixed deterministic windows for fair cross-run comparisons."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        if dataset == "tinystories":
            batch_iter = iter_batches(split=split)
        else:
            eval_rng = np.random.RandomState(42 if split == "val" else 43)
            batch_iter = iter_batches(split=split, eval_rng=eval_rng)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_e, Y_e = next(batch_iter)
            with ctx:
                model(X_e, Y_e)
                losses[k] = raw_model.last_loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it: int) -> float:
    """Linear warmup followed by cosine decay down to min_lr."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)
t_start = time.time()
t0 = t_start
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_ppl = math.exp(min(20.0, float(losses["val"])))
        elapsed = time.time() - t_start
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, val ppl {val_ppl:.2f}, elapsed {elapsed:.1f}s"
        )
        if wandb_log:
            try:
                wandb.log({
                    "iter": iter_num, "tokens": iter_num * tokens_per_iter,
                    "loss/train": losses["train"], "loss/val": losses["val"],
                    "ppl/val": val_ppl, "lr": lr, "mfu": running_mfu * 100,
                }, step=iter_num)
            except Exception as e:
                print(f"logging to wandb failed: {e}")

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                ckpt = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))
                if export_bin and not use_unified_gravity:
                    from export import model_export
                    model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            model(X, Y)
            loss = raw_model.last_loss / gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f} | lr {lr:.2e} | time {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
