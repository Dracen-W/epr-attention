# Google Colab CUDA (L4 / A100 / G4 / T4) — Exact 128-token parity with M1 runs
# 3000 steps finishes in ~20-40 seconds on L4/A100/G4!

dataset = "wikitext2"
batch_size = 32
max_seq_len = 128
gradient_accumulation_steps = 1

# Exact same 1.33M architecture as M1 benchmark (directly comparable to Pure Llama 2 4.3126 / v4 noRBF 4.2585)
dim = 128
n_layers = 4
n_heads = 4
n_kv_heads = 4
multiple_of = 32
dropout = 0.1

learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 150
max_iters = 3000
lr_decay_iters = 3000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 250
eval_iters = 20
log_interval = 100
always_save_checkpoint = False
export_bin = False

device = "cuda"
dtype = "bfloat16"   # Auto-falls back to float16 + GradScaler on T4
compile = False      # Set --compile=True for extra throughput on longer runs
