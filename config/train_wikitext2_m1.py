# Modern Llama 2 (RMSNorm + RoPE + SwiGLU + FlashAttention)
# WikiText-2 (4K BPE) 3000-step convergence benchmark preset for Apple Silicon (M1 MPS)

dataset = "wikitext2"
batch_size = 32
max_seq_len = 128
gradient_accumulation_steps = 1

# Model (~1.33M params; exact decayed parameter alignment between Pure Llama 2 and Dynamic Field)
dim = 128
n_layers = 4
n_heads = 4
n_kv_heads = 4
multiple_of = 32
dropout = 0.1
num_fields = 4
align_params = True

# Optimization (3000 steps ~= 12.3M tokens ~= 4.3 epochs on WikiText-2 to reach convergence)
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 150
max_iters = 3000
lr_decay_iters = 3000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Eval & logging
eval_interval = 250
eval_iters = 20
log_interval = 50
always_save_checkpoint = False
export_bin = False

# Hardware
device = "mps"
dtype = "float32"
compile = False
