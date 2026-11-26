# 7.9M Trilingual Pivot & Zero-Shot De<->Fr Translation Benchmark
# dim=256, 6 layers, 4 heads (head_dim=64 -> 48 fiber + 14 base + 2 = 64), vocab=12288
# 8,000 steps * 16,384 tokens/step = 131.1M tokens (600K En<->De & En<->Fr pairs; 0 De<->Fr pairs!)

dataset = "wmt14_trilingual_12k"
batch_size = 32
max_seq_len = 512
gradient_accumulation_steps = 1

dim = 256
n_layers = 6
n_heads = 4
n_kv_heads = 4
multiple_of = 32
dropout = 0.05

learning_rate = 1e-3
min_lr = 2e-5
warmup_iters = 400
max_iters = 8000
lr_decay_iters = 8000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 1000
eval_iters = 30
log_interval = 500
always_save_checkpoint = False
export_bin = False

device = "cuda"
dtype = "bfloat16"
compile = True
