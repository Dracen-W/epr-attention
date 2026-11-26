# 6.8M Extreme Overtraining / Asymptotic Convergence on FineWeb-Edu (245.8M unique tokens)
dataset = "fineweb_edu_8k"
batch_size = 32
max_seq_len = 512
gradient_accumulation_steps = 1

dim = 256
n_layers = 6
n_heads = 4        # head_dim = 64 -> 48 fiber + 14 base + 2 = 64
n_kv_heads = 4
multiple_of = 32
dropout = 0.0

learning_rate = 1e-3
min_lr = 1e-5
warmup_iters = 500
max_iters = 15000
lr_decay_iters = 15000
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
