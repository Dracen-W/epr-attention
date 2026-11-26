# Google Colab CUDA (L4 / A100 / G4) — Scaled-up 6.0M Llama 2 (head_dim=64, max_seq_len=512)
# Exact 64-dim hardware packing: d_r=48 resonance fiber + d_g=14 base spacetime + 2 = 64!

dataset = "wikitext2"
batch_size = 32
max_seq_len = 512
gradient_accumulation_steps = 1

dim = 256
n_layers = 6
n_heads = 4        # head_dim = 64 -> 4 Solar-System scales (Moon/Sat/Earth/Sun) with rich 48-dim fiber + 14-dim base!
n_kv_heads = 4
multiple_of = 32
dropout = 0.15

learning_rate = 8e-4
min_lr = 8e-5
warmup_iters = 200
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
dtype = "bfloat16"
compile = True
