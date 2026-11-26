# 28M Real-World Benchmark (FineWeb-Edu 45% + Wikipedia 35% + CNN/DM News 20%)
# dim=512, 6 layers, 8 heads (head_dim=64 -> 48 fiber + 14 base + 2 = 64 exact packing!)
dataset = "realworld_16k"
batch_size = 32
max_seq_len = 512
gradient_accumulation_steps = 1

dim = 512
n_layers = 6
n_heads = 8        # 7 nested damped heads + 1 ER=EPR Wormhole head
n_kv_heads = 8
multiple_of = 64
dropout = 0.0

learning_rate = 6e-4
min_lr = 1e-5
warmup_iters = 400
max_iters = 12000  # 196.6M unique tokens
lr_decay_iters = 12000
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
