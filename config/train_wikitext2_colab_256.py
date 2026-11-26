# Google Colab CUDA (L4 / A100 / G4) — 256-token Solar-System Long-Context Benchmark
# Enough context length for Sun (~150t) -> Earth (~35t) -> Satellite (~8t) -> Moon (~2t) nesting to shine!

dataset = "wikitext2"
batch_size = 32
max_seq_len = 256
gradient_accumulation_steps = 1

# Clean 1.33M model with 256 context window (~8,192 tokens/iter * 3000 iters ~= 24.6M tokens)
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
dtype = "bfloat16"
compile = True
