# Quick smoke test for ChronoMoE Phase 2 controller
# Small model, fast iteration, MoE enabled

out_dir = 'out-chrono-test'
eval_interval = 100
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

# Small model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# MoE settings
n_exp = 4
top_k = 2
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = 1.25
eval_capacity = 2.0
stride = 2  # MoE every 2 layers
use_switch_tfm_init = True
router_use_full_prec = True

# ChronoMoE Phase 2
use_chrono_controller = True
chrono_lens_rank = 4  # Small rank for quick test

# Training
learning_rate = 1e-3
max_iters = 500
lr_decay_iters = 500
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 50

# For Mac
device = 'mps'  # Use Metal on Mac, change to 'cuda' for GPU
compile = False
