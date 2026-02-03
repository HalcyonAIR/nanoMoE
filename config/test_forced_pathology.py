# Forced pathology test for ChronoMoE Phase 2 controller
# Induces expert collapse via router logit bias, tests controller recovery

out_dir = 'out-pathology-test'
eval_interval = 50  # More frequent evals to see recovery
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
chrono_lens_rank = 4
# More sensitive thresholds for testing - will trigger on milder collapse
chrono_neff_threshold_ratio = 0.85  # Neff < 0.85*4 = 3.4 triggers debt
chrono_top2_warning = 0.60          # Top2 > 0.60 triggers debt

# Forced pathology: bias expert 0 heavily from step 100-250
# This should cause Top2 to spike and Neff to drop
# Controller should respond with pressure increase and lens intervention
collapse_bias_expert_id = 0
collapse_bias_strength = 15.0  # Very strong bias to overwhelm aux_loss
collapse_bias_start_step = 100
collapse_bias_end_step = 250

# Training - run longer to see recovery after bias ends
learning_rate = 1e-3
max_iters = 400
lr_decay_iters = 400
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 50

# For Mac
device = 'mps'
compile = False
