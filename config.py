import os
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Base directories ────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
MRI_SLICES_DIR = os.path.join(DATA_DIR, "mri_slices")
SPLITS_DIR     = os.path.join(DATA_DIR, "splits")
SRC_DIR        = os.path.join(BASE_DIR, "src")
CHECKPOINTS_DIR= os.path.join(BASE_DIR, "checkpoints")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")

# ── CSV files ───────────────────────────────────────────────────────────────
LABELS_CSV     = os.path.join(DATA_DIR, "labels.csv")
SLICES_CSV     = os.path.join(DATA_DIR, "slices.csv")
TRAIN_CSV      = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV        = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV        = os.path.join(SPLITS_DIR, "test.csv")

# ── Class mapping ───────────────────────────────────────────────────────────
CLASS_MAP = {
    "Non Demented":       0,
    "Very mild Dementia": 1,
    "Mild Dementia":      1,
    "Moderate Dementia":  1
}
CLASS_NAMES    = ["Non Demented", "Demented"]
NUM_CLASSES    = 2

# ── Image settings ──────────────────────────────────────────────────────────
IMAGE_SIZE     = 224        # EfficientNetB2 default input size
IMAGE_CHANNELS = 3          # MRI slices are grayscale but models expect 3ch

# ── MIL / slice settings ────────────────────────────────────────────────────
MAX_SLICES_PER_PATIENT = 10  # cap slices per patient for memory
MIN_SLICES_PER_PATIENT = 3   # drop patients with too few slices
USE_SSIM_SELECTION     = False # select informative slices via SSIM

# ── Dataset split ratios ────────────────────────────────────────────────────
TRAIN_SPLIT    = 0.70
VAL_SPLIT      = 0.15
TEST_SPLIT     = 0.15
RANDOM_SEED    = 42

# ── Training hyperparameters ────────────────────────────────────────────────
BATCH_SIZE     = 2          # MIL bags are large — keep batch small
NUM_EPOCHS     = 50
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
LR_STEP_SIZE   = 15          # reduce LR every N epochs
LR_GAMMA       = 0.5         # multiply LR by this on each step
EARLY_STOPPING_PATIENCE = 10 # stop if val loss doesn't improve

# ── Model settings ──────────────────────────────────────────────────────────
BACKBONE       = "efficientnet_b2"   # timm model name
PRETRAINED     = True                # use ImageNet pretrained weights
DROPOUT_RATE   = 0.3
HIDDEN_DIM     = 256         # GRU hidden dimension
GRU_LAYERS     = 1           # number of GRU layers
ATTENTION_DIM  = 128         # MIL attention layer dimension

# ── Loss / class imbalance ──────────────────────────────────────────────────
USE_WEIGHTED_LOSS = True     # weighted BCE to handle imbalance
LABEL_SMOOTHING   = 0.1      # prevents overconfidence

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Augmentation settings ───────────────────────────────────────────────────
AUGMENT_TRAIN  = True
ROTATION_LIMIT = 15          # degrees
FLIP_PROB      = 0.5
ZOOM_LIMIT     = 0.1

# ── Logging / checkpointing ─────────────────────────────────────────────────
USE_WANDB      = False        # set True if you have a wandb account
WANDB_PROJECT  = "alzheimers-detection"
SAVE_BEST_ONLY = True         # only save checkpoint when val loss improves
LOG_INTERVAL   = 5            # print training stats every N epochs


# ── Auto-create directories if they don't exist ─────────────────────────────
for _dir in [DATA_DIR, MRI_SLICES_DIR, SPLITS_DIR,
             CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
    

CLASS_WEIGHTS = [0.6505, 2.1607]