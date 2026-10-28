import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, EARLY_STOPPING_PATIENCE,
    CLASS_WEIGHTS, LABEL_SMOOTHING, LOGS_DIR
)
from src.dataset import get_dataloaders
from src.model   import AlzheimerMILModel
from src.utils   import (
    compute_metrics, print_metrics,
    save_checkpoint, plot_training_curves
)


# ── Loss function ─────────────────────────────────────────────────────────────

def get_loss_fn():
    """
    Weighted BCE with logits loss.
    pos_weight upweights the demented class to handle imbalance.
    CLASS_WEIGHTS = [non_dem_weight, dem_weight]
    pos_weight    = dem_weight / non_dem_weight
    """
    pos_weight = torch.tensor(
        [CLASS_WEIGHTS[1] / CLASS_WEIGHTS[0]]
    ).to(DEVICE)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ── One epoch of training ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, epoch):
    model.train()

    ACCUMULATION_STEPS = 4    # effective batch = 2 × 4 = 8 patients
    total_loss  = 0.0
    all_labels  = []
    all_probs   = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    optimizer.zero_grad()

    for step, (bags, labels, masks, _) in enumerate(pbar):
        bags   = bags.to(DEVICE)
        labels = labels.to(DEVICE)
        masks  = masks.to(DEVICE)

        logits, _ = model(bags, masks)
        logits    = logits.squeeze(1)

        smooth_labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        loss = loss_fn(logits, smooth_labels) / ACCUMULATION_STEPS

        loss.backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}")

    # handle remaining steps not divisible by accumulation
    optimizer.step()
    optimizer.zero_grad()

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ── One epoch of validation ───────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, loss_fn, epoch, split="Val"):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs  = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [{split}]", leave=False)

    for bags, labels, masks, _ in pbar:
        bags   = bags.to(DEVICE)
        labels = labels.to(DEVICE)
        masks  = masks.to(DEVICE)

        logits, _ = model(bags, masks)
        logits    = logits.squeeze(1)

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ── Full training loop ────────────────────────────────────────────────────────

def train():

    print("="*55)
    print(f"Device : {DEVICE}")
    print(f"Epochs : {NUM_EPOCHS}")
    print(f"LR     : {LEARNING_RATE}")
    print(f"Weights: {CLASS_WEIGHTS}")
    print("="*55)

    # data
    train_loader, val_loader, _ = get_dataloaders()

    # model
    model   = AlzheimerMILModel().to(DEVICE)
    loss_fn = get_loss_fn()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = LR_STEP_SIZE,
        gamma     = LR_GAMMA
    )

    # tracking
    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    best_val_auc   = 0.0
    patience_count = 0

    # log file
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "training_log.txt")
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | "
        f"{'Train AUC':>9} | {'Val AUC':>7} | {'Val F1':>6} | "
        f"{'Sens':>6} | {'Spec':>6} | {'LR':>8}")
    log("-" * 85)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, epoch)
        val_loss,   val_metrics   = validate(
            model, val_loader,   loss_fn, epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # record
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_metrics["auc"])
        val_aucs.append(val_metrics["auc"])

        # log
        log(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
            f"{train_metrics['auc']:>9.4f} | {val_metrics['auc']:>7.4f} | "
            f"{val_metrics['f1']:>6.4f} | {val_metrics['sensitivity']:>6.4f} | "
            f"{val_metrics['specificity']:>6.4f} | {current_lr:>8.6f}")

        # save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc   = val_metrics["auc"]
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_metrics)
        else:
            patience_count += 1
            if patience_count >= EARLY_STOPPING_PATIENCE:
                log(f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    log_file.close()

    print("\nTraining complete.")
    print(f"Best Val AUC : {best_val_auc:.4f}")
    print(f"Log saved    : {log_path}")

    plot_training_curves(train_losses, val_losses,
                         train_aucs,   val_aucs)


if __name__ == "__main__":
    train()