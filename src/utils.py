import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    f1_score, classification_report,
    roc_curve
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import OUTPUTS_DIR, CHECKPOINTS_DIR, CLASS_NAMES


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_probs, threshold=0.5):
    """
    Args:
        all_labels : list of true binary labels
        all_probs  : list of predicted probabilities (after sigmoid)
        threshold  : cutoff for binary prediction

    Returns dict of all key metrics.
    """
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= threshold).astype(int)

    auc  = roc_auc_score(all_labels, all_probs)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    # sensitivity = recall for demented class
    # specificity = recall for non-demented class
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "auc":         round(auc,         4),
        "f1":          round(f1,          4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "accuracy":    round(accuracy,    4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn)
    }


def print_metrics(metrics, split="Val"):
    print(f"\n{split} Metrics:")
    print(f"  AUC         : {metrics['auc']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}  (recall for Demented)")
    print(f"  Specificity : {metrics['specificity']:.4f}  (recall for Non-Demented)")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  TP:{metrics['tp']}  TN:{metrics['tn']}  FP:{metrics['fp']}  FN:{metrics['fn']}")


# ── Checkpoint saving / loading ───────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, filename="best_model.pt"):
    path = os.path.join(CHECKPOINTS_DIR, filename)
    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":    metrics
    }, path)
    print(f"  [Saved] checkpoint -> {path}  (AUC: {metrics['auc']:.4f})")


def load_checkpoint(model, optimizer=None, filename="best_model.pt"):
    path = os.path.join(CHECKPOINTS_DIR, filename)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(AUC: {checkpoint['metrics']['auc']:.4f})")
    return checkpoint["epoch"], checkpoint["metrics"]


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses,
                         train_aucs,   val_aucs, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # loss curve
    axes[0].plot(train_losses, label="Train Loss", color="#534AB7", linewidth=2)
    axes[0].plot(val_losses,   label="Val Loss",   color="#D85A30", linewidth=2)
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC curve
    axes[1].plot(train_aucs, label="Train AUC", color="#534AB7", linewidth=2)
    axes[1].plot(val_aucs,   label="Val AUC",   color="#D85A30", linewidth=2)
    axes[1].set_title("AUC per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUTS_DIR, "training_curves.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] training curves -> {path}")
    plt.show()


def plot_confusion_matrix(all_labels, all_probs,
                          threshold=0.5, save=True):
    all_preds = (np.array(all_probs) >= threshold).astype(int)
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] confusion matrix -> {path}")
    plt.show()


def plot_roc_curve(all_labels, all_probs, auc_score, save=True):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#534AB7", linewidth=2,
            label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_title("ROC Curve (Test Set)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] ROC curve -> {path}")
    plt.show()


def plot_attention_weights(weights, patient_id, label, save=True):
    """Visualize per-slice attention weights for one patient."""
    w = weights.squeeze().cpu().numpy()
    n = len(w)

    fig, ax = plt.subplots(figsize=(max(10, n // 2), 3))
    bars = ax.bar(range(n), w,
                  color=["#534AB7" if v > w.mean() else "#AFA9EC" for v in w])
    ax.axhline(w.mean(), color="#D85A30", linestyle="--",
               linewidth=1, label="Mean weight")
    ax.set_title(f"Slice Attention Weights — {patient_id} "
                 f"({'Demented' if label == 1 else 'Non-Demented'})")
    ax.set_xlabel("Slice index")
    ax.set_ylabel("Attention weight")
    ax.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, f"attention_{patient_id}.png")
        plt.savefig(path, dpi=150)
        print(f"  [Saved] attention plot -> {path}")
    plt.show()