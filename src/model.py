import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    BACKBONE, PRETRAINED, DROPOUT_RATE,
    HIDDEN_DIM, GRU_LAYERS, ATTENTION_DIM,
    NUM_CLASSES, DEVICE
)


# ── CBAM : Channel + Spatial Attention ──────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C))
        mx  = self.fc(self.max_pool(x).view(B, C))
        scale = self.sigmoid(avg + mx).view(B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Highlights which spatial regions matter most."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ── MIL Attention Pooling ────────────────────────────────────────────────────

class MILAttentionPool(nn.Module):
    """
    Learns a scalar importance weight per slice.
    Weighted sum of slice features → patient-level representation.

    Gated attention (Ilse et al. 2018) — more stable than plain softmax.
    """
    def __init__(self, feature_dim, attention_dim):
        super().__init__()
        self.V = nn.Linear(feature_dim, attention_dim, bias=False)
        self.U = nn.Linear(feature_dim, attention_dim, bias=False)
        self.w = nn.Linear(attention_dim, 1,            bias=False)

    def forward(self, x, mask):
        """
        Args:
            x    : (B, num_slices, feature_dim)
            mask : (B, num_slices)  True = real slice

        Returns:
            pooled  : (B, feature_dim)
            weights : (B, num_slices)  attention weights per slice
        """
        # gated attention scores
        scores = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))
        # (B, num_slices, 1)

        # mask out padding slices before softmax
        scores = scores.squeeze(-1)                        # (B, num_slices)
        scores = scores.masked_fill(~mask, float("-inf"))  # padding → -inf
        weights = F.softmax(scores, dim=1)                 # (B, num_slices)

        # weighted sum
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)   # (B, feature_dim)
        return pooled, weights


# ── Full Model ───────────────────────────────────────────────────────────────

class AlzheimerMILModel(nn.Module):
    """
    Full pipeline per forward pass:

    1. CNN backbone  : EfficientNetB2 extracts features per slice
                       (B*num_slices, C, H, W) → (B*num_slices, feature_dim)
    2. CBAM          : channel + spatial attention inside CNN features
    3. GRU           : models sequential context across slices
                       (B, num_slices, feature_dim) → (B, num_slices, hidden_dim)
    4. MIL attention : learns slice importance weights
                       (B, num_slices, hidden_dim) → (B, hidden_dim)
    5. Classifier    : patient-level binary prediction
                       (B, hidden_dim) → (B, 1)
    """

    def __init__(self):
        super().__init__()

        # ── 1. EfficientNetB2 backbone ───────────────────────────────────
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained   = PRETRAINED,
            num_classes  = 0,       # remove classification head
            global_pool  = ""       # remove global pooling — we apply CBAM first
        )
        feature_dim = self.backbone.num_features  # 1408 for EfficientNetB2

        # ── 2. CBAM attention on CNN feature maps ────────────────────────
        self.cbam = CBAM(in_channels=feature_dim)

        # ── global average pool after CBAM ───────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── projection to lower dim before GRU ───────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE)
        )

        # ── 3. GRU for slice sequence context ────────────────────────────
        self.gru = nn.GRU(
            input_size  = HIDDEN_DIM,
            hidden_size = HIDDEN_DIM,
            num_layers  = GRU_LAYERS,
            batch_first = True,
            bidirectional = True,      # forward + backward slice context
            dropout = DROPOUT_RATE if GRU_LAYERS > 1 else 0.0
        )
        gru_out_dim = HIDDEN_DIM * 2   # bidirectional doubles output size

        # ── 4. MIL attention pooling ─────────────────────────────────────
        self.mil_attention = MILAttentionPool(gru_out_dim, ATTENTION_DIM)

        # ── 5. Classifier head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, HIDDEN_DIM // 2),
            nn.LayerNorm(HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM // 2, 1)   # single logit → BCEWithLogitsLoss
        )

    def forward(self, bags, mask):
        """
        Args:
            bags : (B, num_slices, C, H, W)
            mask : (B, num_slices)

        Returns:
            logits  : (B, 1)
            weights : (B, num_slices)  — slice attention weights for Grad-CAM
        """
        B, N, C, H, W = bags.shape

        # flatten patients and slices for CNN
        x = bags.view(B * N, C, H, W)              # (B*N, C, H, W)

        # backbone features
        x = self.backbone(x)                        # (B*N, feature_dim, h, w)

        # CBAM
        x = self.cbam(x)                            # (B*N, feature_dim, h, w)

        # global average pool
        x = self.gap(x).squeeze(-1).squeeze(-1)     # (B*N, feature_dim)

        # project to hidden dim
        x = self.proj(x)                            # (B*N, HIDDEN_DIM)

        # reshape back to sequence
        x = x.view(B, N, -1)                        # (B, N, HIDDEN_DIM)

        # GRU over slice sequence
        x, _ = self.gru(x)                          # (B, N, HIDDEN_DIM*2)

        # MIL attention pooling
        x, weights = self.mil_attention(x, mask)    # (B, HIDDEN_DIM*2)

        # classify
        logits = self.classifier(x)                 # (B, 1)

        return logits, weights


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Building model...")
    model = AlzheimerMILModel().to(DEVICE)

    # count parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # dummy forward pass — same shape as real DataLoader output
    B, N, C, H, W = 2, 20, 3, 224, 224
    dummy_bags  = torch.randn(B, N, C, H, W).to(DEVICE)
    dummy_mask  = torch.ones(B, N, dtype=torch.bool).to(DEVICE)

    print(f"\nDummy input  : {dummy_bags.shape}")

    with torch.no_grad():
        logits, weights = model(dummy_bags, dummy_mask)

    print(f"Logits shape : {logits.shape}    -> (B, 1)")
    print(f"Weights shape: {weights.shape}  -> (B, num_slices)")
    print(f"Sample logits: {logits.squeeze().tolist()}")
    print(f"Weights sum  : {weights.sum(dim=1).tolist()}  (should be ~1.0 each)")
    print("\n[OK] Model forward pass successful")