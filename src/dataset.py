import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    IMAGE_SIZE, IMAGE_CHANNELS,
    MAX_SLICES_PER_PATIENT, MIN_SLICES_PER_PATIENT,
    USE_SSIM_SELECTION, AUGMENT_TRAIN,
    ROTATION_LIMIT, FLIP_PROB, ZOOM_LIMIT,
    BATCH_SIZE, RANDOM_SEED
)


# ── Augmentation pipelines ───────────────────────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Rotate(limit=ROTATION_LIMIT, p=0.5),
        A.HorizontalFlip(p=FLIP_PROB),
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(1 - ZOOM_LIMIT, 1.0),
            p=0.3
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats — model is pretrained on these
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms():
    """No augmentation for val/test — only resize and normalize."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# ── SSIM-based slice selector ────────────────────────────────────────────────

def select_informative_slices(image_paths, max_slices):
    """
    Rank slices by edge density (high edge = more brain structure info).
    Falls back to uniform sampling if not enough slices.
    Returns list of selected paths.
    """
    if len(image_paths) <= max_slices:
        return image_paths

    scores = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            scores.append(0.0)
            continue
        # Sobel edge density as informativeness proxy
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = np.sqrt(sobelx**2 + sobely**2).mean()
        scores.append(edge_density)

    # pick top-k slices by edge density
    ranked = sorted(zip(scores, image_paths), key=lambda x: x[0], reverse=True)
    selected = [path for _, path in ranked[:max_slices]]
    return selected


# ── Core Dataset class ───────────────────────────────────────────────────────

class PatientDataset(Dataset):
    """
    MIL (Multiple Instance Learning) Dataset.

    Each item returned is a BAG — all MRI slices for one patient.
    Shape returned: (num_slices, C, H, W)

    Args:
        csv_path   : path to train.csv / val.csv / test.csv
        transform  : albumentations transform pipeline
        is_train   : if True, applies SSIM selection + augmentation
    """

    def __init__(self, csv_path, transform=None, is_train=False):

        self.transform = transform
        self.is_train  = is_train

        df = pd.read_csv(csv_path)

        # group slices by patient
        self.patients = []   # list of dicts: {patient_id, label, slice_paths}
        skipped = 0

        for patient_id, group in df.groupby("patient_id"):
            label       = int(group["label"].iloc[0])
            slice_paths = group["filepath"].tolist()

            # drop patients with too few slices
            if len(slice_paths) < MIN_SLICES_PER_PATIENT:
                skipped += 1
                continue

            # select informative slices
            if USE_SSIM_SELECTION and is_train:
                slice_paths = select_informative_slices(
                    slice_paths, MAX_SLICES_PER_PATIENT)
            else:
                # for val/test just take evenly spaced slices
                slice_paths = self._uniform_sample(
                    slice_paths, MAX_SLICES_PER_PATIENT)

            self.patients.append({
                "patient_id":  patient_id,
                "label":       label,
                "slice_paths": slice_paths
            })

        print(f"Loaded {len(self.patients)} patients "
              f"({skipped} skipped — fewer than {MIN_SLICES_PER_PATIENT} slices)")
        self._print_distribution()

    def _uniform_sample(self, paths, max_slices):
        """Evenly space slice selection across the volume."""
        if len(paths) <= max_slices:
            return paths
        indices = np.linspace(0, len(paths) - 1, max_slices, dtype=int)
        return [paths[i] for i in indices]

    def _print_distribution(self):
        labels = [p["label"] for p in self.patients]
        n_nd = labels.count(0)
        n_d  = labels.count(1)
        print(f"  Non-Demented : {n_nd} patients")
        print(f"  Demented     : {n_d}  patients")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient    = self.patients[idx]
        label      = patient["label"]
        slice_paths= patient["slice_paths"]

        images = []
        for path in slice_paths:
            img = cv2.imread(path)

            if img is None:
                # corrupt/missing file — replace with blank slice
                img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(image=img)["image"]
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            images.append(img)

        # stack into (num_slices, C, H, W)
        bag    = torch.stack(images, dim=0)
        label  = torch.tensor(label, dtype=torch.float32)

        return bag, label, patient["patient_id"]


# ── Custom collate for variable-length bags ──────────────────────────────────

def mil_collate_fn(batch):
    """
    DataLoader collate for MIL bags.
    Each patient has a different number of slices, so we
    pad bags to the same length within a batch.

    Returns:
        bags   : (B, max_slices, C, H, W)
        labels : (B,)
        masks  : (B, max_slices)  — True = real slice, False = padding
        ids    : list of patient_id strings
    """
    bags, labels, ids = zip(*batch)

    max_slices = max(b.shape[0] for b in bags)
    C, H, W    = bags[0].shape[1], bags[0].shape[2], bags[0].shape[3]

    padded_bags = torch.zeros(len(bags), max_slices, C, H, W)
    masks       = torch.zeros(len(bags), max_slices, dtype=torch.bool)

    for i, bag in enumerate(bags):
        n = bag.shape[0]
        padded_bags[i, :n] = bag
        masks[i, :n]       = True

    labels = torch.stack(labels)

    return padded_bags, labels, masks, list(ids)


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders():
    """
    Returns train, val, test DataLoaders ready for training.
    """
    train_dataset = PatientDataset(
        csv_path  = TRAIN_CSV,
        transform = get_train_transforms(),
        is_train  = True
    )
    val_dataset = PatientDataset(
        csv_path  = VAL_CSV,
        transform = get_val_transforms(),
        is_train  = False
    )
    test_dataset = PatientDataset(
        csv_path  = TEST_CSV,
        transform = get_val_transforms(),
        is_train  = False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = mil_collate_fn,
        num_workers = 0,        # set to 2-4 if on Linux; keep 0 on Windows
        pin_memory  = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = mil_collate_fn,
        num_workers = 0,
        pin_memory  = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = mil_collate_fn,
        num_workers = 0,
        pin_memory  = True
    )

    return train_loader, val_loader, test_loader


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":

    print("="*50)
    print("TRAIN")
    print("="*50)
    train_loader, val_loader, test_loader = get_dataloaders()

    print("\n" + "="*50)
    print("Checking one batch from train loader...")
    print("="*50)

    bags, labels, masks, ids = next(iter(train_loader))

    print(f"Bag shape      : {bags.shape}")
    print(f"  B            : {bags.shape[0]}  (patients in batch)")
    print(f"  max_slices   : {bags.shape[1]}  (padded to longest bag)")
    print(f"  C x H x W   : {bags.shape[2]} x {bags.shape[3]} x {bags.shape[4]}")
    print(f"Labels shape   : {labels.shape}  -> {labels.tolist()}")
    print(f"Mask shape     : {masks.shape}")
    print(f"Real slices    : {masks.sum(dim=1).tolist()}  (per patient)")
    print(f"Patient IDs    : {ids}")

    print("\n[OK] Dataset is working correctly")