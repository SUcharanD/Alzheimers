import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (LABELS_CSV, SLICES_CSV, SPLITS_DIR,
                    TRAIN_CSV, VAL_CSV, TEST_CSV,
                    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED)


def split_dataset():

    # ── load patient-level labels ────────────────────────────────────────
    labels_df = pd.read_csv(LABELS_CSV)
    slices_df = pd.read_csv(SLICES_CSV)

    print(f"Total patients : {len(labels_df)}")
    print(f"Total slices   : {len(slices_df)}")
    print(f"Split ratio    : {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    print(f"Random seed    : {RANDOM_SEED}\n")

    # ── separate by class for stratified split ───────────────────────────
    non_demented = labels_df[labels_df["label"] == 0]["patient_id"].tolist()
    demented     = labels_df[labels_df["label"] == 1]["patient_id"].tolist()

    print(f"Non-Demented patients : {len(non_demented)}")
    print(f"Demented patients     : {len(demented)}")

    def stratified_split(patients, train_r, val_r, test_r, seed):
        """Split a list of patient IDs into train/val/test."""
        train, temp = train_test_split(
            patients,
            test_size=(val_r + test_r),
            random_state=seed
        )
        val, test = train_test_split(
            temp,
            test_size=test_r / (val_r + test_r),
            random_state=seed
        )
        return train, val, test

    # split each class separately to preserve ratio in every split
    nd_train, nd_val, nd_test = stratified_split(
        non_demented, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED)
    d_train,  d_val,  d_test  = stratified_split(
        demented,     TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED)

    train_ids = nd_train + d_train
    val_ids   = nd_val   + d_val
    test_ids  = nd_test  + d_test

    # ── verify no patient leakage across splits ──────────────────────────
    assert len(set(train_ids) & set(val_ids))  == 0, "LEAKAGE: train/val overlap!"
    assert len(set(train_ids) & set(test_ids)) == 0, "LEAKAGE: train/test overlap!"
    assert len(set(val_ids)   & set(test_ids)) == 0, "LEAKAGE: val/test overlap!"
    print("\n[OK] No patient leakage detected across splits")

    # ── build slice-level split CSVs ─────────────────────────────────────
    # each CSV has: patient_id, filepath, label
    # grouped by patient so DataLoader can build MIL bags
    def get_split_df(patient_ids):
        return slices_df[slices_df["patient_id"].isin(patient_ids)].copy()

    train_df = get_split_df(train_ids)
    val_df   = get_split_df(val_ids)
    test_df  = get_split_df(test_ids)

    # ── save splits ──────────────────────────────────────────────────────
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,     index=False)
    test_df.to_csv(TEST_CSV,   index=False)

    # ── summary ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"{'Split':<10} {'Patients':>10} {'Slices':>10} {'Non-Dem':>10} {'Dem':>10}")
    print("="*50)

    for name, df, ids in [
        ("Train", train_df, train_ids),
        ("Val",   val_df,   val_ids),
        ("Test",  test_df,  test_ids)
    ]:
        total_patients = len(ids)
        total_slices   = len(df)
        n_nd = sum(1 for p in ids if p in set(nd_train + nd_val + nd_test)
                   and labels_df[labels_df["patient_id"]==p]["label"].values[0] == 0)
        n_d  = total_patients - n_nd
        print(f"{name:<10} {total_patients:>10} {total_slices:>10} {n_nd:>10} {n_d:>10}")

    print("="*50)
    print(f"{'Total':<10} {len(labels_df):>10} {len(slices_df):>10}")
    print()
    print(f"Train -> {TRAIN_CSV}")
    print(f"Val   -> {VAL_CSV}")
    print(f"Test  -> {TEST_CSV}")

    # ── class weights for weighted loss ──────────────────────────────────
    # only computed from train set to avoid test data influencing training
    train_labels = labels_df[labels_df["patient_id"].isin(train_ids)]["label"]
    n_total  = len(train_labels)
    n_nd     = (train_labels == 0).sum()
    n_d      = (train_labels == 1).sum()

    weight_nd = n_total / (2 * n_nd)
    weight_d  = n_total / (2 * n_d)

    print(f"\nClass weights for weighted loss (add to config.py):")
    print(f"  CLASS_WEIGHTS = [{weight_nd:.4f}, {weight_d:.4f}]")
    print(f"  Non-Demented weight : {weight_nd:.4f}")
    print(f"  Demented weight     : {weight_d:.4f}")


if __name__ == "__main__":
    split_dataset()