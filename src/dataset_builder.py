# src/dataset_builder.py

import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, DATA_DIR

MRI_SLICES_DIR = os.path.join(DATA_DIR, "mri_slices")
OUTPUT_FILE    = os.path.join(DATA_DIR, "labels.csv")
SLICES_FILE    = os.path.join(DATA_DIR, "slices.csv")

CLASS_MAP = {
    "Non Demented":      0,
    "Very mild Dementia": 1,
    "Mild Dementia":      1,
    "Moderate Dementia":  1
}

def extract_patient_id(filename):
    """
    OAS1_0001_MR1_mpr-1_101.jpg → OAS1_0001
    """
    parts = os.path.splitext(filename)[0].split("_")
    if len(parts) < 2:
        return None
    return parts[0] + "_" + parts[1]


def build_labels_csv():

    patient_labels  = {}   # patient_id → label
    conflicts       = []   # patients with slices in multiple class folders
    all_slices      = []   # every slice row: patient_id, filepath, label

    for class_name, label in CLASS_MAP.items():

        class_path = os.path.join(MRI_SLICES_DIR, class_name)

        if not os.path.isdir(class_path):
            print(f"WARNING: folder not found → {class_path}")
            continue

        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        print(f"{class_name}: {len(files)} slices found")

        for file in files:

            patient_id = extract_patient_id(file)

            if patient_id is None:
                print(f"  Skipping unrecognised filename: {file}")
                continue

            filepath = os.path.join(class_path, file)

            # detect label conflict
            if patient_id in patient_labels:
                if patient_labels[patient_id] != label:
                    conflicts.append({
                        "patient_id": patient_id,
                        "existing_label": patient_labels[patient_id],
                        "new_label": label,
                        "conflicting_file": file
                    })
                    continue   # skip conflicting slice
            else:
                patient_labels[patient_id] = label

            all_slices.append({
                "patient_id": patient_id,
                "filepath":   filepath,
                "label":      label
            })

    # ── labels.csv  (one row per patient) ──────────────────────────────
    labels_df = pd.DataFrame([
        {"patient_id": pid, "label": lbl}
        for pid, lbl in patient_labels.items()
    ])
    labels_df.sort_values("patient_id", inplace=True)
    labels_df.to_csv(OUTPUT_FILE, index=False)

    # ── slices.csv  (one row per image slice) ──────────────────────────
    slices_df = pd.DataFrame(all_slices)
    slices_df.sort_values(["patient_id", "filepath"], inplace=True)
    slices_df.to_csv(SLICES_FILE, index=False)

    # ── summary ────────────────────────────────────────────────────────
    print("\n" + "="*45)
    print("labels.csv ->", OUTPUT_FILE)
    print("slices.csv ->", SLICES_FILE)
    print("="*45)
    print(f"Total patients : {len(labels_df)}")
    print(f"Total slices   : {len(slices_df)}")
    print("\nClass distribution (patients):")
    counts = labels_df["label"].value_counts().sort_index()
    print(f"  Non-Demented (0) : {counts.get(0, 0)}")
    print(f"  Demented     (1) : {counts.get(1, 0)}")

    ratio = counts.get(0, 0) / max(counts.get(1, 0), 1)
    print(f"\nImbalance ratio   : {ratio:.2f}:1  (non:demented)")
    if ratio > 2:
        print("  Significant imbalance — use weighted loss during training")

    if conflicts:
        print(f"\n{len(conflicts)} label conflicts detected:")
        for c in conflicts:
            print(f"  {c['patient_id']} — existing:{c['existing_label']} "
                  f"new:{c['new_label']} file:{c['conflicting_file']}")
    else:
        print("\nNo label conflicts detected")

    print("\nSample from labels.csv:")
    print(labels_df.head())

    return labels_df, slices_df


if __name__ == "__main__":
    build_labels_csv()