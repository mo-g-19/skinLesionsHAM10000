#!/usr/bin/env python

"""
Prepare HAM10000 dataset: split into train/val/test folders.

Instead of doing this manually, we use pandas and sklearn to split the dataset into train/val/test sets.
This will be run inside the main_experiments notebook
"""
import os
from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. CONFIG: adjust these if your paths are different
# ---------------------------------------------------------
base_dir = "/content/HAM10000"  # Folder where Kaggle files are
part1_dir = os.path.join(base_dir, "HAM10000_images_part_1")
part2_dir = os.path.join(base_dir, "HAM10000_images_part_2")
metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")

# Where we want: data_ham10000/train, val, test
output_root = "/content/data_ham10000"

val_frac = 0.15
test_frac = 0.15
random_seed = 42

# ---------------------------------------------------------
# 2. Load metadata and collect paths + labels
# ---------------------------------------------------------
os.makedirs(output_root, exist_ok=True)

meta = pd.read_csv(metadata_path)
print(meta.head())
print("Unique diagnosis labels:", meta["dx"].unique())

image_paths = []
labels = []
missing_count = 0

for _, row in meta.iterrows():
    image_id = row["image_id"]  # e.g. 'ISIC_0024306'
    label = row["dx"]           # e.g. 'mel'
    filename = image_id + ".jpg"

    p1 = os.path.join(part1_dir, filename)
    p2 = os.path.join(part2_dir, filename)

    if os.path.exists(p1):
        img_path = p1
    elif os.path.exists(p2):
        img_path = p2
    else:
        missing_count += 1
        continue

    image_paths.append(img_path)
    labels.append(label)

print("Total images found:", len(image_paths))
print("Missing images:", missing_count)

classes = sorted(list(set(labels)))
print("Classes:", classes)

# Put into a DataFrame for easier splitting
df = pd.DataFrame({"filepath": image_paths, "label": labels})

# ---------------------------------------------------------
# 3. Stratified train / val / test split
# ---------------------------------------------------------
if val_frac + test_frac <= 0 or val_frac + test_frac >= 1:
    raise ValueError("val_frac + test_frac must be in (0, 1).")

temp_frac = val_frac + test_frac

# train vs (val+test)
train_df, temp_df = train_test_split(
    df,
    test_size=temp_frac,
    stratify=df["label"],
    random_state=random_seed,
)

# val vs test (split temp 50/50 in the desired ratio)
relative_test_frac = test_frac / (val_frac + test_frac)

val_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_frac,
    stratify=temp_df["label"],
    random_state=random_seed,
)

print(f"Train: {len(train_df)} images")
print(f"Val:   {len(val_df)} images")
print(f"Test:  {len(test_df)} images")

# ---------------------------------------------------------
# 4. Copy files into output_root/split/label/
# ---------------------------------------------------------
def copy_split(split_df, split_name, root_dir):
    for _, row in split_df.iterrows():
        src = Path(row["filepath"])
        label = row["label"]
        dst_dir = Path(root_dir) / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)

print(f"Writing split images under: {output_root}")

copy_split(train_df, "train", output_root)
copy_split(val_df, "val", output_root)
copy_split(test_df, "test", output_root)

print("Done. Final structure:")
for split in ["train", "val", "test"]:
    split_dir = Path(output_root) / split
    print(split, "classes:", [p.name for p in split_dir.iterdir() if p.is_dir()])
