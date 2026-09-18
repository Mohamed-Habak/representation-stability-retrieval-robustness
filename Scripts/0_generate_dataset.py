"""
===============================================================================
CHESTMNIST TEST DATASET PREPARATION
===============================================================================

Purpose:
    Downloads the ChestMNIST test split at 224x224 resolution, saves the
    original test dataset, links its images to NIH ChestX-ray metadata, removes
    duplicate patients, and saves the resulting patient-deduplicated dataset.

Workflow:
    1. Define the input/output paths for the ChestMNIST dataset and its source
       metadata using os.path.join() and create the required directories with
       os.makedirs().

    2. Download the ChestMNIST test split at 224x224 resolution using the
       public ChestMNIST class from the medmnist package with:
           split="test"
           size=224
           as_rgb=True
           download=True

    3. Extract the downloaded images and labels from the dataset's .imgs and
       .labels attributes and save the complete, unmodified test split as a
       compressed NumPy .npz file using numpy.savez_compressed().

    4. Load the ChestMNIST split mapping and NIH metadata using
       pandas.read_csv().

    5. Restrict the mapping file to test-set images and construct the
       corresponding NIH "Image Index" filename by appending ".png" to each
       ChestMNIST image ID.

    6. Match ChestMNIST test images to their NIH Patient IDs using
       pandas.DataFrame.merge().

    7. Remove repeated images belonging to the same patient using
       pandas.DataFrame.drop_duplicates() on the "Patient ID" column, keeping
       the first occurrence for each patient.

    8. Extract the original dataset indices of the retained images and use
       NumPy array indexing to create the duplicate-free image and label arrays.

    9. Save the patient-deduplicated images and labels as a compressed NumPy
       .npz file using numpy.savez_compressed().

   10. Print a final report showing the original and duplicate-free dataset
       sizes and the locations of both saved datasets.

Input/source data:
    medmnist.ChestMNIST:
        ChestMNIST test split downloaded automatically at 224x224 resolution.

    data/Chestmnist_source_files/chestmnist_split_info.csv
        ChestMNIST image-to-split mapping used to identify test images.

    data/Chestmnist_source_files/Data_Entry_2017_v2020.csv
        NIH metadata used to obtain Patient IDs for the corresponding images.

Output:
    data/chestmnist_test/chestmnist_224.npz
        Complete ChestMNIST test split.

    data/chestmnist_test_no_duplicate_patients/
        chestmnist_test_no_duplicate_patients.npz
        Test split with duplicate patients removed.

Public/external packages and functions:
    medmnist:
        ChestMNIST

    NumPy:
        np.savez_compressed()

    pandas:
        pd.read_csv()
        DataFrame.merge()
        DataFrame.drop_duplicates()

    Python standard library:
        os.path.join()
        os.makedirs()

No custom Funcs_*.py functions are used in this script.

===============================================================================
"""

import os
import numpy as np
import pandas as pd
from medmnist import ChestMNIST


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORIGINAL_OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "chestmnist_test"
)

ORIGINAL_OUTPUT_FILE = os.path.join(
    ORIGINAL_OUTPUT_DIR, "chestmnist_224.npz"
)

DEDUP_OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "chestmnist_test_no_duplicate_patients"
)

DEDUP_OUTPUT_FILE = os.path.join(
    DEDUP_OUTPUT_DIR,
    "chestmnist_test_no_duplicate_patients.npz"
)

MAPPING_FILE = os.path.join(
    BASE_DIR, "data", "Chestmnist_source_files", "chestmnist_split_info.csv"
)

NIH_FILE = os.path.join(
    BASE_DIR, "data", "Chestmnist_source_files", "Data_Entry_2017_v2020.csv"
)

os.makedirs(ORIGINAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEDUP_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Download ChestMNIST test split
# ---------------------------------------------------------

print("Downloading ChestMNIST test split (224x224)...")

dataset = ChestMNIST(
    split="test",
    size=224,
    as_rgb=True,
    download=True
)

images = dataset.imgs
labels = dataset.labels

print(f"Downloaded images: {len(images)}")
print(f"Image shape: {images.shape}")
print(f"Labels shape: {labels.shape}")


# ---------------------------------------------------------
# 3. Save original dataset
# ---------------------------------------------------------

np.savez_compressed(
    ORIGINAL_OUTPUT_FILE,
    test_images=images,
    test_labels=labels
)

print(f"Original dataset saved to: {ORIGINAL_OUTPUT_FILE}")


# ---------------------------------------------------------
# 4. Load metadata
# ---------------------------------------------------------

print("Loading metadata...")

mapping = pd.read_csv(MAPPING_FILE)
nih = pd.read_csv(NIH_FILE)


# Keep test images
mapping = mapping[mapping["split"] == "test"].copy()

# Make filenames match NIH metadata
mapping["Image Index"] = mapping["image_id"] + ".png"


# ---------------------------------------------------------
# 5. Connect images to Patient IDs
# ---------------------------------------------------------

merged = mapping.merge(
    nih[["Image Index", "Patient ID"]],
    on="Image Index",
    how="inner"
)

print(f"Successfully matched images: {len(merged)}")


# ---------------------------------------------------------
# 6. Remove duplicate patients
# ---------------------------------------------------------

deduped = merged.drop_duplicates(
    subset="Patient ID",
    keep="first"
)

print(f"Before deduplication: {len(merged)}")
print(f"After deduplication: {len(deduped)}")


# ---------------------------------------------------------
# 7. Select corresponding images
# ---------------------------------------------------------

indices = deduped["index"].astype(int).to_numpy()

new_images = images[indices]
new_labels = labels[indices]


# ---------------------------------------------------------
# 8. Save duplicate-free dataset
# ---------------------------------------------------------

np.savez_compressed(
    DEDUP_OUTPUT_FILE,
    images=new_images,
    labels=new_labels
)


# ---------------------------------------------------------
# 9. Final report
# ---------------------------------------------------------

print()
print("=" * 60)
print("COMPLETE")
print("=" * 60)

print(f"Original dataset: {len(images)} images")
print(f"Duplicate-free dataset: {len(new_images)} images")

print(f"\nOriginal saved to:")
print(ORIGINAL_OUTPUT_FILE)

print(f"\nDuplicate-free saved to:")
print(DEDUP_OUTPUT_FILE)