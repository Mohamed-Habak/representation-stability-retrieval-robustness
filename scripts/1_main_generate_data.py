"""
===============================================================================
EMBEDDING GENERATION
===============================================================================

Purpose:
    Generates image embeddings for each configured dataset, model, and image
    resolution, then saves them for downstream stability and retrieval analysis.

Workflow:
    1. Load each configured dataset:
       - ChestMNIST: loaded using the custom ChestMNISTDataset class defined here.
       - STL-10: loaded using torchvision.datasets.STL10.

    2. Build the three feature-extraction models using custom functions imported
       from Funcs_gen.py:
       - build_resnet()
       - build_vgg()
       - build_vit()

    3. Compute embeddings for the base and degraded image resolutions using the
       custom compute_embeddings_batched() function imported from Funcs_gen.py.

    4. Save the generated embeddings together with image labels and image IDs
       as PyTorch .pt files using torch.save().

Inputs:
    Dataset configuration:
        DATASETS from config.py
    Run configuration:
        RUN_NAME from config.py
    Dataset files:
        data/

Output:
    outputs/<RUN_NAME>/embeddings/<dataset>/
        <model>_<limit>_embeddings.pt

Custom functions used:
    Funcs_gen.py:
        build_resnet()
        build_vgg()
        build_vit()
        compute_embeddings_batched()

Public/external functions and classes used:
    torchvision.datasets.STL10
    torch.utils.data.DataLoader
    torch.save()

===============================================================================
"""

import os
import warnings
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10

from Funcs_gen import (
    build_resnet,
    build_vgg,
    build_vit,
    compute_embeddings_batched
)

from config import DATASETS, RUN_NAME

warnings.filterwarnings("ignore")


# ============================================================
# ChestMNIST Dataset
# ============================================================

class ChestMNISTDataset(Dataset):

    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = self.transform(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image, label


# ============================================================
# Dataset loading dispatch — add new datasets here only
# ============================================================

def load_dataset(dataset_name, base_dir):

    if dataset_name == "chestmnist":

        dataset_path = os.path.join(base_dir, "data", "chestmnist_test_no_duplicate_patients", "chestmnist_test_no_duplicate_patients.npz")
        dataset = ChestMNISTDataset(dataset_path)
        print(f"Dataset: ChestMNIST ({len(dataset)} images loaded)")
        return dataset

    elif dataset_name == "stl10":

        dataset = STL10(root=os.path.join(base_dir, "data"), split="test", download=True, transform=transforms.ToTensor())
        print(f"Dataset: STL-10 ({len(dataset)} images loaded)")
        return dataset

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}' in config.DATASETS. "
            f"Add a loader for it in load_dataset()."
        )


# ============================================================
# Main
# ============================================================

def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(BASE_DIR, "outputs", RUN_NAME)

    models_to_run = {
        "ResNet18": build_resnet,
        "VGG16": build_vgg,
        "ViT": build_vit
    }

    for dataset_name, dataset_cfg in DATASETS.items():

        print("=" * 60)
        print(f"DATASET: {dataset_name}  (limit={dataset_cfg['limit']}, "
              f"base_resolution={dataset_cfg['base_resolution']})")
        print("=" * 60)

        limit = dataset_cfg["limit"]
        sizes = dataset_cfg["sizes"]
        base_resolution = dataset_cfg["base_resolution"]

        emb_dir = os.path.join(run_dir, "embeddings", dataset_name)
        os.makedirs(emb_dir, exist_ok=True)

        dataset = load_dataset(dataset_name, BASE_DIR)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        for model_name, model_builder in models_to_run.items():

            print("-" * 60)
            print(f"Running {model_name} on {dataset_name}")
            print("-" * 60)

            extractor, preprocess = model_builder()

            embeddings_by_size, labels, image_ids = compute_embeddings_batched(extractor, preprocess, dataloader, limit, sizes, base_resolution)

            save_path = os.path.join(emb_dir, f"{model_name.lower()}_{limit}_embeddings.pt")

            torch.save(
                {
                    "embeddings": embeddings_by_size,
                    "labels": labels,
                    "image_ids": image_ids,
                },
                save_path
            )

            print(f"Saved: {save_path}\n")

        print(f"{dataset_name} complete.\n")

    print("=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()