"""
===============================================================================
RETRIEVAL RANK GENERATION
===============================================================================

Purpose:
    Measures image retrieval performance by determining the rank of the
    corresponding base-resolution image when each resolution-specific
    embedding is used as a query against the base-resolution embedding gallery.

Workflow:
    1. Load the previously generated embedding .pt file for each dataset and
       model using torch.load().

    2. Use the embeddings at the configured base resolution as the retrieval
       gallery.

    3. For every configured resolution, compare its embeddings against the
       base-resolution gallery and determine the rank of the corresponding
       original image using the custom retrieval_rank() function imported
       from Funcs_retrieval.py.

    4. Store the resulting retrieval rank for each image together with its
       label and image ID.

    5. Combine the retrieval results from ResNet18, VGG16, and ViT using
       pandas.concat() and save them as one CSV file per dataset using
       DataFrame.to_csv().

Input:
    outputs/<RUN_NAME>/embeddings/<dataset>/
        <model>_<limit>_embeddings.pt

Output:
    outputs/<RUN_NAME>/csv/<dataset>_retrieval.csv

Custom functions used:
    Funcs_retrieval.py:
        retrieval_rank()

Public/external functions used:
    torch.load()
    pandas.DataFrame()
    pandas.concat()
    DataFrame.to_csv()

===============================================================================
"""

import os
import warnings
import torch
import pandas as pd
from Funcs_retrieval import retrieval_rank
from config import DATASETS, RUN_NAME
warnings.filterwarnings("ignore")


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(BASE_DIR, "outputs", RUN_NAME)
    csv_dir = os.path.join(run_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    model_names = ["ResNet18", "VGG16", "ViT"]
    file_prefixes = {"ResNet18": "resnet18", "VGG16": "vgg16", "ViT": "vit"}

    for dataset_name, dataset_cfg in DATASETS.items():

        print("=" * 60)
        print(f"DATASET: {dataset_name}")
        print("=" * 60)

        limit = dataset_cfg["limit"]
        sizes = dataset_cfg["sizes"]
        base_resolution = dataset_cfg["base_resolution"]

        emb_dir = os.path.join(run_dir, "embeddings", dataset_name)

        model_dfs = []

        for model_name in model_names:
            print("-" * 60), print(f"Processing {model_name}"), print("-" * 60)

            prefix = file_prefixes[model_name]
            emb_path = os.path.join(emb_dir, f"{prefix}_{limit}_embeddings.pt")
            data = torch.load(emb_path)

            labels = data["labels"]
            image_ids = data["image_ids"]
            embeddings_by_size = data["embeddings"]

            gallery = embeddings_by_size[base_resolution]

            rows = [{"label": labels[i], "image_id": image_ids[i]} for i in range(len(image_ids))]

            for s in sizes:
                rank = retrieval_rank(gallery, embeddings_by_size[s])
                for i in range(len(image_ids)):
                    rows[i][s] = int(rank[i])

            df = pd.DataFrame(rows)
            df.insert(0, "Model", model_name)
            model_dfs.append(df)

            print(f"{model_name} complete.\n")

        combined_df = pd.concat(model_dfs, ignore_index=True)
        output_path = os.path.join(csv_dir, f"{dataset_name}_retrieval.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}\n")

    print("=" * 60), print("RETRIEVAL GENERATION COMPLETE"), print("=" * 60)


if __name__ == "__main__":
    main()