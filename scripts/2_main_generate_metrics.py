"""
===============================================================================
PAIRWISE COSINE METRIC GENERATION
===============================================================================

Purpose:
    Computes embedding stability by measuring the cosine similarity between
    each image's base-resolution embedding and its embedding at every
    configured resolution.

Workflow:
    1. Load the previously generated embedding .pt file for each dataset and
       model using torch.load().

    2. Extract the base-resolution embeddings as the reference representation.

    3. For every configured resolution, compute the cosine similarity between
       the base-resolution embedding and the corresponding resolution-specific
       embedding using the custom compute_pairwise_metrics() function imported
       from Funcs_metrics.py.

    4. Store the cosine similarity for each image together with its label and
       image ID.

    5. Combine the results from ResNet18, VGG16, and ViT into one DataFrame
       using pandas.concat() and save the result as a CSV file with
       pandas.DataFrame.to_csv().

Input:
    outputs/<RUN_NAME>/embeddings/<dataset>/
        <model>_<limit>_embeddings.pt

Output:
    outputs/<RUN_NAME>/csv/<dataset>_cosine.csv

Custom functions used:
    Funcs_metrics.py:
        compute_pairwise_metrics()

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
from Funcs_metrics import compute_pairwise_metrics
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

            base_emb = embeddings_by_size[base_resolution]

            rows = [{"label": labels[i], "image_id": image_ids[i]} for i in range(len(image_ids))]

            for s in sizes:
                cosine = compute_pairwise_metrics(base_emb, embeddings_by_size[s])
                for i in range(len(image_ids)):
                    rows[i][s] = float(cosine[i])

            df = pd.DataFrame(rows)
            df.insert(0, "Model", model_name)
            model_dfs.append(df)

            print(f"{model_name} complete.\n")

        combined_df = pd.concat(model_dfs, ignore_index=True)
        output_path = os.path.join(csv_dir, f"{dataset_name}_cosine.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}\n")

    print("=" * 60), print("METRIC GENERATION COMPLETE"), print("=" * 60)


if __name__ == "__main__":
    main()