"""
===============================================================================
MODEL AND EMBEDDING FUNCTIONS
===============================================================================

Purpose:
    Provides functions for building pretrained feature-extraction models and
    generating image embeddings at multiple resolutions.

Functions:
    build_vit()
        Builds a pretrained ViT-B/16 feature extractor and its 224x224
        ImageNet preprocessing pipeline.

    build_resnet()
        Builds a pretrained ResNet18 feature extractor and its 224x224
        ImageNet preprocessing pipeline.

    build_vgg()
        Builds a pretrained VGG16 feature extractor using the penultimate
        classifier representation and its 224x224 ImageNet preprocessing
        pipeline.

    compute_embeddings_batched()
        Processes images in batches at each requested resolution, generates
        embeddings, and keeps the resulting embeddings, labels, and image IDs
        aligned. Completed embeddings are moved to CPU memory to limit GPU use.

Returns:
    The model-building functions return a feature extractor and its
    preprocessing pipeline. compute_embeddings_batched() returns embeddings
    grouped by resolution, along with the corresponding labels and image IDs.

Note:
    This module only handles model construction and embedding generation.
    Similarity and retrieval calculations are performed by later pipeline
    stages.

===============================================================================
"""


import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Model setup ----------

def build_vit():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.eval().to(device)
    model.heads.head = nn.Identity()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    ])
    return model, preprocess

def build_resnet():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval().to(device)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    ])
    return feature_extractor, preprocess

def build_vgg():
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    model.eval().to(device)
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *list(model.classifier.children())[:-1]
    ).to(device)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    ])
    return feature_extractor, preprocess

# ---------- Batched GPU Processing ----------

def compute_embeddings_batched(feature_extractor, preprocess, dataloader, limit, sizes, base_resolution):
   
    embeddings_by_size = {s: [] for s in sizes}
    all_labels = []
    all_image_ids = []
    total_processed = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            batch_size = imgs.size(0)
            if total_processed >= limit:
                break

            if total_processed + batch_size > limit:
                rem = limit - total_processed
                imgs = imgs[:rem]
                labels = labels[:rem]
                batch_size = rem

            imgs = imgs.to(device)

            for s in sizes:
                if s == base_resolution:
                    downscaled = imgs
                else:
                    downscaled = TF.resize(imgs, [s, s], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

                preprocessed = preprocess(downscaled)
                emb = feature_extractor(preprocessed).flatten(1)
                embeddings_by_size[s].append(emb.cpu())  # move off GPU immediately, don't hoard VRAM

            all_labels.extend(labels.tolist())
            all_image_ids.extend(range(total_processed, total_processed + batch_size))
            total_processed += batch_size
            print(f"Processed {total_processed}/{limit} images...")

    # Concatenate batches into single tensors per resolution
    embeddings_by_size = {s: torch.cat(v, dim=0) for s, v in embeddings_by_size.items()}

    return embeddings_by_size, all_labels, all_image_ids