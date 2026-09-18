"""
===============================================================================
PAIRWISE EMBEDDING METRICS
===============================================================================

Purpose:
    Provides the function used to compare base-resolution embeddings with
    embeddings generated from other image resolutions.

Functions:
    compute_pairwise_metrics()
        Computes the per-image cosine similarity between the base-resolution
        embedding and the corresponding embedding at another resolution.

Inputs:
    base_emb
        Tensor containing the base-resolution embeddings.

    other_emb
        Tensor containing the embeddings to compare against the base embeddings.

Returns:
    A tensor containing one cosine-similarity value per image.

===============================================================================
"""

import torch.nn.functional as F

def compute_pairwise_metrics(base_emb, other_emb):
    cosine = F.cosine_similarity(base_emb, other_emb, dim=1)
    return cosine