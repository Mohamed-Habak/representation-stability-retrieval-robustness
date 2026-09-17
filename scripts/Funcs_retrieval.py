"""
===============================================================================
RETRIEVAL FUNCTIONS
===============================================================================

Purpose:
    Provides functions for evaluating image retrieval by comparing degraded
    query embeddings against a base-resolution embedding gallery.

Functions:
    retrieval_rank()
        Calculates the retrieval rank of the correct gallery image for each
        query. A rank of 0 means the correct image was retrieved first.

    retrieval_rank_and_distractor_similarity()
        Calculates the correct-image retrieval rank and the highest similarity
        to any incorrect gallery image, allowing the retrieval margin between
        the true match and its nearest distractor to be measured.

===============================================================================
"""

import torch
import torch.nn.functional as F


def retrieval_rank(gallery_emb, query_emb):

    g = F.normalize(gallery_emb, dim=1)
    q = F.normalize(query_emb, dim=1)

    sims = q @ g.T  # (N x N), sims[i, j] = similarity of query i to gallery image j

    n = sims.shape[0]
    true_sims = sims[torch.arange(n), torch.arange(n)].unsqueeze(1)

    # Rank = number of gallery images strictly more similar than the true
    # match. Ties count as beating the true match (conservative).
    rank = (sims >= true_sims).sum(dim=1) - 1
    return rank


def retrieval_rank_and_distractor_similarity(gallery_emb, query_emb):
    
    g = F.normalize(gallery_emb, dim=1)
    q = F.normalize(query_emb, dim=1)

    sims = q @ g.T  # (N x N)
    n = sims.shape[0]

    true_sims = sims[torch.arange(n), torch.arange(n)]

    sims_no_self = sims.clone()
    sims_no_self[torch.arange(n), torch.arange(n)] = -float("inf")
    nearest_distractor_sim = sims_no_self.max(dim=1).values

    rank = (sims > true_sims.unsqueeze(1)).sum(dim=1)

    return rank, nearest_distractor_sim