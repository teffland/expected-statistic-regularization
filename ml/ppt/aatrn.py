# Copyright (c) 2021 Kemal Kurniawan

from typing import Optional
import torch

from einops import rearrange
from torch import BoolTensor, Tensor

from ml.ppt.crf import DepTreeCRF


def compute_aatrn_loss(
    scores: Tensor,
    aa_mask: BoolTensor,
    mask: Optional[BoolTensor] = None,
    projective: bool = False,
    multiroot: bool = True,
) -> Tensor:
    assert aa_mask.shape == scores.shape
    masked_scores = scores.masked_fill(~aa_mask, -1e9)
    crf = DepTreeCRF(masked_scores, mask, projective, multiroot)
    crf_z = DepTreeCRF(scores, mask, projective, multiroot)
    return -crf.log_partitions().sum() + crf_z.log_partitions().sum()


def compute_ambiguous_tags_mask(scores: Tensor, threshold: float = 0.95) -> BoolTensor:
    assert scores.dim() == 3
    bsz, slen, n_types = scores.shape
    assert 0 <= threshold <= 1
    marginals = torch.softmax(scores, dim=2)
    og_marginals = marginals.clone()

    # select high-prob arcs until their cumulative probability exceeds threshold
    marginals, orig_indices = marginals.sort(dim=2, descending=True)
    tag_mask = marginals.cumsum(dim=2) < threshold

    # mark the tag that makes the cum sum exceeds threshold
    last_idx = tag_mask.long().sum(dim=2, keepdim=True).clamp(max=slen * n_types - 1)
    tag_mask = tag_mask.scatter(2, last_idx, True)

    # restore the tag_mask order and shape
    _, restore_indices = orig_indices.sort(dim=2)
    tag_mask = tag_mask.gather(2, restore_indices)

    return tag_mask


def compute_ambiguous_arcs_mask(
    scores: Tensor, threshold: float = 0.95, projective: bool = False, multiroot: bool = True,
) -> BoolTensor:
    assert scores.dim() == 4
    bsz, slen, _, n_types = scores.shape
    assert 0 <= threshold <= 1

    crf = DepTreeCRF(scores, projective=projective, multiroot=multiroot)
    marginals = crf.marginals()

    # select high-prob arcs until their cumulative probability exceeds threshold
    marginals = rearrange(marginals, "bsz hlen dlen ntypes -> bsz dlen (hlen ntypes)")
    marginals, orig_indices = marginals.sort(dim=2, descending=True)
    arc_mask = marginals.cumsum(dim=2) < threshold

    # mark the arc that makes the cum sum exceeds threshold
    last_idx = arc_mask.long().sum(dim=2, keepdim=True).clamp(max=slen * n_types - 1)
    arc_mask = arc_mask.scatter(2, last_idx, True)

    # restore the arc_mask order and shape
    _, restore_indices = orig_indices.sort(dim=2)
    arc_mask = arc_mask.gather(2, restore_indices)

    # ensure best tree is selected
    # each shape: (bsz, slen)
    best_heads, best_types = crf.argmax()
    best_idx = best_heads * n_types + best_types
    arc_mask = arc_mask.scatter(2, best_idx.unsqueeze(2), True)

    arc_mask = rearrange(arc_mask, "bsz dlen (hlen ntypes) -> bsz hlen dlen ntypes", hlen=slen)
    return arc_mask
