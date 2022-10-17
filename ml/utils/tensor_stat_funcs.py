""" Generic functions for computing statistics on a dataset. 

Each function takes the data from a UDPDataset as first arg with addition optional args and returns a float.
"""
import math
from time import time

import torch
import torch.distributions as D
import opt_einsum as oe

from ml.utils.stat_funcs import (
    dirs,
    dists,
    valencies,
    pos_tags,
    coarse_dep_tags,
    coarse_dep_tags_set,
    clean_dep,
    parse_argstr,
)

# This has to be at the top
def cached_stat_func(get_stat_tensor):
    """Decorator that converts a raw stat function to an efficient cached implementation.

    The decorated function expects tensors, vocab, and zero or more indices, and passthrough kwargs.
    If there are zero indices, the full tensor is returned
    """

    def wrapped(tensors, vocab, *indexes, preindex=True, mean_after_loss=False, no_cache=False, **kwargs):
        cache_name = get_stat_tensor.__name__
        do_mean = not mean_after_loss
        if indexes:
            idxs = get_token_index(indexes, vocab)
            # print("\nindexes", indexes, idxs, flush=True)

        # print("cache", cache_name)
        if cache_name not in tensors:
            if preindex:
                pre_indexes = []
                post_indexes = []
                for index in idxs:
                    # print("index", index)
                    if type(index) is int:
                        # A single index will always be at the first (and only) element
                        pre_index = torch.LongTensor([index])
                        post_index = torch.LongTensor([0])
                    else:
                        pre_index = index.unique()
                        aslist = pre_index.detach().cpu().numpy().tolist()
                        post_index = torch.LongTensor([aslist.index(i) for i in index])
                    # print(f"preindex: {index}\npre:{pre_index}\npost:{post_index}")
                    pre_indexes.append(pre_index)
                    post_indexes.append(post_index)
                if mean_after_loss:
                    post_indexes = [slice(None)] + post_indexes  # save the batch dim

                # print(f"preindex reduce {[index.shape for index in indexes]} to {[idx.shape for idxs in pre_indexes]}")
                stat_tensor = get_stat_tensor(tensors, indexes=pre_indexes, do_mean=do_mean, **kwargs)
                tensors[f"{cache_name}_post_indexes"] = post_indexes
            else:
                stat_tensor = get_stat_tensor(tensors, do_mean=do_mean, **kwargs)
            tensors[cache_name] = stat_tensor
        stat_tensor = tensors[cache_name]
        if no_cache:
            tensors.pop(cache_name)
        if preindex:
            idxs = tensors[f"{cache_name}_post_indexes"]
            return stat_tensor[idxs]
        elif indexes:
            if mean_after_loss:
                idxs = (slice(None),) + tuple(idxs)
            return stat_tensor[idxs]
        else:
            return stat_tensor

    return wrapped


# =========================
# Stat function definitions
# =========================
@cached_stat_func
def pos_freq(tensors, sample=False, indexes=None, **kwargs):
    """Marginal prob of POS tag in any position over all sentences."""
    pos_ps, mask = get_pos_rep(tensors, sample=sample), tensors["mask"]
    N_total = get_N_total_edges(tensors)
    indexes = (indexes[0], None) if indexes is not None else None
    E_counts = contract("bit,bi->t", pos_ps, mask, indexes=indexes)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def pos_sent_freq(tensors, sample=False, indexes=None, do_mean=True, **kwargs):
    """Marginal prob of POS tag in any position over all sentences."""
    pos_ps, mask = get_pos_rep(tensors, sample=sample), tensors["mask"]

    # if indexes:
    #     pos_ps = pos_ps[:,:,indexes[0]]
    E_counts = contract("bit,bi->bt", pos_ps, mask)
    Ns = get_Ns(tensors).reshape(-1, 1)
    E_freq = E_counts / Ns
    if do_mean:
        E_freq = E_freq.mean(dim=0)
    return E_freq


@cached_stat_func
def dep_freq(tensors, sample=False, **kwargs):
    """Marginal prob of any dependency label in any edge over all sentences."""
    arc_ps, label_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_label_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bijr->r", arc_ps * mask_adj, label_ps)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def dep_sent_freq(tensors, sample=False, do_mean=True, **kwargs):
    """Marginal prob of any dependency label in any edge over all sentences."""
    arc_ps, label_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_label_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    Ns = get_Ns(tensors).reshape(-1, 1)
    E_counts = contract("bij,bijr->br", arc_ps * mask_adj, label_ps)
    E_freq = E_counts / Ns
    if do_mean:
        E_freq = E_freq.mean(dim=0)
    return E_freq


@cached_stat_func
def pos_pair_freq(tensors, sample=False, indexes=None, **kwargs):
    """Marginal prob of seeing `left_tag` at position i and `right_tag` at i+1."""
    pos_ps, mask_adj = get_pos_rep(tensors, sample=sample), tensors["mask_adj"]
    N_total = get_N_total_edges(tensors)
    pair_mask = mask_adj * torch.diag(torch.ones(pos_ps.shape[1] - 1), 1).unsqueeze(0).to(mask_adj.device)
    indexes = (indexes[0], indexes[1], None) if indexes is not None else None
    E_counts = contract("bil,bjr,bij->lr", pos_ps, pos_ps, pair_mask, indexes=indexes)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_freq(tensors, sample=False, indexes=None, **kwargs):
    """Marginal prob of `head_tag` being the head of an edge over all sentences."""
    arc_ps, pos_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    if indexes:
        pos_ps = pos_ps[:, :, indexes[0]]
    E_counts = contract("bij,bit->t", arc_ps * mask_adj, pos_ps)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_sent_freq(tensors, sample=False, indexes=None, do_mean=True, **kwargs):
    """Marginal prob of `head_tag` being the head of an edge over all sentences."""
    arc_ps, pos_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    Ns = get_Ns(tensors).reshape(-1, 1)
    if indexes:
        pos_ps = pos_ps[:, :, indexes[0]]
    E_counts = contract("bij,bit->bt", arc_ps * mask_adj, pos_ps)
    E_freq = E_counts / Ns
    if do_mean:
        E_freq = E_freq.mean(dim=0)
    return E_freq


@cached_stat_func
def tail_freq(tensors, sample=False, **kwargs):
    """Marginal prob of `tail_tag` being the tail of an edge over all sentences."""
    arc_ps, pos_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_pos_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bjt->t", arc_ps * mask_adj, pos_ps)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def tail_sent_freq(tensors, sample=False, do_mean=True, **kwargs):
    """Marginal prob of `tail_tag` being the tail of an edge over all sentences."""
    arc_ps, pos_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_pos_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    Ns = get_Ns(tensors).reshape(-1, 1)
    E_counts = contract("bij,bjt->bt", arc_ps * mask_adj, pos_ps)
    E_freq = E_counts / Ns
    if do_mean:
        E_freq = E_freq.mean(dim=0)
    return E_freq


@cached_stat_func
def head_dep_freq(tensors, sample=False, **kwargs):
    """Marginal prob of (head,dep) pairs in any edge over all sentences."""
    arc_ps, pos_ps, label_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bih,bijr->hr", arc_ps * mask_adj, pos_ps, label_ps)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def tail_dep_freq(tensors, sample=False, **kwargs):
    """Marginal prob of (tail,dep) pairs in any edge over all sentences."""
    arc_ps, pos_ps, label_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_pos_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bjt,bijr->tr", arc_ps * mask_adj, pos_ps, label_ps)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_tail_freq(tensors, sample=False, **kwargs):
    """Marginal prob of (head,tail) POS pairs in any edge over all sentences."""
    arc_ps, pos_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bih,bjt->ht", arc_ps * mask_adj, pos_ps, pos_ps)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_tail_dep_freq(tensors, sample=False, **kwargs):
    """Marginal prob of (head,tail,dep) edges in any edge over all sentences."""
    arc_ps, pos_ps, label_ps, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bih,bjt,bijr->htr", arc_ps * mask_adj, pos_ps, pos_ps, label_ps)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def dir_freq(tensors, sample=False, **kwargs):
    """Marginal prob of any dependency label in any edge over all sentences."""
    arc_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bijd->d", arc_ps, dir_mask_adj)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_dir_freq(tensors, sample=False, **kwargs):
    """Marginal prob of head tag having an edge in direction dir over all sentences."""
    arc_ps, pos_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bih,bijd->hd", arc_ps, pos_ps, dir_mask_adj)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def tail_dir_freq(tensors, sample=False, **kwargs):
    """Marginal prob of tail tag having an edge in direction dir over all sentences."""
    arc_ps, pos_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_pos_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bjt,bijd->td", arc_ps, pos_ps, dir_mask_adj)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def dep_dir_freq(tensors, sample=False, **kwargs):
    """Marginal prob of dependency in direction dir over all sentences."""
    arc_ps, label_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_label_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bijr,bijd->rd", arc_ps, label_ps, dir_mask_adj)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_dep_dir_freq(tensors, sample=False, **kwargs):
    """Marginal prob of (tail,dep) pairs in direction dir over all sentences."""
    arc_ps, pos_ps, label_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bih,bijr,bijd->hrd", arc_ps, pos_ps, label_ps, dir_mask_adj)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def tail_dep_dir_freq(tensors, sample=False, **kwargs):
    """Marginal prob of (tail,dep) pairs in direction dir over all sentences."""
    arc_ps, pos_ps, label_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_pos_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bjt,bijr,bijd->trd", arc_ps, pos_ps, label_ps, dir_mask_adj)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_tail_dir_freq(tensors, sample=False, indexes=None, **kwargs):
    """Marginal prob of (head,tail) POS pairs in direction dir over all sentences."""
    arc_ps, pos_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_edges(tensors)
    if indexes:
        head_ps = pos_ps[:, :, indexes[0]]
        tail_ps = pos_ps[:, :, indexes[1]]
        dir_mask_adj = dir_mask_adj[:, :, :, indexes[2]]

        print(f"head_ps, tail_ps, dir_mask_adj: {head_ps.shape}, {tail_ps.shape}, {dir_mask_adj.shape}")
        print(f"idxs: {indexes[0].shape}, {indexes[1].shape}, {indexes[2].shape}")
    else:
        head_ps = pos_ps
        tail_ps = pos_ps
    E_counts = contract("bij,bih,bjt,bijd->htd", arc_ps, head_ps, tail_ps, dir_mask_adj)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_tail_dep_dir_freq(tensors, sample=False, indexes=None, **kwargs):
    """Marginal prob of (head,tail,dep) tuples in direction dir over all sentences."""
    arc_ps, pos_ps, label_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    indexes = [None] + list(indexes) if indexes is not None else None
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bih,bjt,bijr,bijd->htrd", arc_ps, pos_ps, pos_ps, label_ps, dir_mask_adj, indexes=indexes)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_tail_dep_dir_sent_freq(tensors, sample=False, do_mean=True, **kwargs):
    """Marginal prob of (head,tail,dep) tuples in direction dir over all sentences."""
    arc_ps, pos_ps, label_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
    )
    Ns = get_Ns(tensors).reshape(-1, 1, 1, 1, 1)
    E_counts = contract("bij,bih,bjt,bijr,bijd->bhtrd", arc_ps, pos_ps, pos_ps, label_ps, dir_mask_adj)
    E_freq = E_counts / Ns
    if do_mean:
        E_freq = E_freq.mean(dim=0)
    return E_freq


# =======================
# Distance marginal probs
# =======================
def bin_distance(dist):
    if dist <= 0.5:
        return "root"
    elif 0.5 < dist <= 1.5:
        return "adjacent"
    elif 1.5 < dist <= 3.5:
        return "close"
    elif 3.5 < dist <= 7.5:
        return "near"
    else:
        return "far"


@cached_stat_func
def dist_freq(tensors, sample=False, **kwargs):
    """Marginal prob of edge distance categories."""
    arc_ps, dist_mask, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_dist_mask(tensors),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bijz->z", arc_ps * mask_adj, dist_mask)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def dep_dist_freq(tensors, sample=False, **kwargs):
    """Marginal prob of dependency with dist."""
    arc_ps, label_ps, dist_mask, mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_label_rep(tensors, sample=sample),
        get_dist_mask(tensors),
        tensors["mask_adj"],
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bijr,bijz->rz", arc_ps * mask_adj, label_ps, dist_mask)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def dir_dist_freq(tensors, sample=False, **kwargs):
    """Marginal prob of an arc going in direction with dist."""
    arc_ps, dir_mask_adj, dist_mask = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_dir_mask_adj(tensors),
        get_dist_mask(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bijd,bijz->dz", arc_ps, dir_mask_adj, dist_mask)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def dep_dir_dist_freq(tensors, sample=False, **kwargs):
    """Marginal prob of dependency going in direction with dist."""
    arc_ps, label_ps, dir_mask_adj, dist_mask = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_label_rep(tensors, sample=sample),
        get_dir_mask_adj(tensors),
        get_dist_mask(tensors),
    )
    N_total = get_N_total_edges(tensors)
    E_counts = contract("bij,bijr,bijd,bijz->rdz", arc_ps, label_ps, dir_mask_adj, dist_mask)
    E_freq = E_counts / N_total
    return E_freq


# ==================
# Head Valency stats
# ==================
def bin_valency(count):
    if count <= 0.5:
        return "zero"
    elif 0.5 < count <= 1.5:
        return "one"
    elif 1.5 < count <= 2.5:
        return "two"
    elif 2.5 < count <= 3.5:
        return "three"
    else:
        return "many"


@cached_stat_func
def valency_freq(tensors, sample=True, **kwargs):
    """Marginal prob that words some word heads `target_bin` edges for any word over all sentences."""

    arc_sample, mask = get_arc_rep(tensors, sample=sample, allow_root=False), tensors["mask"]
    valencies = contract("bij,bi->bi", arc_sample, mask).unsqueeze(2)
    valency_anchors = torch.arange(0, 5).float().reshape(1, 1, 5).to(valencies.device)

    valency_scores = -((valencies - valency_anchors) ** 2)  # B, N, V
    temp = sample if type(sample) is float else (1.0 if sample else 0.0)
    if temp == 0.0:
        max_idx = valency_scores.argmax(2, keepdim=True)
        valency_indicators = torch.zeros_like(valency_scores)
        valency_indicators.scatter_(2, max_idx, 1)
    else:
        valency_indicators = torch.softmax(valency_scores / temp, dim=2)

    E_counts = contract("biv,bi->v", valency_indicators, mask)
    N_total = get_N_total_edges(tensors)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def pos_valency_freq(tensors, sample=True, pos_sample=False, **kwargs):
    """Marginal prob that words with tag `head_tag` heads `target_bin` edges for any word over all sentences."""
    arc_sample, pos_ps, mask = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=pos_sample),
        tensors["mask"],
    )
    valencies = contract("bij,bi->bi", arc_sample, mask).unsqueeze(2)
    valency_anchors = torch.arange(0, 5).float().reshape(1, 1, 5).to(valencies.device)

    valency_scores = -((valencies - valency_anchors) ** 2)  # B, N, V
    temp = sample if type(sample) is float else (1.0 if sample else 0.0)
    if temp == 0.0:
        max_idx = valency_scores.argmax(2, keepdim=True)
        valency_indicators = torch.zeros_like(valency_scores)
        valency_indicators.scatter_(2, max_idx, 1)
    else:
        valency_indicators = torch.softmax(valency_scores / temp, dim=2)

    E_counts = contract("biv,bih,bi->hv", valency_indicators, pos_ps, mask)
    N_total = get_N_total_edges(tensors)
    E_freq = E_counts / N_total
    return E_freq


# ==========================
# Arc-Pair Sibling Marginals
# ==========================
@cached_stat_func
def head_tail1_tail2_dir1_dir2_freq(tensors, sample=True, pos_sample=False, **kwargs):
    """Marginal prob of sibling pair of arcs i<-h->j, i<=j with arcs in dir1, dir2."""
    arc_sample, pos_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=pos_sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_siblings(tensors, sample=sample, allow_root=False)
    # i_le_j_mask = torch.triu(torch.ones_like(arc_sample), diagonal=1)
    # E_counts = contract(
    #     "bhi,bhj,bij,bhx,biy,bjz,bhil,bhjr->xyzlr",
    #     arc_sample,  # first child arc
    #     arc_sample,  # second child arc
    #     i_le_j_mask,  # ensures we only count each pair of edges once
    #     pos_ps,  # head tag
    #     pos_ps,  # left child tag
    #     pos_ps,  # right child tag
    #     dir_mask_adj,  # left child dir
    #     dir_mask_adj,  # right child dir
    # )
    E_counts = contract(
        "bhi,bhj,bhx,biy,bjz,bhil,bhjr->xyzlr",
        arc_sample,  # first child arc
        arc_sample,  # second child arc
        # i_le_j_mask,  # ensures we only count each pair of edges once
        pos_ps,  # head tag
        pos_ps,  # left child tag
        pos_ps,  # right child tag
        dir_mask_adj,  # left child dir
        dir_mask_adj,  # right child dir
    )
    print("T", N_total)
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_tail_gtail_dir_gdir_freq(tensors, sample=True, pos_sample=False, **kwargs):
    """Marginal prob of sibling pair of arcs h->tail->gtail with arcs in dir, gdir."""
    arc_sample, pos_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=pos_sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_grandchild_pairs(tensors, sample=sample)
    E_counts = contract(
        "bht,btg,bhx,bty,bgz,bhtl,btgr->xyzlr",
        arc_sample,  # first child arc
        arc_sample,  # second child arc
        pos_ps,  # head tag
        pos_ps,  # left child tag
        pos_ps,  # right child tag
        dir_mask_adj,  # left child dir
        dir_mask_adj,  # right child dir
    )
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def head_dep1_dep2_dir1_dir2_freq(tensors, sample=True, pos_sample=False, label_sample=False, **kwargs):
    """Marginal prob of sibling pair of arcs <-[dep1]-h-[dep2]-> with arcs in dir1, dir2."""
    arc_sample, pos_ps, label_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=pos_sample),
        get_label_rep(tensors, sample=label_sample),
        get_dir_mask_adj(tensors),
    )
    i_le_j_mask = torch.triu(torch.ones_like(arc_sample), diagonal=1)
    N_total = get_N_total_siblings(tensors, sample=sample, allow_root=False)
    E_counts = contract(
        "bhi,bhj,bij,bhx,bhiy,bhjz,bhil,bhjr->xyzlr",
        arc_sample,  # first child arc
        arc_sample,  # second child arc
        i_le_j_mask,  # ensures we only count each pair of edges once
        pos_ps,  # head tag
        label_ps,  # left child dep
        label_ps,  # right child dep
        dir_mask_adj,  # left child dir
        dir_mask_adj,  # right child dir
    )
    E_freq = E_counts / N_total
    return E_freq


# =============================
# Arc-Pair Grandchild Marginals
# =============================
@cached_stat_func
def head_tail_gtail_dir_gdir_freq(tensors, sample=True, pos_sample=False, **kwargs):
    """Marginal prob of sibling pair of arcs h->tail->gtail with arcs in dir, gdir."""
    arc_sample, pos_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=pos_sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_grandchild_pairs(tensors, sample=sample)
    E_counts = contract(
        "bht,btg,bhx,bty,bgz,bhtl,btgr->xyzlr",
        arc_sample,  # first child arc
        arc_sample,  # second child arc
        pos_ps,  # head tag
        pos_ps,  # left child tag
        pos_ps,  # right child tag
        dir_mask_adj,  # left child dir
        dir_mask_adj,  # right child dir
    )
    E_freq = E_counts / N_total
    return E_freq


@cached_stat_func
def tail_dep_gdep_dir_gdir_freq(tensors, sample=True, pos_sample=False, label_sample=False, indexes=None, **kwargs):
    """Marginal prob of sibling pair of arcs -[dep]->tail-[gdep]-> with arcs in dir, gdir."""
    parent_arc_sample, child_arc_sample, pos_ps, label_ps, dir_mask_adj = (
        get_arc_rep(tensors, sample=sample, allow_root=True),
        get_arc_rep(tensors, sample=sample, allow_root=False),
        get_pos_rep(tensors, sample=pos_sample),
        get_label_rep(tensors, sample=label_sample),
        get_dir_mask_adj(tensors),
    )
    N_total = get_N_total_grandchild_pairs(tensors, sample=sample)
    indexes = [None, None] + list(indexes) if indexes is not None else None
    # print("c", indexes)
    E_counts = contract(
        "bht,btg,btx,bhty,btgz,bhtl,btgr->xyzlr",
        parent_arc_sample,  # first child arc
        child_arc_sample,  # second child arc
        pos_ps,  # tail tag
        label_ps,  # parent label
        label_ps,  # child label
        dir_mask_adj,  # left child dir
        dir_mask_adj,  # right child dir
        indexes=indexes,
    )
    E_freq = E_counts / N_total
    return E_freq


# ====================
# Distributional Stats
# e.g., entropy
# ====================
def pos_entropy(tensors, vocab, *args, sample=False, **kwargs):
    """The marginal entropy of the pos tags (marginalizing over positions)"""
    mask = tensors["mask"]
    pos_ps = get_pos_rep(tensors, sample=sample)
    B, N, T = pos_ps.shape
    marginal_ps = pos_ps.reshape(-1, T).sum() / mask.sum()
    dist = D.OneHotCategorical(probs=marginal_ps)
    return dist.entropy()


def pos_avg_entropy(tensors, vocab, *args, sample=False, **kwargs):
    """The average entropy of pos-tags computed per position."""
    pos_ps, mask = get_pos_rep(tensors, sample=sample), tensors["mask"]
    if "pos_logits" in tensors:
        dist = D.OneHotCategorical(logits=tensors["pos_logits"])
    else:
        # Fill pads with ones instead of zeros so dist args validate (have nonzero sum)
        ps = mask.unsqueeze(2) * tensors["pos_ps"] + (1 - mask.unsqueeze(2))
        dist = D.OneHotCategorical(probs=ps)

    avg_entropy = (dist.entropy() * mask).sum() / mask.sum()
    return avg_entropy


def dep_avg_entropy(tensors, vocab, *args, sample=False, **kwargs):
    """The average entropy of dep labels computed per position, marginalizing over edges."""
    arc_ps, label_ps, mask, mask_adj = (
        get_arc_rep(tensors, sample=sample),
        get_label_rep(tensors, sample=sample),
        tensors["mask"],
        tensors["mask_adj"],
    )
    # Fill pads with ones instead of zeros so dist args validate (have nonzero sum)
    # print("label_ps", label_ps.shape)
    ps = mask_adj.unsqueeze(3) * label_ps + (1 - mask_adj.unsqueeze(3))
    dist = D.OneHotCategorical(probs=ps)
    N_total = get_N_total_edges(tensors)

    # print(f"mask_adj: {mask_adj.shape}, arc_ps: {arc_ps.shape}, ent: {dist.entropy().shape}")
    avg_entropy = (mask_adj * arc_ps * dist.entropy()).sum() / N_total
    return avg_entropy


def arc_avg_entropy(tensors, vocab, *args, sample=False, **kwargs):
    """The average entropy of dep labels computed per position, marginalizing over edges."""
    arc_ps, mask, mask_adj = get_arc_rep(tensors, sample=sample), tensors["mask"], tensors["mask_adj"]
    # Fill pads with ones instead of zeros so dist args validate (have nonzero sum)
    # and switch arcs from head,tail to tail,head so that we get entropy over head choices
    ps = mask_adj * arc_ps + (1 - mask_adj)
    try:
        dist = D.OneHotCategorical(probs=ps.transpose(1, 2))
    except ValueError as e:
        with open("arc_avg_tensors.pkl", "wb") as f:
            torch.save(tensors, f)
        raise e
    N_total = get_N_total_edges(tensors)

    # print(f"mask_adj: {mask_adj.shape}, arc_ps: {arc_ps.shape}, ent: {dist.entropy().shape}")
    avg_entropy = (mask * dist.entropy()).sum() / N_total
    return avg_entropy


def tree_entropy(tensors, vocab, *args, **kwargs):
    """The average entropy of the parse tree (edge) distribution."""
    return tensors["arc_crf"].entropy.mean()


# ====================
# Meta function allows multiprocessing without passing around function objects.
# We can pass around the name and arg tuples instead.
# It also allows us to compute compound stats, such as conditional probs very easily.
#   e.g.  head_tail_freq|head_freq = p(head,tail)/p(head) = p(tail|head)
# ====================
def stat_func_by_name(tensors, vocab, f_name, *f_args, **f_kwargs):
    if len(f_args) == 1 and type(f_args[0]) == str:
        f_args = parse_argstr(f_args[0])
    if "|" in f_name:
        # For computing conditional probs that are joint over given marginal: p(x|y) = p(x,y)/p(y)
        # Assumes the function combinations specified are mathematically correct
        left, right = f_name.split("|")
        left_args, right_args = f_args[0], f_args[1]
        left_kwargs = {k: v[0] for k, v in f_kwargs.items()}
        right_kwargs = {k: v[1] for k, v in f_kwargs.items()}
        top = globals()[left](tensors, vocab, *left_args, **left_kwargs)
        bottom = globals()[right](tensors, vocab, *right_args, **right_kwargs)
        # print("tsf", f_name, top, bottom)
        if (bottom > 0.0).all():
            return top / bottom
        else:
            return torch.tensor(-1.0).to(tensors["arc_ps"].device)

    else:
        # print(f_name, f_args, f_kwargs)
        return globals()[f_name](tensors, vocab, *f_args, **f_kwargs)


# ===============================
# Helper functions
# ===============================
def contract(formula, *args, indexes=None):
    """Perform an efficient tensor contraction.

    If indexes is passed, it is assumed that there is one index per tensor and that it gathers along the last dimension
    of the main tensor. If it is None, we assume a full slice.
    """
    tensors = args
    indexed_tensors = []
    if indexes is not None:
        assert len(tensors) == len(indexes), f"t:{len(tensors)}, i:{len(indexes)}"
        for tensor, index in zip(tensors, indexes):
            if index is not None:
                idxs = [slice(None)] * (len(tensor.shape) - 1) + [index]
                indexed_tensor = tensor[idxs]
            else:
                indexed_tensor = tensor
            indexed_tensors.append(indexed_tensor)
    else:
        indexed_tensors = tensors

    t0 = time()
    result = oe.contract(formula, *indexed_tensors, backend="torch", optimize="optimal")
    # print(f"Contract shape, time: {result.shape}, {time()-t0}")
    # path_info = oe.contract_path(formula, *indexed_tensors, optimize="optimal")
    # print(f"Contract path: {path_info}")
    return result


def get_token_index(token, vocab, namespace="infer"):
    """Convert tokens or lists of tokens to indices."""
    if isinstance(token, (list, tuple)):
        return type(token)([get_token_index(t, vocab, namespace) for t in token])
    if type(token) == str:
        og_token = token
        if namespace == "infer":
            if token in dirs:
                return dirs.index(token)
            elif token in dists:
                return dists.index(token)
            elif token in valencies:
                return valencies.index(token)
            elif token in pos_tags:
                return get_token_index(token, vocab, namespace="upos")
            elif token in coarse_dep_tags_set:
                return get_token_index(token, vocab, namespace="dep")
            else:
                raise ValueError(f"Failed to infer namespace for token: {token}")

        if namespace == "dep":
            token = coarse_dep_tags.index(token)
        else:
            token = vocab.get_token_index(token, namespace)
            if (
                vocab._oov_token in vocab._token_to_index[namespace]
                and token == vocab._token_to_index[namespace][vocab._oov_token]
            ):
                raise ValueError(
                    f"Invalid target token `{og_token}` returns {vocab._oov_token} in namespace {namespace}"
                )
    return token


def get_N_total_edges(tensors):
    if "N_total_edges" not in tensors:
        tensors["N_total_edges"] = tensors["mask"].sum()
    return tensors["N_total_edges"]


def get_Ns(tensors):
    if "Ns" not in tensors:
        tensors["Ns"] = tensors["mask"].sum(dim=1)
    return tensors["Ns"]


def get_N_total_siblings(tensors, **kwargs):
    cache_name = "N_total_siblings"
    if cache_name not in tensors:
        arc_rep = get_arc_rep(tensors, **kwargs)
        # i_le_j_mask = torch.triu(torch.ones_like(arc_rep), diagonal=1)
        # N_total = contract(
        #     "bhi,bhj,bij->",
        #     arc_rep,
        #     arc_rep,
        #     i_le_j_mask,
        # )
        N_total = contract(
            "bhi,bhj->",
            arc_rep,
            arc_rep,
        )
        tensors[cache_name] = N_total
    return tensors[cache_name]


def get_N_total_grandchild_pairs(tensors, **kwargs):
    cache_name = "N_total_grandchild_pairs"
    if cache_name not in tensors:
        parent_arc_rep = get_arc_rep(tensors, allow_root=True, **kwargs)
        child_arc_rep = get_arc_rep(tensors, allow_root=False, **kwargs)
        N_total = contract(
            "bht,btg->",
            parent_arc_rep,
            child_arc_rep,
        )
        tensors[cache_name] = N_total
    return tensors[cache_name]


def get_dir_mask_adj(tensors):
    if "dir_mask" not in tensors:
        mask_adj = tensors["mask_adj"].float()
        left_mask = torch.tril(mask_adj, diagonal=-1)  # lefts must have i < j
        right_mask = torch.triu(
            mask_adj, diagonal=0
        )  # rights may have i = j (root attachments are from front of sent)
        tensors["dir_mask"] = torch.stack([left_mask, right_mask], dim=-1)  # b x n x n x 2
    return tensors["dir_mask"]


def get_non_root_mask(tensors):
    N = tensors["arc_ps"].shape[1]
    return 1 - torch.eye(N, N).to(tensors["arc_ps"].device).unsqueeze(0)


def get_pos_rep(tensors, sample=False):
    if sample:
        if "pos_sample" not in tensors:
            temp = sample if type(sample) is float else 1.0
            pos_sample = D.RelaxedOneHotCategorical(temp, logits=tensors["pos_logits"]).rsample()
            tensors["pos_sample"] = pos_sample
        return tensors["pos_sample"]
    else:
        return tensors["pos_ps"]


def get_arc_rep(tensors, sample=False, allow_root=True):
    if sample:
        if "arc_sample" not in tensors:
            temp = sample if type(sample) is float else 1.0
            tensors["arc_sample"] = tensors["arc_crf"].rsample(temp)
        arc_rep = tensors["arc_sample"]
    else:
        arc_rep = tensors["arc_ps"]
    if not allow_root:
        arc_rep = arc_rep * get_non_root_mask(tensors)
    return arc_rep


def get_label_rep(tensors, sample=False):
    if sample:
        if "label_sample" not in tensors:
            temp = sample if type(sample) is float else 1.0
            label_sample = D.RelaxedOneHotCategorical(temp, logits=tensors["label_logits"]).rsample(temp)
            tensors["label_sample"] = label_sample
        return tensors["label_sample"]
    else:
        return tensors["label_ps"]


def get_dist_mask(tensors):
    if "dist_mask" not in tensors:
        ranges = {
            "root-dist": [-1, 0.5],
            "adjacent": [0.5, 1.5],
            "close": [1.5, 3.5],
            "near": [3.5, 7.5],
            "far": [7.5, 1e10],
        }
        B, N = tensors["arc_ps"].shape[:2]
        dist_masks = []
        r = torch.arange(N).to(tensors["arc_ps"].device)
        ds = (r.reshape(1, N, 1) - r.reshape(1, 1, N)).abs()
        for dist in dists:
            a, b = ranges[dist]
            is_dist = ((a < ds) * (ds <= b)).float()  # 1, N, N
            dist_masks.append(is_dist)
        tensors["dist_mask"] = torch.stack(dist_masks, dim=-1).repeat(B, 1, 1, 1)  # B, N, N, D
    return tensors["dist_mask"]


def get_predictor(archive, **kwargs):
    # Avoid circlular input
    from ml.udify.predictors.predictor import UdifyPredictor
    from ml.models.expected_syntax_udify_model import ExpectedSyntaxUdifyModel

    predictor = UdifyPredictor.from_path(archive, predictor_name="udify_predictor", **kwargs)
    return predictor


def tensorize(data, vocab):
    """Convert discrete dictionary format of a sentence parses into indicator tensors like those from the model."""
    B = len(data)
    n = max(len(d["words"]) for d in data) - 1
    t2i = vocab.get_token_to_index_vocabulary("upos")
    l2i = vocab.get_token_to_index_vocabulary("head_tags")

    pos_ps = torch.zeros(B, n, len(t2i))
    arc_ps = torch.zeros(B, n + 1, n + 1)
    label_ps = torch.zeros(B, len(l2i), n + 1, n + 1)
    mask = torch.zeros(B, n + 1)
    for b, datum in enumerate(data):
        for i, w in enumerate(datum["words"][1:]):  # skip root
            j = t2i[w["tag"]]
            pos_ps[b, i, j] = 1

        for arc in datum["arcs"]:
            i, j = arc["head"], arc["tail"]
            arc_ps[b, i, j] = 1
            k = l2i[arc["label"]]
            label_ps[b, k, i, j] = 1

        mask[b, : len(datum["words"])] = 1

    return pos_ps, arc_ps, label_ps, mask


def tensorize_and_format(data, predictor):
    """Tensorize a raw datum and convert it to the format used by the tensor risk functions."""
    pos_ps, arc_ps, label_ps, mask = tensorize(data, predictor._model.vocab)
    to_logits = lambda ps: ps.log().clamp(-1e8)
    (pos_ps, label_ps, arc_ps, arc_logits, arc_crf, mask_adj,) = predictor._model.parse_logits_to_probs(
        to_logits(pos_ps),
        to_logits(label_ps),
        to_logits(arc_ps),
        mask,
        safe_arc_ps=False,
    )
    tensors = dict(
        pos_ps=pos_ps,
        arc_ps=arc_ps,
        label_ps=label_ps,
        mask=mask[:, 1:],
        mask_adj=mask_adj,
        arc_logits=arc_logits,
        arc_crf=arc_crf,
    )
    return tensors


def tensorize_direct(data, predictor, requires_grad=False):
    """Directly convert discrete dictionary format of sentence parses to indicator tensors used by risk functions.

    This circumvents numerical issues where the marginal arc probabilities via inverse laplacian don't exactly sum to N.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocab = predictor._model.vocab
    B = len(data)
    n = max(len(d["words"]) for d in data) - 1
    t2i = vocab.get_token_to_index_vocabulary("upos")
    l2i = {
        l: i for i, l in enumerate(predictor._model.coarse_dep_tags)
    }  # vocab.get_token_to_index_vocabulary("head_tags")

    pos_ps = torch.zeros(B, n, len(t2i), requires_grad=requires_grad).to(device)
    arc_ps = torch.zeros(B, n, n, requires_grad=requires_grad).to(device)
    label_ps = torch.zeros(B, n, n, len(l2i), requires_grad=requires_grad).to(device)
    mask = torch.zeros(B, n, requires_grad=requires_grad).to(device)
    for b, datum in enumerate(data):
        for i, w in enumerate(datum["words"][1:]):  # skip root
            j = t2i[w["tag"]]
            pos_ps[b, i, j] = 1

        for arc in datum["arcs"]:
            i, j = arc["head"] - 1, arc["tail"] - 1
            assert j >= 0
            if i == -1:
                i = j  # root attachements go to diagonals
            arc_ps[b, i, j] = 1
            k = l2i[clean_dep(arc["label"])]
            label_ps[b, i, j, k] = 1

        mask[b, : len(datum["words"]) - 1] = 1

    mask_adj = mask.unsqueeze(1) * mask.unsqueeze(2)

    tensors = dict(
        pos_ps=pos_ps,
        arc_ps=arc_ps,
        label_ps=label_ps,
        mask=mask,
        mask_adj=mask_adj,
    )
    return tensors
