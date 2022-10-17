""" Generic functions for computing statistics on a dataset. 

Each function takes the data from a UDPDataset as first arg with addition optional args and returns a float.
"""
import json
import math
import itertools
from collections import Counter
from scipy.stats import entropy
from tqdm import tqdm
from time import time

# ====================
# Distributional Stats
# e.g., entropy
# ====================
def pos_entropy(data, *args, **kwargs):
    possible_tags = {w["tag"] for d in data for w in d["words"][1:]}
    ps = [pos_freq(data, tag) for tag in possible_tags]
    return float(entropy(ps))


def _mu(raw_generator, target_arg, counts=None, N=None, return_counts=False, verbose=False, smoothing_total=0):
    """Abstract wrapper for computing generic count-based marginal distributions.

    Parameters:
      raw_generator (generator): an iterator over the raw statistics derived from the data
      target_arg (str or tuple(str)): the query element to get the marginal for
      counts (Counter): precomputed stats for caching
      N (int): = sum(counts.values()), allows for caching
      return_counts (bool): return `counts` and `N` for caching
    """
    if counts is None:
        counts = Counter(raw_generator)
    if N is None:
        N = sum(counts.values())
    arg_count = counts[target_arg]
    count = N
    if smoothing_total:
        arg_count += 1
        count += smoothing_total
    s = arg_count / count
    if verbose:
        print(f"  {target_arg}:  {s:2.6f} =  {arg_count} / {count}")
    if return_counts:
        return s, counts, N
    else:
        return s


def _per_datum_mu(gen_func, data, arg, return_counts=False, **kwargs):
    all_stats, all_counts, all_Ns = [], [], []
    # print('counts', len(kwargs.get('counts', []) or []))
    # if kwargs.get('counts', None) is not None:
    #     print(kwargs['counts'][0])
    times = Counter()
    # t0 = time()
    d_kwargs = {k: v for k, v in kwargs.items() if k not in ("counts", "N")}
    for i, d in enumerate(data):  # tqdm(enumerate(data), "per datum", len(data)):
        # if (i % 1000) == 0:
        #     print(i, end=',', flush=True)
        if return_counts:
            # t = time()
            if kwargs.get("counts", None) is not None:
                # times['dkwarg'] += time()-t
                # t = time()
                s, counts, N = _mu(
                    None, arg, return_counts=return_counts, counts=kwargs["counts"][i], N=kwargs["N"][i], **d_kwargs
                )
                # times['mu'] += time()-t
            else:
                raw_generator = gen_func(d)
                s, counts, N = _mu(raw_generator, arg, return_counts=return_counts, **kwargs)

            all_stats.append(s)
            all_counts.append(counts)
            all_Ns.append(N)
        else:
            s = _mu(raw_generator, arg, return_counts=return_counts, **kwargs)
            all_stats.append(s)
    # t = time()
    s = sum(all_stats) / len(data)
    # print(f'avg time: {time()-t}')
    # print(f'total time: {time()-t0}')
    # print(times.most_common())

    if return_counts:
        return s, all_counts, all_Ns
    else:
        return s


# =========================
# Individual marginal probs
# =========================
def pos_freq(data, tag, **kwargs):
    """Marginal prob of POS tag in any position over all sentences."""
    raw_generator = (w["tag"] for d in data for w in d["words"][1:])
    return _mu(raw_generator, tag, **kwargs)


def pos_sent_freq(data, tag, **kwargs):
    gen_func = lambda d: (w["tag"] for w in d["words"][1:])
    return _per_datum_mu(gen_func, data, tag, **kwargs)


def pos_gold_pred_freq(data, tag, other_tag, **kwargs):
    raw_generator = (
        (w["tag"], w_pred["tag"]) for d, d_pred in data for w, w_pred in zip(d["words"][1:], d_pred["words"][1:])
    )
    return _mu(raw_generator, (tag, other_tag), **kwargs)


def pos_pred_freq(data, other_tag, **kwargs):
    raw_generator = (w_pred["tag"] for _, d_pred in data for w_pred in d_pred["words"][1:])
    return _mu(raw_generator, other_tag, **kwargs)


def dep_freq(data, dep, **kwargs):
    """Marginal prob of any dependency label in any edge over all sentences."""
    raw_generator = (clean_dep(a["label"]) for d in data for a in d["arcs"])
    return _mu(raw_generator, dep, **kwargs)


def dep_sent_freq(data, dep, **kwargs):
    gen_func = lambda d: (clean_dep(a["label"]) for a in d["arcs"])
    return _per_datum_mu(gen_func, data, dep, **kwargs)


# =====================================
# Undirected combination marginal probs
# =====================================
def pos_pair_freq(data, left_tag, right_tag, **kwargs):
    """Joint prob of seeing `left_tag` at position i and `right_tag` at i+1."""
    raw_generator = (
        (d["words"][i]["tag"], d["words"][i + 1]["tag"]) for d in data for i in range(len(d["words"]) - 1)
    )
    return _mu(raw_generator, (left_tag, right_tag), **kwargs)


def head_freq(data, head_tag, **kwargs):
    """Marginal prob of `head_tag` being the head of an edge over all sentences."""
    raw_generator = (d["words"][a["head"]]["tag"] for d in data for a in d["arcs"])
    return _mu(raw_generator, head_tag, **kwargs)


def head_sent_freq(data, head_tag, **kwargs):
    gen_func = lambda d: (d["words"][a["head"]]["tag"] for a in d["arcs"])
    return _per_datum_mu(gen_func, data, head_tag, **kwargs)


def tail_freq(data, tail_tag, **kwargs):
    """Marginal prob of `tail_tag` being the tail of an edge over all sentences."""
    raw_generator = (d["words"][a["tail"]]["tag"] for d in data for a in d["arcs"])
    return _mu(raw_generator, tail_tag, **kwargs)


def tail_sent_freq(data, tail_tag, **kwargs):
    gen_func = lambda d: (d["words"][a["tail"]]["tag"] for a in d["arcs"])
    return _per_datum_mu(gen_func, data, tail_tag, **kwargs)


def head_dep_freq(data, head_tag, dep_tag, **kwargs):
    """Marginal prob of (head,dep) pairs in any edge over all sentences."""
    raw_generator = ((d["words"][a["head"]]["tag"], clean_dep(a["label"])) for d in data for a in d["arcs"])
    return _mu(raw_generator, (head_tag, dep_tag), **kwargs)


def tail_dep_freq(data, tail_tag, dep_tag, **kwargs):
    """Marginal prob of (tail,dep) pairs in any edge over all sentences."""
    raw_generator = ((d["words"][a["tail"]]["tag"], clean_dep(a["label"])) for d in data for a in d["arcs"])
    return _mu(raw_generator, (tail_tag, dep_tag), **kwargs)


def head_tail_freq(data, head_tag, tail_tag, **kwargs):
    """Marginal prob of (head,tail) POS pairs in any edge over all sentences."""
    raw_generator = ((d["words"][a["head"]]["tag"], d["words"][a["tail"]]["tag"]) for d in data for a in d["arcs"])
    return _mu(raw_generator, (head_tag, tail_tag), **kwargs)


def head_tail_dep_freq(data, head_tag, tail_tag, dep_tag, **kwargs):
    """Marginal prob of (head,tail,dep) tuples in any edge over all sentences."""
    raw_generator = (
        (d["words"][a["head"]]["tag"], d["words"][a["tail"]]["tag"], clean_dep(a["label"]))
        for d in data
        for a in d["arcs"]
    )
    return _mu(raw_generator, (head_tag, tail_tag, dep_tag), **kwargs)


# ===================================
# Directed combination marginal probs
# ===================================
def dir_freq(data, dir, **kwargs):
    """Marginal prob of having an edge in direction dir over all sentences."""
    raw_generator = (a["dir"] for d in data for a in d["arcs"])
    return _mu(raw_generator, dir, **kwargs)


def head_dir_freq(data, head_tag, dir, **kwargs):
    """Marginal prob of head tag having an edge in direction dir over all sentences."""
    raw_generator = ((d["words"][a["head"]]["tag"], a["dir"]) for d in data for a in d["arcs"])
    return _mu(raw_generator, (head_tag, dir), **kwargs)


def tail_dir_freq(data, tail_tag, dir, **kwargs):
    """Marginal prob of tail tag having an edge in direction dir over all sentences."""
    raw_generator = ((d["words"][a["tail"]]["tag"], a["dir"]) for d in data for a in d["arcs"])
    return _mu(raw_generator, (tail_tag, dir), **kwargs)


def dep_dir_freq(data, dep_tag, dir, **kwargs):
    """Marginal prob of dependency in direction dir over all sentences."""
    raw_generator = ((clean_dep(a["label"]), a["dir"]) for d in data for a in d["arcs"])
    return _mu(raw_generator, (dep_tag, dir), **kwargs)


def tail_dep_dir_freq(data, tail_tag, dep_tag, dir, **kwargs):
    """Marginal prob of (tail,dep) pairs in direction dir over all sentences."""
    raw_generator = ((d["words"][a["tail"]]["tag"], clean_dep(a["label"]), a["dir"]) for d in data for a in d["arcs"])
    return _mu(raw_generator, (tail_tag, dep_tag, dir), **kwargs)


def head_dep_dir_freq(data, head_tag, dep_tag, dir, **kwargs):
    """Marginal prob of (head,dep) pairs in direction dir over all sentences."""
    raw_generator = ((d["words"][a["head"]]["tag"], clean_dep(a["label"]), a["dir"]) for d in data for a in d["arcs"])
    return _mu(raw_generator, (head_tag, dep_tag, dir), **kwargs)


def head_tail_dir_freq(data, head_tag, tail_tag, dir, **kwargs):
    """Marginal prob of (head,tail) POS pairs in direction dir over all sentences."""
    raw_generator = (
        (d["words"][a["head"]]["tag"], d["words"][a["tail"]]["tag"], a["dir"]) for d in data for a in d["arcs"]
    )
    return _mu(raw_generator, (head_tag, tail_tag, dir), **kwargs)


def head_tail_dep_dir_freq(data, head_tag, tail_tag, dep_tag, dir, **kwargs):
    """Marginal prob of (head,tail,dep) tuples in direction dir over all sentences."""
    raw_generator = (
        (
            d["words"][a["head"]]["tag"],
            d["words"][a["tail"]]["tag"],
            clean_dep(a["label"]),
            a["dir"],
        )
        for d in data
        for a in d["arcs"]
    )
    return _mu(raw_generator, (head_tag, tail_tag, dep_tag, dir), **kwargs)


def head_tail_dep_dir_sent_freq(data, head_tag, tail_tag, dep_tag, dir, **kwargs):
    gen_func = lambda d: (
        (
            d["words"][a["head"]]["tag"],
            d["words"][a["tail"]]["tag"],
            clean_dep(a["label"]),
            a["dir"],
        )
        for a in d["arcs"]
    )
    return _per_datum_mu(gen_func, data, (head_tag, tail_tag, dep_tag, dir), **kwargs)


def head_tail_dep_dir_gold_pred_freq(
    data, head_tag, tail_tag, dep_tag, dir, pred_head_tag, pred_tail_tag, pred_dep_tag, pred_dir, **kwargs
):
    """Marginal prob of (head,tail,dep) tuples in direction dir over all sentences."""
    raw_generator = (
        (
            d["words"][a["head"]]["tag"],
            d["words"][a["tail"]]["tag"],
            clean_dep(a["label"]),
            a["dir"],
            d["words"][b["head"]]["tag"],
            d["words"][b["tail"]]["tag"],
            clean_dep(b["label"]),
            b["dir"],
        )
        for d, d_pred in data
        for a, b in zip(d["arcs"], d_pred["arcs"])
    )
    return _mu(
        raw_generator,
        (head_tag, tail_tag, dep_tag, dir, pred_head_tag, pred_tail_tag, pred_dep_tag, pred_dir),
        **kwargs,
    )


def head_tail_dep_dir_pred_freq(data, head_tag, tail_tag, dep_tag, dir, **kwargs):
    """Marginal prob of (head,tail,dep) tuples in direction dir over all sentences."""
    raw_generator = (
        (
            d_pred["words"][b["head"]]["tag"],
            d_pred["words"][b["tail"]]["tag"],
            clean_dep(b["label"]),
            b["dir"],
        )
        for _, d_pred in data
        for b in d_pred["arcs"]
    )
    return _mu(
        raw_generator,
        (head_tag, tail_tag, dep_tag, dir),
        **kwargs,
    )


# =======================
# Distance marginal probs
# =======================
def bin_distance(dist):
    if dist <= 0.5:
        return "root-dist"
    elif 0.5 < dist <= 1.5:
        return "adjacent"
    elif 1.5 < dist <= 3.5:
        return "close"
    elif 3.5 < dist <= 7.5:
        return "near"
    else:
        return "far"


arc_dist = lambda a: bin_distance(abs(a["head"] - a["tail"]) if a["head"] != 0 else 0)


def dist_freq(data, target_dist, **kwargs):
    """Marginal prob of edge distance categories."""
    # This data models ROOT attachment as an arc to 0, but we we define it as a self-attachment
    raw_generator = (arc_dist(a) for d in data for a in d["arcs"])
    return _mu(raw_generator, target_dist, **kwargs)


def dep_dist_freq(data, target_dep, target_dist, **kwargs):
    """Marginal prob of dependency with dist."""
    # This data models ROOT attachment as an arc to 0, but we we define it as a self-attachment
    raw_generator = ((a["label"], arc_dist(a)) for d in data for a in d["arcs"])
    return _mu(raw_generator, (target_dep, target_dist), **kwargs)


def dir_dist_freq(data, target_dir, target_dist, **kwargs):
    """Marginal prob of an arc going in direction with dist."""
    # This data models ROOT attachment as an arc to 0, but we we define it as a self-attachment
    raw_generator = ((a["dir"], arc_dist(a)) for d in data for a in d["arcs"])
    return _mu(raw_generator, (target_dir, target_dist), **kwargs)


def dep_dir_dist_freq(data, target_dep, target_dir, target_dist, **kwargs):
    """Marginal prob of dependency going in direction with dist."""
    # This data models ROOT attachment as an arc to 0, but we we define it as a self-attachment
    raw_generator = ((a["label"], a["dir"], arc_dist(a)) for d in data for a in d["arcs"])
    return _mu(raw_generator, (target_dep, target_dir, target_dist), **kwargs)


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


def valency_freq(data, target_bin, **kwargs):
    """Marginal prob that words with tag `head_tag` heads `target_bin` edges for any word over all sentences."""
    # Skip root since it should be 1 by definition
    raw_generator = (
        bin_valency(sum(a["head"] == i for a in d["arcs"])) for d in data for i, w in enumerate(d["words"]) if i > 0
    )
    return _mu(raw_generator, target_bin, **kwargs)


def pos_valency_freq(data, head_tag, target_bin, **kwargs):
    """Marginal prob that words with tag `head_tag` heads `target_bin` edges for any word over all sentences."""
    # Skip root since it should be 1 by definition
    raw_generator = (
        (w["tag"], bin_valency(sum(a["head"] == i for a in d["arcs"])))
        for d in data
        for i, w in enumerate(d["words"])
        if i > 0
    )
    return _mu(raw_generator, (head_tag, target_bin), **kwargs)


# ==========================
# Arc-Pair Sibling Marginals
# ==========================
def head_tail1_tail2_dir1_dir2_freq(data, head, tail1, tail2, dir1, dir2, **kwargs):
    """Marginal prob of sibling pair of arcs c1<-h->c2 with arcs in dir1, dir2."""
    raw_generator = (
        (
            d["words"][a1["head"]]["tag"],
            d["words"][a1["tail"]]["tag"],
            d["words"][a2["tail"]]["tag"],
            a1["dir"],
            a2["dir"],
        )
        for d in data
        for a1 in d["arcs"]
        for a2 in d["arcs"]
        if a1["head"] == a2["head"] and a1["head"] != 0
    )
    return _mu(raw_generator, (head, tail1, tail2, dir1, dir2), **kwargs)


def head_dep1_dep2_dir1_dir2_freq(data, head, dep1, dep2, dir1, dir2, **kwargs):
    """Marginal prob of sibling pair of arcs <-[dep1]-h-[dep2]-> with arcs in dir1, dir2."""
    raw_generator = (
        (
            d["words"][a1["head"]]["tag"],
            a1["label"],
            a2["label"],
            a1["dir"],
            a2["dir"],
        )
        for d in data
        for a1 in d["arcs"]
        for a2 in d["arcs"]
        if a1["head"] == a2["head"] and a1["tail"] < a2["tail"]
    )
    return _mu(raw_generator, (head, dep1, dep2, dir1, dir2), **kwargs)


# =============================
# Arc-Pair Grandchild Marginals
# =============================
def head_tail_gtail_dir_gdir_freq(data, head, tail, gtail, dir, gdir, **kwargs):
    """Marginal prob of sibling pair of arcs h->tail->gtail with arcs in dir, gdir."""
    raw_generator = (
        (
            d["words"][a1["head"]]["tag"],
            d["words"][a1["tail"]]["tag"],
            d["words"][a2["tail"]]["tag"],
            a1["dir"],
            a2["dir"],
        )
        for d in data
        for a1 in d["arcs"]
        for a2 in d["arcs"]
        if a1["tail"] == a2["head"]
    )
    return _mu(raw_generator, (head, tail, gtail, dir, gdir), **kwargs)


def tail_dep_gdep_dir_gdir_freq(data, tail, hlabel, glabel, hdir, gdir, **kwargs):
    """Marginal prob of sibling pair of arcs -[hlabel]>tail-[glabel]-> with arcs in hdir, gdir."""
    raw_generator = (
        (
            d["words"][a1["tail"]]["tag"],
            a1["label"],
            a2["label"],
            a1["dir"],
            a2["dir"],
        )
        for d in data
        for a1 in d["arcs"]
        for a2 in d["arcs"]
        if a1["tail"] == a2["head"]
    )
    return _mu(raw_generator, (tail, hlabel, glabel, hdir, gdir), **kwargs)


# ====================
# Meta function allows multiprocessing without passing around function objects.
# We can pass around the name and arg tuples instead.
# It also allows us to compute compound stats, such as conditional probs very easily.
# ====================
def parse_argstr(argstr):
    if "|" in argstr:
        left_args, right_args = argstr.split("|")
        f_args = (tuple(left_args.split(",")), tuple(right_args.split(",")))
    else:
        f_args = tuple(argstr.split(","))
    return f_args


def stat_func_by_name(data, f_name, *f_args, smoothed=False, **f_kwargs):
    if len(f_args) == 1 and type(f_args[0]) == str:
        f_args = parse_argstr(f_args[0])
    if "|" in f_name:
        # For computing conditional probs that are joint over given marginal: p(x|y) = p(x,y)/p(y)
        # Assumes the function combinations specified are mathematically correct
        left, right = f_name.split("|")
        left_args, right_args = f_args[0], f_args[1]
        left_kwargs = {k: v[0] for k, v in f_kwargs.items()}
        right_kwargs = {k: v[1] for k, v in f_kwargs.items()}
        if smoothed:
            left_kwargs["smoothing_total"] = right_kwargs["smoothing_total"] = len(stat_func_arglists[f_name])
        top = globals()[left](data, *left_args, **left_kwargs)
        bottom = globals()[right](data, *right_args, **right_kwargs)
        # print("sf", f_name, top, bottom)
        if left_kwargs.get("return_counts", False):
            t, t_counts, t_N = top
            b, b_counts, b_N = bottom
            v = t / b if b > 0.0 else -1.0
            return v, (t_counts, b_counts), (t_N, b_N)
        else:
            return top / bottom if bottom > 0.0 else -1.0

    else:
        if smoothed:
            f_kwargs["smoothing_total"] = len(stat_func_arglists[f_name])
        return globals()[f_name](data, *f_args, **f_kwargs)


# =================================
# Definitions of atomic parse units
# =================================
pos_tags = {
    "PRON",
    "NUM",
    "NOUN",
    "PART",
    "AUX",
    "CCONJ",
    "SCONJ",
    "ADJ",
    "X",
    "INTJ",
    "SYM",
    "PROPN",
    "ADV",
    "PUNCT",
    "ADP",
    "VERB",
    "DET",
}
dep_fine_tags = {
    "nmod:comp",
    "acl:appos",
    "obl:agent",
    "xcomp:ds",
    "discourse:emo",
    "aux:mood",
    "ccomp:pmod",
    "mark:comp",
    "nummod:entity",
    "obj",
    "nsubj",
    "compound:vv",
    "obj:periph",
    "compound:conjv",
    "compound",
    "obj:cau",
    "cop:locat",
    "fixed",
    "det:def",
    "compound:affix",
    "compound:a",
    "aux:q",
    "discourse:filler",
    "reparandum",
    "flat",
    "iobj",
    "acl:poss",
    "mark:rel",
    "obl:mod",
    "acl:part",
    "advcl:cond",
    "parataxis:parenth",
    "advcl:coverb",
    "parataxis:hashtag",
    "parataxis:nsubj",
    "compound:v",
    "expl:poss",
    "nmod:obllvc",
    "parataxis:dislocated",
    "cc:preconj",
    "advcl:appos",
    "compound:vo",
    "nummod:gov",
    "nmod:part",
    "nsubj:periph",
    "iobj:caus",
    "csubj:cleft",
    "nsubj:advmod",
    "csubj",
    "nsubj:nc",
    "obl:comp",
    "aux",
    "advmod:obl",
    "advcl:svc",
    "goeswith",
    "dep:obj",
    "flat:sibl",
    "advmod:to",
    "expl:pv",
    "mark",
    "advcl:cleft",
    "nmod:npmod",
    "acl:adv",
    "root",
    "ccomp:pred",
    "compound:lvc",
    "parataxis:deletion",
    "advmod:emph",
    "parataxis",
    "conj:svc",
    "case:dec",
    "case:det",
    "obj:obl",
    "cop:own",
    "flat:foreign",
    "obl:npmod",
    "advmod:appos",
    "obl:prep",
    "advmod:mode",
    "dep:iobj",
    "mark:prt",
    "case:pref",
    "list",
    "compound:dir",
    "conj:dicto",
    "ccomp:obj",
    "compound:coll",
    "parataxis:newsent",
    "acl:focus",
    "flat:name",
    "nsubj:obj",
    "obl:cau",
    "aux:pass",
    "advcl",
    "case:aspect",
    "case:acc",
    "nmod:dat",
    "csubj:pass",
    "case:circ",
    "nummod",
    "obj:lvc",
    "ccomp",
    "xcomp:adj",
    "expl",
    "nsubj:pass",
    "det:predet",
    "advmod:periph",
    "flat:repeat",
    "parataxis:obj",
    "advcl:sp",
    "obj:advmod",
    "obl",
    "advmod:discourse",
    "compound:prt",
    "dep:prt",
    "nsubj:own",
    "nmod:own",
    "obj:advneg",
    "case:suff",
    "punct",
    "acl:cleft",
    "nmod:appos",
    "det:numgov",
    "acl:relcl",
    "advmod",
    "aux:caus",
    "nsubj:caus",
    "xcomp",
    "amod:attlvc",
    "nsubj:quasi",
    "nmod:gobj",
    "nmod:ins",
    "det:rel",
    "vocative:mention",
    "advmod:tmod",
    "obl:loc",
    "nmod:gmod",
    "det:nummod",
    "compound:redup",
    "conj",
    "compound:svc",
    "amod:att",
    "aux:part",
    "compound:nn",
    "advmod:locy",
    "conj:extend",
    "nmod:pmod",
    "advcl:arg",
    "advmod:neg",
    "advmod:df",
    "nmod:ref",
    "expl:pass",
    "nmod:obl",
    "fixed:name",
    "nmod:advmod",
    "orphan",
    "flat:abs",
    "acl:inf",
    "nmod:cmp",
    "case:voc",
    "det",
    "aux:aglt",
    "parataxis:conj",
    "xcomp:obj",
    "mark:adv",
    "ccomp:cleft",
    "amod:advmod",
    "csubj:cop",
    "amod",
    "nmod",
    "cc",
    "obl:patient",
    "expl:impers",
    "dep",
    "advmod:tlocy",
    "nmod:abl",
    "advmod:det",
    "nsubj:expl",
    "compound:smixut",
    "compound:plur",
    "appos:conj",
    "conj:appos",
    "nmod:agent",
    "mark:q",
    "obj:agent",
    "parataxis:restart",
    "nmod:attlvc",
    "nmod:poss",
    "appos",
    "case:loc",
    "obl:poss",
    "nmod:clas",
    "parataxis:discourse",
    "vocative",
    "nmod:arg",
    "advmod:tto",
    "appos:nmod",
    "compound:quant",
    "mark:advmod",
    "compound:ext",
    "xcomp:sp",
    "iobj:agent",
    "nsubj:lvc",
    "advmod:tfrom",
    "cop",
    "cc:nc",
    "advmod:que",
    "obl:arg",
    "nmod:gen",
    "flat:range",
    "amod:mode",
    "acl",
    "advcl:tcl",
    "parataxis:insert",
    "mark:obj",
    "mark:relcl",
    "aux:neg",
    "obl:advmod",
    "compound:n",
    "compound:preverb",
    "ccomp:obl",
    "mark:obl",
    "nmod:gsubj",
    "nmod:att",
    "obl:x",
    "mark:advb",
    "det:poss",
    "case:gen",
    "goeswith:alt",
    "advcl:periph",
    "vocative:cl",
    "nmod:cau",
    "obl:periph",
    "xcomp:pred",
    "nsubj:appos",
    "discourse",
    "dislocated",
    "conj:coord",
    "nsubj:cop",
    "csubj:quasi",
    "parataxis:rel",
    "case",
    "advmod:cc",
    "obl:own",
    "nmod:tmod",
    "obl:tmod",
    "amod:obl",
    "flat:title",
    "discourse:sp",
    "clf",
    "parataxis:appos",
}
clean_dep = lambda tag: tag.split(":")[0]
coarse_dep_tags_set = {clean_dep(tag) for tag in dep_fine_tags}
coarse_dep_tags = sorted(coarse_dep_tags_set)

dirs = ["left", "right"]
dists = ["root-dist", "adjacent", "close", "near", "far"]
valencies = ["zero", "one", "two", "three", "many"]
pos_tags_list = list(pos_tags)
T, D, L = pos_tags_list, coarse_dep_tags, dirs
pos_pairs = [f"{a},{b}" for a in T for b in T]
tag_dep_pairs = [f"{t},{d}" for t in T for d in D]
tag_tag_dep_triples = [f"{h},{t},{d}" for t in T for h in T for d in D]
tag_dir_pairs = [f"{t},{lr}" for t in T for lr in L]
dep_dir_pairs = [f"{d},{lr}" for d in D for lr in L]
tag_tag_dir_triples = [f"{t},{h},{lr}" for t in T for h in T for lr in L]
tag_dep_dir_triples = [f"{t},{d},{lr}" for t in T for d in D for lr in L]
tag_tag_dep_dir_tuples = [f"{h},{t},{d},{lr}" for h in T for t in T for d in D for lr in L]
dep_dists = [f"{d},{dist}" for d in D for dist in dists]
dir_dists = [f"{lr},{dist}" for lr in L for dist in dists]
dep_dir_dists = [f"{d},{lr},{dist}" for d in D for lr in L for dist in dists]
tag_valencies = [f"{t},{v}" for t in T for v in valencies]
tag_tag_tag_dir_dirs = [f"{h},{c},{g},{lr1},{lr2}" for h in T for c in T for g in T for lr1 in L for lr2 in L]
tag_dep_dep_dir_dirs = [f"{h},{d1},{d2},{lr1},{lr2}" for h in T for d1 in D for d2 in D for lr1 in L for lr2 in L]
tag_tag_tag_dir_dirs_ordered = [
    f"{h},{c},{g},{lr1},{lr2}"
    for h in T
    for c in T
    for g in T
    for lr1 in L
    for lr2 in L
    if (lr1, lr2) != ("right", "left")
]
tag_dep_dep_dir_dirs_ordered = [
    f"{h},{d1},{d2},{lr1},{lr2}"
    for h in T
    for d1 in D
    for d2 in D
    for lr1 in L
    for lr2 in L
    if (lr1, lr2) != ("right", "left")
]

stat_func_arglists = {
    # Entropies
    "pos_avg_entropy": [""],
    "dep_avg_entropy": [""],
    "arc_avg_entropy": [""],
    "tree_entropy": [""],
    "pos_entropy": [""],
    # Undirected marginal probs
    "pos_freq": T,
    "dep_freq": D,
    "pos_pair_freq": pos_pairs,
    "pos_gold_pred_freq": pos_pairs,
    "pos_pred_freq": T,
    "head_freq": T,
    "tail_freq": T,
    "head_dep_freq": tag_dep_pairs,
    "tail_dep_freq": tag_dep_pairs,
    "head_tail_freq": pos_pairs,
    "head_tail_dep_freq": tag_tag_dep_triples,
    # Directed marginal probs
    "dir_freq": dirs,
    "head_dir_freq": tag_dir_pairs,
    "tail_dir_freq": tag_dir_pairs,
    "dep_dir_freq": dep_dir_pairs,
    "head_tail_dir_freq": tag_tag_dir_triples,
    "head_dep_dir_freq": tag_dep_dir_triples,
    "tail_dep_dir_freq": tag_dep_dir_triples,
    "head_tail_dep_dir_freq": tag_tag_dep_dir_tuples,
    "head_tail_dep_dir_pred_freq": tag_tag_dep_dir_tuples,
    # "head_tail_dep_dir_gold_pred_freq": [f"{a},{b}" for a in tag_tag_dep_dir_tuples for b in tag_tag_dep_dir_tuples],
    # Distance
    "dist_freq": dists,
    "dep_dist_freq": dep_dists,
    "dir_dist_freq": dir_dists,
    "dep_dir_dist_freq": dep_dir_dists,
    # Valencies
    "valency_freq": valencies,
    "pos_valency_freq": tag_valencies,
    # Two-arc siblings
    "head_tail1_tail2_dir1_dir2_freq": tag_tag_tag_dir_dirs_ordered,
    "head_dep1_dep2_dir1_dir2_freq": tag_dep_dep_dir_dirs_ordered,
    # Two-arc grandchildren
    "head_tail_gtail_dir_gdir_freq": tag_tag_tag_dir_dirs,
    "tail_dep_gdep_dir_gdir_freq": tag_dep_dep_dir_dirs,
}

# Add all the per_sent ones
for func in ["pos_freq", "dep_freq", "head_freq", "tail_freq", "head_tail_dep_dir_freq"]:
    args = stat_func_arglists[func]
    if "sent" not in func and func.endswith("_freq"):
        stat_func_arglists[func.replace("_freq", "_sent_freq")] = args

# Add various conditionals probs
og_arglists = list(stat_func_arglists.items())
stat_func_arglists["pos_pair_freq|pos_freq"] = [f"{a},{b}|{a}" for a in T for b in T]
stat_func_arglists["pos_pair_sent_freq|pos_sent_freq"] = [f"{a},{b}|{a}" for a in T for b in T]
stat_func_arglists["pos_gold_pred_freq|pos_pred_freq"] = [f"{a},{b}|{b}" for a in T for b in T]
# stat_func_arglists["head_tail_dep_dir_gold_pred_freq|head_tail_dep_dir_pred_freq"] = [
#     f"{a},{b}|{b}" for a in tag_tag_dep_dir_tuples for b in tag_tag_dep_dir_tuples
# ]
# print("set")
skip = {
    "pos_pair_freq",
    "pos_pair_sent_freq",
    "pos_pred_freq",
    "pos_gold_pred_freq",
    "head_tail1_tail2_dir1_dir2_freq",
    "head_tail1_tail2_dir1_dir2_sent_freq",
    "head_dep1_dep2_dir1_dir2_freq",
    "head_dep1_dep2_dir1_dir2_sent_freq",
    "head_tail_gtail_dir_gdir_freq",
    "head_tail_gtail_dir_gdir_sent_freq",
    "tail_dep_gdep_dir_gdir_freq",
    "tail_dep_gdep_dir_gdir_sent_freq",
}
for func, args in og_arglists:
    if func.endswith("_freq") and not (func in skip or "pred" in func):
        # print(func)
        split = func.split("_")
        if "sent" in func:
            dims = split[:-2]
            suffix = "_sent_freq"
        else:
            dims = split[:-1]
            suffix = "_freq"
        N = len(dims)
        if N > 1:
            # Conditionals of size n, eg. "head_tail_freq|head_freq"
            for n in range(1, N):
                for cond in itertools.combinations(range(len(dims)), n):
                    cond_func = func + "|" + "_".join([dims[c] for c in cond]) + suffix
                    args_split = [a.split(",") for a in args]
                    cond_arglist = [",".join(a) + "|" + ",".join([a[c] for c in cond]) for a in args_split]
                    stat_func_arglists[cond_func] = cond_arglist
                    # print(f"{cond_func} :: {cond_arglist[:5]}")

# Add two-arcs manually since we have only defined a subset
two_arc_joints = [
    "head_tail1_tail2_dir1_dir2_freq",
    "head_dep1_dep2_dir1_dir2_freq",
    "head_tail_gtail_dir_gdir_freq",
    "tail_dep_gdep_dir_gdir_freq",
]
for func in two_arc_joints:
    i = 1 if func == "head_tail_gtail_dir_gdir_freq" else 0  # that one conditions on the second arg
    s = func.split("_")
    cond_func = func + "|" + s[i] + "_freq"
    args = stat_func_arglists[func]
    args_split = [a.split(",") for a in args]
    cond_arglist = [",".join(a) + "|" + ",".join([a[c] for c in [i]]) for a in args_split]
    # print(f"{cond_func} :: {cond_arglist[:5]}")
    stat_func_arglists[cond_func] = cond_arglist
