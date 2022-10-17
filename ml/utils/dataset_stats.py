""" Compare two datasets using a list of stat functions. Optionally we can compute the stats w/ paired bootstrap samples.

It can be run in multi-processing mode.
"""
import math
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import os
import json
from collections import defaultdict, Counter

from ml.utils.stat_funcs import stat_func_by_name


def dump_json_path(out_path, specs):
    directory = os.path.dirname(out_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(out_path, "w") as f:
        json.dump(specs, f, indent=2)


def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"

    return "".join(safe_char(c) for c in s).rstrip("_")


DEFAULT_SUMMARY_STATS = dict(
    mean=np.mean,
    stddev=lambda vs: (np.std(vs, ddof=(0 if len(vs) <= 1 else 1))),
    # min=np.min,
    # max=np.max,
    # The 7-number summary -- if these percentiles are evenly spaced, then the dist is approximately normal
    # percentile_2=lambda vs: np.percentile(vs, 2.15),
    # percentile_9=lambda vs: np.percentile(vs, 8.87),
    # percentile_25=lambda vs: np.percentile(vs, 25),
    # percentile_50=lambda vs: np.percentile(vs, 50),
    # percentile_75=lambda vs: np.percentile(vs, 75),
    # percentile_91=lambda vs: np.percentile(vs, 91.13),
    # percentile_98=lambda vs: np.percentile(vs, 97.85),
)


def compute_stats(
    data,
    stat_func_name,
    stat_func_args,
    n_bootstrap=0,
    batch_size=None,
    seed=0,
    other_data=None,
    loss_func=None,
    pop_raw_stats=True,
    verbose=False,
    ignore_negative=True,
    summary_stats_template=None,
):
    summary_stats_template = summary_stats_template or DEFAULT_SUMMARY_STATS
    out = defaultdict(dict)
    is_cond = "|" in stat_func_name
    empty_cache = lambda: (((None, None), (None, None)) if is_cond else (None, None))
    kwargs = dict(return_counts=(True, True) if is_cond else True)

    # Compute full sample stats
    counts, N = empty_cache()
    if verbose > 1:
        print("Computing stats:", flush=True)
    for arg in stat_func_args:
        stat, counts, N = stat_func_by_name(data, stat_func_name, arg, counts=counts, N=N, **kwargs)
        if verbose > 1:
            print(f"  {arg} : {stat}")
        out[arg].update(dict(stat_func=stat_func_name, arg=arg, value=stat,))

    if other_data:
        if verbose > 1:
            print("Computing other stats:", flush=True)
        counts, N = empty_cache()
        for arg in stat_func_args:
            stat, counts, N = stat_func_by_name(other_data, stat_func_name, arg, counts=counts, N=N, **kwargs)
            if verbose > 1:
                print(f"  {arg} : {stat}")
            out[arg].update(dict(stat_func=stat_func_name, arg=arg, other_value=stat,))

    # Maybe tack on bootstrap stats
    if n_bootstrap:
        b_out, b_other_out = defaultdict(list), defaultdict(list)
        B = n_bootstrap

        npr.seed(seed)
        iterator = range(B)
        if verbose:
            iterator = tqdm(iterator, f"Stat: {stat_func_name}", B, mininterval=1.0)
        for b in iterator:
            size = len(data) if not batch_size else batch_size
            replace = True  # size >= len(data)
            b_idxs = npr.choice(np.arange(len(data)), replace=replace, size=size)
            counts, N = empty_cache()
            b_data = [data[i] for i in b_idxs]
            for arg in stat_func_args:
                stat, counts, N = stat_func_by_name(
                    b_data, stat_func_name, arg, counts=counts, N=N, return_counts=kwargs["return_counts"]
                )
                b_out[arg].append(stat)

            if other_data:
                paired = len(data) == len(other_data)
                counts, N = empty_cache()
                if not paired:
                    b_idxs = npr.choice(np.arange(len(other_data)), replace=replace, size=size)
                b_other_data = [other_data[i] for i in b_idxs]

                for arg in stat_func_args:
                    stat, counts, N = stat_func_by_name(
                        b_other_data, stat_func_name, arg, counts=counts, N=N, return_counts=kwargs["return_counts"]
                    )
                    b_other_out[arg].append(stat)

        for arg in stat_func_args:
            bs = np.array(b_out[arg])
            out_bs = bs[bs >= 0] if ignore_negative else bs
            out_bs = out_bs if len(out_bs) else bs
            neg_count = sum(bs < 0)
            if ignore_negative and neg_count:
                pass  # print(f"{arg}: ignoring {neg_count} negatives")
            out[arg].update(
                dict(
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    n_invalid=neg_count,
                    batch_size=batch_size,
                    **{k: s(out_bs) for k, s in summary_stats_template.items()},
                )
            )
            if not pop_raw_stats:
                out[arg]["values"] = out_bs

            if other_data:
                other_bs = np.array(b_other_out[arg])
                out_other_bs = other_bs[other_bs >= 0] if ignore_negative else other_bs
                out_other_bs = out_other_bs if len(out_other_bs) else other_bs
                neg_count = sum(other_bs < 0)
                if ignore_negative and neg_count:
                    pass  # print(f"{arg}: ignoring {neg_count} other negatives")
                out[arg].update(
                    dict(
                        n_other_invalid=neg_count,
                        **{f"other_{k}": s(out_other_bs) for k, s in summary_stats_template.items()},
                    )
                )
                if not pop_raw_stats:
                    out[arg]["other_values"] = out_other_bs

                diff_bs = bs - other_bs
                out_diff_bs = diff_bs[(bs >= 0) & (other_bs >= 0)]
                out_diff_bs = out_diff_bs if len(out_diff_bs) else diff_bs
                neg_count = sum((bs < 0) | (other_bs < 0))
                if ignore_negative and neg_count:
                    pass  # print(f"{arg}: ignoring {neg_count} diff negatives")
                out[arg].update(
                    dict(
                        n_diff_invalid=neg_count,
                        **{f"diff_{k}": s(out_other_bs) for k, s in summary_stats_template.items()},
                    )
                )
                if not pop_raw_stats:
                    out[arg]["diff_values"] = out_diff_bs

                if loss_func is not None:
                    xs = out_other_bs if len(out_other_bs) else other_bs
                    mean, std = out[arg]["mean"], out[arg]["stddev"]
                    try:
                        loss_values = loss_func(xs=xs, mean=mean, stddev=std)
                    except Exception as e:
                        print("e", e)
                        print(mean, std, len(xs), xs[:10])
                    out[arg].update({f"loss_{k}": s(loss_values) for k, s in summary_stats_template.items()})
    return out


# counts_cache = {}


# def compute_stats(
#     data, stat_func_name, stat_func_args, n_bootstrap, batch_size, seed=0, pop_raw_stats=True, verbose=False,
# ):
#     og_stat_func_args = stat_func_args
#     if type(stat_func_args) is str:
#         stat_func_args = tuple([stat_func_args])

#     stats = []
#     if n_bootstrap:
#         B = n_bootstrap
#         npr.seed(seed)
#         iterator = range(B)
#         if verbose:
#             iterator = tqdm(iterator, f"Stat: {stat_func_name}({og_stat_func_args})", B, mininterval=1.0)
#         for b in iterator:
#             size = len(data) if not batch_size else batch_size
#             replace = size >= len(data)
#             # This is a paired sample to increase power -- the gold and pred datasets are sampled with the same indices
#             b_idxs = npr.choice(np.arange(len(data)), replace=replace, size=size)
#             b_data = [data[i] for i in b_idxs]
#             stats.append(stat_func_by_name(b_data, stat_func_name, *stat_func_args))

#         stats = np.array(stats)
#         out = dict(
#             stat=stat_func_name,
#             stat_args=stat_func_args,
#             og_stat_args=og_stat_func_args,
#             seed=seed,
#             n_bootstrap=n_bootstrap,
#             batch_size=batch_size,
#             mean=np.mean(stats),
#             median=np.median(stats),
#             min=np.min(stats),
#             max=np.max(stats),
#             stddev=np.std(stats),
#         )
#         if not pop_raw_stats:
#             out["raw_stats"] = stats
#     else:
#         v = stat_func_by_name(data, stat_func_name, *stat_func_args)
#         out = dict(stat=stat_func_name, stat_args=stat_func_args, og_stat_args=og_stat_func_args, seed=seed, value=v,)

#     return out


def compare_datasets(
    gold_dataset,
    pred_dataset,
    stat_func_name,
    stat_func_args,
    out_dir=None,
    n_bootstrap=None,
    size=16,
    seed=0,
    verbose=True,
    pop_raw_stats=True,
):
    """
    Parameters:
    -----------
      gold_dataset (UDPDataset)
      pred_dataset (UDPDataset)
      stat_func (function): a function that calculates a stat given a dataset
      f_args (tuple): args to pass to stat_func after dataset
      out_dir
    """
    # print("stat func args", stat_func_args)
    og_stat_func_args = stat_func_args
    # if type(stat_func_args) is str:
    #     stat_func_args = tuple([stat_func_args])
    gold_data = gold_dataset.data
    pred_data = pred_dataset.data
    uid = lambda d: d["meta"]["uid"].split("-")[-1]
    assert len(gold_data) == len(pred_data) and all(
        uid(x) == uid(y) for (x, y) in zip(gold_data, pred_data)
    ), "Datasets are not paired"

    gold_stats, pred_stats = [], []
    if n_bootstrap:
        B = n_bootstrap
        npr.seed(seed)
        # print(f'size: {size}, len(gold): {len(gold_data)}, len(pred): {len(pred_data)}')

        sample_size = size
        iterator = range(B)
        if verbose:
            iterator = tqdm(iterator, f"Stat: {stat_func_name}({og_stat_func_args})", B, mininterval=1.0)
        for b in iterator:
            size = len(gold_data) if sample_size is None else size
            replace = size == len(gold_data)
            # This is a paired sample to increase power -- the gold and pred datasets are sampled with the same indices
            b_idxs = npr.choice(np.arange(len(gold_data)), replace=replace, size=size)
            gold_b = [gold_data[i] for i in b_idxs]
            gold_stats.append(stat_func_by_name(gold_b, stat_func_name, stat_func_args))

            size = len(pred_data) if sample_size is None else size
            replace = size == len(pred_data)
            pred_b = [pred_data[i] for i in b_idxs]
            pred_stats.append(stat_func_by_name(pred_b, stat_func_name, stat_func_args))

        gold_stats = np.array(gold_stats)
        pred_stats = np.array(pred_stats)
        avg_gold, avg_pred = np.mean(gold_stats), np.mean(pred_stats)
        std_gold, std_pred = np.std(gold_stats), np.std(pred_stats)

        if verbose:
            print(f" Gold: avg = {avg_gold:2.5f} +/- {std_gold:2.5f} std", flush=True)
            print(f" Pred: avg = {avg_pred:2.5f} +/- {std_pred:2.5f} std", flush=True)
        stat_info = dict(
            stat=stat_func_name,
            stat_args=stat_func_args,
            gold_stats=list(gold_stats),
            pred_stats=list(pred_stats),
            avg_gold=avg_gold,
            avg_pred=avg_pred,
            std_gold=std_gold,
            std_pred=std_pred,
            n_bootstrap_samples=B,
            sample_size=sample_size,
            seed=seed,
        )
    else:
        gold_stat = stat_func_by_name(gold_data, stat_func_name, stat_func_args)
        pred_stat = stat_func_by_name(pred_data, stat_func_name, stat_func_args)
        stat_info = dict(stat=stat_func_name, stat_args=og_stat_func_args, gold_stat=gold_stat, pred_stat=pred_stat)
        print(f"Stat: {stat_func_name}({og_stat_func_args})")
        print(f" Gold: {gold_stat:2.5f}", flush=True)
        print(f" Pred: {pred_stat:2.5f}", flush=True)

    if out_dir:
        path = os.path.join(out_dir, make_safe_filename(stat_func_name) + ".json")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(path, "w") as outf:
            json.dump(stat_info, outf)

    if pop_raw_stats:
        stat_info.pop("gold_stats", None)
        stat_info.pop("pred_stats", None)
    return stat_info


# import matplotlib.pyplot as plt
# from scipy.stats import norm


# def plot_hist_pair(stat, show_std=True, fig=None, axs=None, show_kde=True, xmin=None, xmax=None, loss_func=None):
#     if fig is None or axs is None:
#         fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
#     s = stat["gold_stats"] + stat["pred_stats"]
#     xmin = min(s) if xmin is None else xmin
#     xmax = max(s) if xmax is None else xmax
#     bins = np.linspace(min(s), max(s), 51)
#     name = f'{stat["stat"]}({stat["stat_args"]})'
#     axs[0].set_title(f"Gold `{name}` bootstrap dist")
#     axs[0].xaxis.set_tick_params(which="both", labelbottom=True)
#     _ = axs[0].hist(stat["gold_stats"], bins=bins, density=True)
#     axs[1].set_title(f"Pred `{name}` bootstrap dist")
#     _ = axs[1].hist(stat["pred_stats"], bins=bins, density=True)
#     axs[1].xaxis.set_tick_params(which="both", labelbottom=True)

#     if show_std:
#         mean = np.mean(stat["gold_stats"])
#         std = np.std(stat["gold_stats"])
#         axs[0].axvline(mean - std, color="red")
#         axs[0].axvline(mean + std, color="red")
#         axs[1].axvline(mean - std, color="red")
#         axs[1].axvline(mean + std, color="red")

#     if show_kde:
#         mean, std = norm.fit(stat["gold_stats"])
#         x = np.linspace(xmin, xmax, 100)
#         y = norm.pdf(x, mean, std)
#         axs[0].plot(x, y, color="green")

#         mean, std = norm.fit(stat["pred_stats"])
#         y = norm.pdf(x, mean, std)
#         axs[1].plot(x, y, color="green")

#     if loss_func is not None:
#         xs = np.linspace(xmin, xmax, 100)
#         ys = np.vectorize(loss_func)(xs)
#         axs[1].plot(xs, ys, color="c")

#     return fig, axs


# def get_top_k_common_or_wrong(func_name, arglist, gold, pred, k=20, top_k=None, worst_k=None):
#     """ Return args with highest top_k support and worst_k incorrect"""
#     top_k, worst_k = (k // 2, k // 2) if k else (top_k, worst_k)
#     scores = [
#         (arg, compare_datasets(gold, pred, func_name, arg, n_bootstrap=None, pop_raw_stats=True)) for arg in arglist
#     ]
#     largest = sorted(scores, key=lambda x: -x[1]["gold_stat"])[:top_k]
#     skip_args = {x[0] for x in largest}
#     worst = sorted(
#         [x for x in scores if x[0] not in skip_args], key=lambda x: -abs(x[1]["gold_stat"] - x[1]["pred_stat"])
#     )[:worst_k]
#     print(f'{func_name}: largest sum @{top_k} = { sum([x[1]["gold_stat"] for x in largest]) } mass')
#     #     print('worst', worst)
#     return [arg for arg, _ in largest + worst]


# def compare_and_plot(func_name, arglist, compare_func, gold, pred, k=10, loss_func=None, make_loss_func=None):
#     """ Compare function on given args up to k and plot their bootstrapped sampling distributions.
#     """
#     if k is not None:
#         k_arglist = get_top_k_common_or_wrong(func_name, arglist, gold, pred, k)
#     else:
#         k_arglist = arglist

#     fig, axs = plt.subplots(2 * len(k_arglist), 1, figsize=(12, 4 * len(k_arglist)), sharex=True)
#     for i, arg in enumerate(k_arglist):
#         print(f"----- {i} / {len(k_arglist)} -----")
#         s = compare_func(gold, pred, func_name, arg)
#         i_axs = axs[2 * i : 2 * i + 2]
#         if make_loss_func is not None:
#             loss_func = make_loss_func(func_name, arg)
#         plot_hist_pair(s, show_std=True, fig=fig, axs=i_axs, show_kde=True, loss_func=loss_func)

#     plt.tight_layout()


def make_min_risk_specs(
    func_arglists,
    compare_func,
    gold,
    pred,
    loss_type="margin",
    margin="auto",
    rescale="auto",
    min_rescale=1e-4,
    func_min_rescales=None,
    out_path=None,
):
    """For each func,arg pair compute its sampling dist for a bootstrapped dataset and save its sufficient params."""
    min_risk_specs = dict()
    for i, (func_name, arglist) in enumerate(func_arglists):
        #         print(f'========== {i} / {len(func_arglists)} ==========')
        vals = dict()
        for j, arg in enumerate(arglist):
            print(f"---------- {i} / {len(func_arglists)} ||| {j} / {len(arglist)} ----------")
            s = compare_func(gold, pred, func_name, arg)
            s_margin = s["std_gold"] if margin == "auto" else margin
            if func_name in func_min_rescales:
                s_min_rescale = func_min_rescales[func_name]
            else:
                s_min_rescale = min_rescale
            s_rescale = max(s["std_gold"], s_min_rescale) if rescale == "auto" else rescale
            vals[arg] = dict(type=loss_type, target=s["avg_gold"], margin=s_margin, rescale=s_rescale)
        min_risk_specs[func_name] = vals

    if out_path is not None:
        dump_json_path(out_path, min_risk_specs)

    return min_risk_specs
