""" Compare a raw (prediction) conllu files to some raw gold files to produce an report of divergences and 
find which stats under the raw predictions have the highest losses.

The script works as follows. We first calculate the descriptive statistics for the gold and pred datasets. 
From here we go per stat_func, identifying the top k arguments from the gold data and then we getting the worst k stats
by comparing the worst 2k stats (in singular divergence) according to the actual bootstrapped loss distribution. 
We use this slightly complicated method for estimating worst k for efficiency, since doing the full bootstrapped loss
calculation for all args often intractable, but doing it for 50 is much more reasonable.

"""
import argparse
import json
import os
from multiprocessing import Pool
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from time import time
from collections import defaultdict
import torch
import torch.nn.functional as F
import pandas as pd

from ml.cmd.evaluate_typology_pair import get_jsd_df
from ml.utils.udp_dataset import UDPDataset
from ml.utils.stat_funcs import stat_func_by_name, stat_func_arglists
from ml.utils.dataset_stats import dump_json_path, compare_datasets, compute_stats


def parse_args(args=[]):
    parser = argparse.ArgumentParser("Compute descriptive stats for some udp treebanks and given stat functions")
    parser.add_argument("--gold-paths", type=str, nargs="+")
    parser.add_argument("--pred-paths", type=str, nargs="+")
    parser.add_argument("--stat-funcs", type=str, nargs="+", help="Names of stat functions to run")
    parser.add_argument("--out-csv-path", type=str, default=None, help="Path to file to write results table to.")
    parser.add_argument("--out-jsd-csv-path", type=str, default=None, help="Path to write the jsd comparison to.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--n-bootstrap-samples", type=int, default=1000, help="Number of dataset resamplings of minibatch size",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Minibatch size for sampling dist",
    )
    parser.add_argument("--uniform-pred-sizes", action="store_true")

    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args):
    gold = UDPDataset("gold", args.gold_paths)
    pred = UDPDataset("pred", args.pred_paths or [], uniform_sizes=args.uniform_pred_sizes)
    if args.pred_paths:
        print(f"Got {len(gold.data)} gold, {len(pred.data)} preds")
    # Compute the stats for each stat_func

    # print(args.stat_funcs)
    return get_df(
        gold,
        pred,
        args.stat_funcs,
        args.n_bootstrap_samples,
        args.batch_size,
        args.out_csv_path,
        args.out_jsd_csv_path,
        args.overwrite,
    )


def get_df(
    gold, pred, stat_funcs, n_bootstrap_samples, batch_size, out_csv_path=None, out_jsd_csv_path=None, overwrite=False
):
    stats = dict()

    old_df = None
    if not overwrite and out_csv_path is not None and os.path.exists(out_csv_path):
        old_df = pd.read_csv(out_csv_path).set_index("id")
        already_finished = set(old_df["stat_func"].unique())
        stat_funcs = list(set(stat_funcs) - already_finished)
        print(f"Skipping existing stat funcs: {already_finished}")

    for i, stat_func_name in enumerate(stat_funcs):
        print(f"\n\nStat func {i+1} / {len(stat_funcs)}", flush=True)
        t0 = time()
        func_domain = stat_func_arglists[stat_func_name]

        def loss_func(xs, mean, stddev):
            xs = torch.tensor(xs)
            # print(f'\nLoss: {mean}, {stddev}')
            if (mean < 0.0) or (stddev < 0.0):
                ys = -1 * torch.ones_like(xs)
            else:
                targets = mean * torch.ones_like(xs)
                ys = torch.nn.functional.smooth_l1_loss(xs, targets, beta=stddev, reduction="none")
            return ys.numpy().reshape(-1)

        func_stats = compute_stats(
            data=gold.data,
            other_data=pred.data,
            stat_func_name=stat_func_name,
            stat_func_args=func_domain,
            n_bootstrap=n_bootstrap_samples,
            batch_size=batch_size,
            seed=0,
            loss_func=loss_func,
            verbose=1,
        )
        stats[stat_func_name] = func_stats
        print(f"\nStat func {i+1} / {len(stat_funcs)} took {(time()-t0)/60.0:2.4f} mins", flush=True)

    # Format as a table and save it
    stat_df = pd.DataFrame([arg for func_stats in stats.values() for arg in func_stats.values()])
    # print('stat_df', stat_df.columns)
    stat_df["id"] = stat_df["stat_func"] + "-" + stat_df["arg"]
    stat_df.set_index("id", inplace=True)
    

    if out_csv_path:
        if not args.overwrite and os.path.exists(out_csv_path):
            stat_df = stat_df.combine_first(old_df)
        print("Writing stats to csv: ", stat_df["stat_func"].unique())
        stat_df.to_csv(out_csv_path)

    # Run a quick JSD analysis and maybe output that
    jsd_df = None
    if out_jsd_csv_path:
        jsd_df = get_jsd_df(stat_df)
        jsd_df.to_csv(out_jsd_csv_path)
        print(f"JSD analysis from {out_csv_path} output to {out_jsd_csv_path}")
        print(f"JSD table is:\n{jsd_df.transpose()}")
    return stat_df, jsd_df


if __name__ == "__main__":
    args = parse_args()
    run(args)
