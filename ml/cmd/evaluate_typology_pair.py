""" Take a dataset comparison typology csv (the output of compare_datasets.py) and output an evaluation summary
of the typological divergence between the two typologies. 
"""
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import dirichlet


def stat_df_jsd(stat_df):
    p = stat_df["value"].to_numpy()
    q = stat_df["other_value"].to_numpy()
    # print(p, q)
    return jsd(p, q, base=2)


def stat_df_random_jsd(stat_df, n=100):
    p = stat_df["value"].to_numpy()
    d = len(p)
    alphas = np.ones(d)
    qs = dirichlet.rvs(alphas, size=n)
    jsds = np.array([jsd(p, qs[i], base=2) for i in range(n)])
    return np.mean(jsds)


def get_jsd_df(stats_df):
    stats = dict()
    for stat_func in stats_df["stat_func"].unique():
        if "|" in stat_func:
            # Conditionals have a different dist for each conditioning arg
            stat_func_df = stats_df[stats_df["stat_func"] == stat_func]
            stat_func_df["cond_arg"] = stat_func_df["arg"].map(lambda v: v.split("|")[1])
            cond_args = stat_func_df["cond_arg"].unique()
            n = len(cond_args)
            valid = set()
            for arg in cond_args:
                stat_df = stat_func_df[stat_func_df["cond_arg"] == arg]
                div = stat_df_jsd(stat_df)
                random_div = stat_df_random_jsd(stat_df)
                if np.isnan(div):
                    print(
                        f"WARNING: JSD is undefined for {stat_func}={arg} since conditioning event {arg} never happened, assigning 0.0 weight"
                    )
                    div = -1.0  # not a valid jsd, but won't NaNify the weighted average
                else:
                    valid.add(arg)
                # print("arg", arg)
                stats[f"{stat_func}={arg}"] = dict(jsd=div, random_jsd=random_div)
            for arg in cond_args:
                stats[f"{stat_func}={arg}"]["weight"] = 1 / len(valid) if arg in valid else 0.0
        else:
            stat_df = stats_df[stats_df["stat_func"] == stat_func]
            div = stat_df_jsd(stat_df)
            random_div = stat_df_random_jsd(stat_df)
            stats[stat_func] = dict(jsd=div, random_jsd=random_div, weight=1.0)

    df2 = pd.DataFrame(stats)
    # df2.head()
    # print(df2.columns, df2.index)
    w = df2.loc["weight"]
    weighted_avg = (w * df2.loc["jsd"]).sum() / w.sum()
    weighted_ravg = (w * df2.loc["random_jsd"]).sum() / w.sum()
    # print(df2.loc["weight"].sum())
    # print(df2.loc["jsd"] * df2.loc["weight"])
    df2["Avg"] = [weighted_avg, weighted_ravg, 0.0]
    return df2


def parse_args(args=[]):
    parser = argparse.ArgumentParser("Concat and subsample input treebanks to file")
    parser.add_argument("incsv", type=str)
    parser.add_argument("outcsv", type=str, nargs="?", default=None)
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args):
    stat_df = pd.read_csv(args.incsv)
    jsd_df = get_jsd_df(stat_df)
    if args.outcsv:
        jsd_df.to_csv(args.outcsv)
    print(f"JSD analysis from {args.incsv} output to {args.outcsv}")
    print(f"JSD table is:\n{jsd_df.transpose()}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
