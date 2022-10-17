""" Subsample examples from a conllu file and output to another file

"""
import argparse
import numpy as np
import conllu

# from ml.utils.stat_funcs import stat_func_by_name


def parse_args(args=[]):
    parser = argparse.ArgumentParser("Concat input treebanks to file")
    parser.add_argument("inpath", type=str)
    # parser.add_argument("splits", type=float, nargs="+")
    parser.add_argument("--max-num-dev", 100, help="Max number of examples to use as dev")
    # parser.add_argument("--small-split-ratio", 0.5, help="What ratio to use for dev if chopping of max_num_dev would create")
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args):
    data = conllu.parse(open(args.inpath).read())
    n = len(data)

    # Keep min(max_dev, n/2) examples for dev
    num_dev = np.floor(0.5 * n) if (n // 2) < args.max_num_dev else args.max_num_dev
    print(f"Splitting {n} instances into {n-num_dev} train and {num_dev} dev examples")
    points = [0, num_dev, len(data)]
    print(f"{n}: {points}")
    datas = [data[points[i] : points[i + 1]] for i in range(len(points) - 1)]
    print([len(ds) for ds in datas])
    assert sum(len(ds) for ds in datas) == len(data)
    assert len(datas) == 2

    outf = args.inpath.replace(".conllu", f"_train_split.conllu")
    with open(outf, "w") as f:
        f.write("".join([d.serialize() for d in datas[0]]))
    print(f"Wrote {len(datas[0])} examples to {outf}")

    outf = args.inpath.replace(".conllu", f"_dev_split.conllu")
    with open(outf, "w") as f:
        f.write("".join([d.serialize() for d in datas[0]]))
    print(f"Wrote {len(datas[0])} examples to {outf}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
