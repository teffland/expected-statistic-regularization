""" Subsample examples from a conllu file and output to another file

"""
import argparse
import numpy.random as npr
import conllu

# from ml.utils.stat_funcs import stat_func_by_name


def parse_args(args=[]):
    parser = argparse.ArgumentParser("Subsample input treebank to file")
    parser.add_argument("inpath", type=str)
    parser.add_argument("-s", "--sample-size", type=int)
    parser.add_argument("-o", "--out-path", type=str, default=None)
    parser.add_argument("-r", "--rseed", type=int, default=0)
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args):
    data = conllu.parse(open(args.inpath).read())
    n = len(data)
    npr.seed(args.rseed)
    idxs = npr.choice(list(range(n)), size=args.sample_size, replace=(args.sample_size > n))
    sample = [data[i] for i in idxs]

    out_path = (
        args.out_path
        if args.out_path
        else f'{args.inpath.replace(".conllu","")}_S{args.sample_size}_R{args.rseed}.conllu'
    )
    with open(out_path, "w") as f:
        f.write("".join([d.serialize() for d in sample]))
    print(f"Wrote {args.sample_size} /{n} examples from {args.inpath} to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
