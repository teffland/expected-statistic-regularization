""" Subsample examples from a conllu file and output to another file

"""
import argparse
import numpy.random as npr
import conllu

# from ml.utils.stat_funcs import stat_func_by_name


def parse_args(args=[]):
    parser = argparse.ArgumentParser("Concat input treebanks to file")
    parser.add_argument("inpaths", type=str, nargs="+")
    parser.add_argument("-o", "--out-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args):
    data = [d for path in args.inpaths for d in conllu.parse(open(path).read())]
    if args.limit is not None:
        data = data[: args.limit]
    with open(args.out_path, "w") as f:
        f.write("".join([d.serialize() for d in data]))
    print(f"Wrote {len(data)} examples to {args.out_path}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
