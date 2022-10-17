import torch
from time import time
from tqdm import tqdm
import numpy.random as npr
from argparse import ArgumentParser

import ml.utils.tensor_stat_funcs as TSF
import ml.utils.stat_funcs as SF
from ml.utils.udp_dataset import UDPDataset


def parse_args(arglist=[]):
    parser = ArgumentParser()
    parser.add_argument("--archive", type=str, default="ml/utils/test_tensor_stat_funcs_model.tar.gz")
    parser.add_argument("--data", type=str, default=f"data/ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud-dev.conllu")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--data-sample-size", type=int, default=None)
    parser.add_argument("--funcs", type=str, default=None, nargs="+")
    parser.add_argument("--offset", type=int, default=0, help="Offset of functions, so we can skip already finished")
    parser.add_argument(
        "--preindex", action="store_true", help="Test preindexed variant of stat computation. Note: turns of caching."
    )
    parser.add_argument("--verbose", action="store_true", help="Show individual arg tests")
    if arglist:
        return parser.parse_args(arglist)
    else:
        return parser.parse_args()


def test_func(func_name, arglist, data, predictor, tol=1e-8, verbose=False, preindex=False):
    # print("Tensorizing...", flush=True, end=" ")
    # tensors = TSF.tensorize_and_format(data, predictor)
    tensors = TSF.tensorize_direct(data, predictor)
    # print("done", flush=True)
    kwargs = dict(sample=(False, False)) if "|" in func_name else dict(sample=False)
    try:
        for args in tqdm(arglist, f"Testing {func_name}", len(arglist)):
            argstr = args
            if type(args) is str:
                args = tuple([args])
            try:
                t0 = time()
                x = TSF.stat_func_by_name(
                    tensors, predictor._model.vocab, func_name, *args, preindex=preindex, no_cache=preindex, **kwargs
                )
                t0 = time() - t0
                t1 = time()
                y = torch.tensor(SF.stat_func_by_name(data, func_name, *args))
                t1 = time() - t1
                if verbose:
                    print(f" {func_name}({args}) time: TSF={t0:2.6f} SF time: {t1:2.6f}")
                    print(f"{func_name}{args}: {x.detach().cpu().numpy()} == {y.numpy()} ? ", end="")
                assert torch.allclose(
                    x, y, atol=tol
                ), f"{(x-y).abs()} not <= {tol} for args:{argstr} (TSF={x}, SF={y})"
            except Exception as e:
                fname = f"failed_test_{func_name}_{argstr}.pkl"
                print(f"saving failed test data to {fname}", flush=True)
                with open(fname, "wb") as f:
                    tensors["func_name"] = func_name
                    tensors["args"] = args
                    torch.save(tensors, f)
                raise e
            # print("yes")
    except Exception as e:
        print(f"Testing {func_name} failed with error: {e}")
        raise e


def sample(data, size):
    idxs = npr.choice(list(range(len(data))), size=min(len(data), size), replace=False)
    return [data[i] for i in idxs]


def get_predictor(args):
    print("Loading archive")
    device = 0 if torch.cuda.is_available() else -1
    predictor = TSF.get_predictor(args.archive, cuda_device=device)
    return predictor


def get_dataset(args):
    print("Loading data")
    dataset = UDPDataset("whatever", [args.data])

    print(f"Got {len(dataset.data)} examples")
    return dataset


def run(args, predictor, data):

    if args.funcs:
        funcs = args.funcs
    else:
        funcs = sorted([k for k in SF.stat_func_arglists if "entropy" not in k])
    for i, func in enumerate(funcs[args.offset :]):
        print(f"Func {i+args.offset} : {func}", flush=True)
        test_func(
            func,
            sample(SF.stat_func_arglists[func], args.sample_size),
            data,
            predictor,
            preindex=args.preindex,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    args = parse_args()
    predictor = get_predictor(args)
    dataset = get_dataset(args)
    if args.data_sample_size:
        data = sample(dataset.data, args.data_sample_size)
        print(f"Downsampled to {len(data)} examples")
    else:
        data = dataset.data
    run(args, predictor, data)
