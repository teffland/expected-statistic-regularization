""" Subsample examples from a conllu file and output to another file

"""
import argparse
import numpy.random as npr
import conllu
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer


# from ml.utils.stat_funcs import stat_func_by_name


def parse_args(args=[]):
    parser = argparse.ArgumentParser("Concat and subsample input treebanks to file")
    parser.add_argument("inpath", type=str)
    parser.add_argument("-s", "--sample-size", type=int, default=0)
    parser.add_argument("-r", "--rseed", type=int, default=0)
    parser.add_argument("-m", "--model", type=str, default="/home/ubuntu//config/bert-base-multilingual-cased/vocab.txt")
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args):
    # Load data and sample it
    print(f'Loading data from {args.inpath}')
    data = conllu.parse(open(args.inpath).read())
    n = len(data)
    npr.seed(args.rseed)
    if args.sample_size:
        idxs = npr.choice(list(range(n)), size=args.sample_size, replace=(args.sample_size > n))
        sample = [data[i] for i in idxs]
    else:
        sample = data

    # Compute and output the flattened tokens
    bert_tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=False)
    tokenize = bert_tokenizer.wordpiece_tokenizer.tokenize

    word_out_path = f'{args.inpath.replace(".conllu","")}_words.txt'
    subword_out_path = f'{args.inpath.replace(".conllu","")}_subwords.txt'
    word_f = open(word_out_path, "w")
    subword_f = open(subword_out_path, "w")

    for sent in tqdm(sample, f"Writing words and subwords", len(sample)):
        words = [t['form'] for t in sent.tokens ]
        subwords = [ token for word in words for token in tokenize(word)]
        word_f.write(f'{" ".join(words)}\n')
        subword_f.write(f'{" ".join(subwords)}\n')

    word_f.close()
    subword_f.close()
    print(f"Done")
    


if __name__ == "__main__":
    args = parse_args()
    run(args)
