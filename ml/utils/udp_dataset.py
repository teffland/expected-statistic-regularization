from glob import glob
import json
import numpy as np
import pandas as pd

# import lang2vec.lang2vec as l2v
import conllu
from collections import defaultdict, Counter
from spacy import displacy
from tqdm import tqdm
from ipywidgets import interact, widgets
from IPython.core.display import display, HTML
from pprint import pprint


class UDPDataset(object):
    """Class for handling UDP language datasets.

    Load a language from a set of conllu files."""

    def __init__(self, lang, conllu_files, results_file=None, uniform_sizes=False, verbose_load=False):
        self.lang = lang
        self.source_files = conllu_files

        datas = []
        for f in tqdm(conllu_files, f"{lang}: reading source files"):
            datas.append(self._read_file(f, verbose=verbose_load))

        if uniform_sizes:
            # Get the average size (rounded), and sample each dataset to that size
            size = round(np.mean([len(data) for data in datas]))
            print(f"Sampling each dataset to size {size}")
            for path, data in zip(conllu_files, datas):
                n = len(data)
                print(f"Choosing from {n} examples for {path}")
                idxs = npr.choice(np.arange(n), size=size, replace=(size > n))
                sample = [data[i] for i in idxs]
                self.data.extend(sample)
        else:
            self.data = [d for data in datas for d in data]

        if results_file is not None:
            self._results = json.load(open(results_file))
        else:
            self._results = {}
        self.stats = {}

    # def calc_stats(self, stat_name, stat_func, *stat_func_args, **stat_func_kwargs):
    #     self.stats[stat_name] = stat_func(self.data, *stat_func_args, **stat_func_kwargs)

    @property
    def num_sentences(self):
        return len(self.data)

    @property
    def num_tokens(self):
        return sum(len(d["words"]) for d in self.data)

    def _read_file(self, f, verbose=False):
        it = tqdm(conllu.parse_incr(open(f)), leave=False) if verbose else conllu.parse_incr(open(f))
        parses = [conllu_to_spacy(ts) for ts in it]
        for i, p in enumerate(parses):
            p["meta"] = {"source_file": f, "uid": f"{f}-S{i}"}
            p["title"] = p["meta"]["uid"]  # [len('../data/ud-treebanks-v2.7/'):]
        return parses

    def results(self, metrics=["UPOS", "UAS", "LAS", "CLAS"], keys=["f1"]):
        return {m: {k: self._results[m][k] for k in keys} for m in metrics}

    def visualize(self, i=0, options=None):
        opts = dict(compact=True, distance=100)
        if options:
            opts.update(options)

        def show(i):
            datum = self.data[i]
            html = displacy.render(datum, style="dep", manual=True, jupyter=False, options=opts)
            html = f'<h2>{datum["title"]}</h2><div><span class="tex2jax_ignore">{html}</span></div>'
            display(HTML(html))

        interact(show, i=widgets.IntSlider(min=0, max=len(self.data) - 1, value=i))

    def visualize_file(self, filename, start=0, end=None, title="Parses", options=None):
        end = len(self.data) if end is None else end
        opts = dict(compact=True, distance=100)
        if options:
            opts.update(options)

        html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
</head>
<body>
        """
        html += f"<h1>{self.lang} - {title}</h1>"
        html += f"<p>{end-start} total parses</p>"
        if self._results:
            html += "".join([f"<p><b>{k}:</b> {v}</p>" for k, v in self.results().items()])

        for i in range(start, end):
            datum = self.data[i]
            parse_html = displacy.render(datum, style="dep", manual=True, jupyter=False, options=opts)
            html += f'<br><h3>{datum["title"]}</h3><div><span class="tex2jax_ignore">{parse_html}</span></div>'

        html += "</body></html>"
        with open(filename + ".html", "w", encoding="utf8") as f:
            f.write(html)
        print("Done")


def conllu_to_spacy(token_list):
    words, arcs = [dict(text="ROOT", tag="ROOT")], []
    i = 1
    clean_dep = lambda dep: dep.split(":")[0]
    for token in token_list:
        #             print('token', token)
        if token["head"] is None:
            continue
        words.append(dict(text=token["form"], tag=token["upostag"]))
        tail, head = i, token["head"]

        start = min(tail, head)
        end = max(tail, head)
        way = "left" if end == head else "right"
        arcs.append(dict(start=start, end=end, tail=tail, head=head, label=clean_dep(token["deprel"]), dir=way))
        i += 1
    return dict(words=words, arcs=sorted(arcs, key=lambda a: a["end"]))
