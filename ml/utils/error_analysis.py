from copy import deepcopy
from collections import defaultdict, Counter
from tqdm import tqdm

from ipywidgets import interact, widgets
from IPython.core.display import display, HTML
from ml.utils.displacy import render, render_pair
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels

import io
from contextlib import redirect_stdout


DEP_GROUPS = {
    "core nominal": ["nsubj", "obj", "iobj"],
    "core clausal": ["csubj", "ccomp", "xcomp"],
    #     'core modifier': [],
    #     'core function': [],
    "noncore nominal": ["obl", "vocative", "expl", "dislocated"],
    "noncore clausal": ["advcl"],
    "noncore modifier": ["advmod", "discourse"],
    "noncore function": ["aux", "cop", "mark"],
    "nominal nominal": ["nmod", "appos", "nummod"],
    "nominal clausal": ["acl"],
    "nominal modifier": ["amod"],
    "nominal function": ["det", "clf", "case"],
    "coordination": ["conj", "cc"],
    "multiword expression": ["fixed", "flat", "compound"],
    "loose": ["list", "parataxis"],
    "special": ["orphan", "goeswith", "reparandum"],
    "other": ["punct", "root", "dep"],
}
DEP_GROUPS_LIST = list(DEP_GROUPS.items())
DEP_TO_GROUP = {v: k for k, vs in DEP_GROUPS.items() for v in vs}

POS_GROUPS = {
    "open": ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"],
    "closed": ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"],
    "other": ["PUNCT", "SYM", "X", "ROOT"],
}
POS_GROUPS_LIST = list(POS_GROUPS.items())
POS_TO_GROUP = {v: k for k, vs in POS_GROUPS.items() for v in vs}


class ErrorAnalysis(object):
    def __init__(self, gold_dataset, pred_dataset):
        try:
            tqdm._instances.clear()
        except:
            pass
        self.uid = lambda d: d["meta"]["uid"].split("-")[-1]
        self.content_rels = set(
            "nsubj, obj, iobj, csubj, ccomp, xcomp, obl, vocative, expl, "
            "dislocated, advcl, advmod, discourse, nmod, appos, nummod, acl, "
            "amod, conj, fixed, flat, compound, list, parataxis, orphan, goeswith, "
            "reparandum, root, dep".split(", ")
        )

        self.gold = deepcopy(gold_dataset)
        self.pred = deepcopy(pred_dataset)
        self.pairs = self.make_pairs()
        self.detect_errors()
        self.metrics = {}
        self.do_metrics()

    def make_pairs(self):
        """Pair up gold and pred instances."""
        pairs = defaultdict(dict)
        for g in self.gold.data:
            pairs[self.uid(g)]["g"] = g
        for p in self.pred.data:
            pairs[self.uid(p)]["p"] = p

        to_drop = set()
        for uid, pair in pairs.items():
            if not ("g" in pair and "p" in pair):
                print(f"{uid} is missing something, skipping: got {list(pair.keys())}")
                to_drop.add(uid)
        for uid in to_drop:
            pairs.pop(uid)
        return pairs

    @property
    def gold_pred_pairs(self):
        return [(p["g"], p["p"]) for p in self.pairs.values()]

    def detect_errors(self):
        """Record errors between gold/pred pairs."""
        for uid, pair in self.pairs.items():
            g, p = pair["g"], pair["p"]
            pair["metrics"] = {}
            # Find tag errors
            assert len(g["words"]) == len(p["words"])
            for a, b in zip(g["words"], p["words"]):
                assert a["text"] == b["text"]
                a["correct"] = b["correct"] = a["tag"] == b["tag"]
            #                 if not a['correct']:
            #                     print(f'Incorrect g:{a}  vs p:{b}')

            # Find arc errors
            assert len(g["arcs"]) == len(p["arcs"])
            gas = sorted(g["arcs"], key=lambda a: a["tail"])
            pas = sorted(p["arcs"], key=lambda a: a["tail"])
            #             print(len(gas), len(pas))
            for a, b in zip(gas, pas):
                assert a["tail"] == b["tail"]
                #                 print(a, b)
                if a["head"] == b["head"]:
                    if a["label"] == b["label"]:
                        a["correct"] = b["correct"] = True
                    else:
                        a["correct"] = b["correct"] = "wrong-label"
                else:
                    a["correct"] = b["correct"] = False

    def do_metrics(self, specs=None):
        specs = specs or [
            ("upos", "upos", []),
            ("uas", "uas", []),
            ("las", "las", []),
            ("clas", "clas", []),
            ("clas-no-mwe", "xlas", [{s for s in self.content_rels if s not in {"fixed", "compound", "flat"}}]),
        ]

        pairs = list(self.pairs.values())

        for (key, name, args) in specs:
            method = getattr(self, f"calc_{name}")
            self.metrics[key] = method(pairs, *args)
            for p in pairs:
                p["metrics"][key] = method([p], *args)

    def calc_metrics(self, pairs, do=None, xlas=None, methods=None):
        do = do or ["upos", "uas", "las", "clas"]
        metrics = {d: getattr(self, f"calc_{d}")(pairs) for d in do}
        xlas = xlas or []
        for name, labels in xlas:
            metrics[name] = self.calc_xlas(pairs, labels)
        for k, f in methods.items():
            metrics[k] = f(pairs, **metrics)
        return metrics

    def calc_upos(self, pairs):
        tp = sum(sum(a["correct"] for a in pair["g"]["words"]) for pair in pairs)
        n = sum(len(pair["g"]["words"]) for pair in pairs)
        return tp / n

    def calc_uas(self, pairs):
        tp = sum(sum(bool(a["correct"]) for a in pair["g"]["arcs"]) for pair in pairs)
        n = sum(len(pair["g"]["arcs"]) for pair in pairs)
        return tp / n

    def calc_las(self, pairs):
        tp = sum(sum(a["correct"] == True for a in pair["g"]["arcs"]) for pair in pairs)
        n = sum(len(pair["g"]["arcs"]) for pair in pairs)
        return tp / n

    def calc_xlas(self, pairs, rel_set):
        is_ok = lambda a: a["correct"] == True and a["label"] in rel_set
        tp = sum(sum(is_ok(a) for a in pair["g"]["arcs"]) for pair in pairs)
        n = sum(sum(a["label"] in rel_set for a in pair["g"]["arcs"]) for pair in pairs)
        return tp / n

    def calc_clas(self, pairs):
        return self.calc_xlas(pairs, self.content_rels)

    def calc_stats(self, stats=None, n_bootstrap_samples=None, batch_size=8, out_csv_path=None, out_jsd_csv_path=None):
        from ml.cmd.compare_datasets import get_df

        if stats is None:
            stats = "pos_freq head_freq tail_freq dep_freq head_tail_dep_freq head_tail_dep_freq|dep_freq".split()
        self._stats_df, self._jsd_df = get_df(
            self.gold, self.pred, stats, n_bootstrap_samples, batch_size, out_csv_path, out_jsd_csv_path
        )
        return self._stats_df, self._jsd_df

    @property
    def stats_df(self):
        if getattr(self, "_stats_df", None) is None:
            self.calc_stats()
        return self._stats_df

    @property
    def jsd_df(self):
        if getattr(self, "_jsd_df", None) is None:
            self.calc_stats()
        return self._jsd_df

    def visualize(self, filter_uids=None, sort_by=None, i=0, limit=None, options=None, filter_func=None):
        pairs = (
            [self.pairs[uid] for uid in self.pairs if uid in filter_uids] if filter_uids else list(self.pairs.values())
        )
        if filter_func is not None:
            pairs = [p for p in pairs if filter_func(p)]
        if sort_by != False:
            sort_by = sort_by or (lambda p: p["metrics"]["clas"])
            pairs = sorted(pairs, key=sort_by)
        pairs = pairs[:limit] if limit else pairs
        print(f"{len(pairs)} sentences kept")

        opts = dict(compact=True, distance=140)
        if options:
            opts.update(options)

        def show(i):
            pair = pairs[i]
            metrics_string = "".join(
                [
                    f'<span style="padding-left: 10px;"><b>{k.upper()}:&nbsp;</b>{round(pair["metrics"][k],3)}</span>'
                    for k in ["upos", "clas", "las", "uas"]
                ]
            )
            html = render_pair(pair, style="dep", manual=True, jupyter=False, options=opts)
            html = (
                f'<p><b>ID:&nbsp;</b>{pair["g"]["meta"]["uid"]}</p>'
                f"<p>{metrics_string}</p>"
                f"<h5>Gold</h5>"
                f'<div><span class="tex2jax_ignore">{html}</span></div>'
                f"<h5>Pred</h5>"
            )
            display(HTML(html))

        interact(show, i=widgets.IntSlider(min=0, max=len(pairs) - 1, value=i))

    def cm(
        self,
        y_true,
        y_pred,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=None,
        include_values=True,
        xticks_rotation=30,
        values_format=None,
        cmap="Blues",
        ax=None,
        colorbar=True,
        title=None,
        errors_only=False,
        print_k_most_common=0,
        grid=False,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if errors_only:
            diff_only = tuple(zip(*[(t, p) for t, p in zip(list(y_true), list(y_pred)) if t != p]))
            y_true, y_pred = list(diff_only[0]), list(diff_only[1])
        if display_labels is None:
            if labels is None:
                display_labels = unique_labels(y_true, y_pred)
            else:
                display_labels = labels
        cm = confusion_matrix(y_true, y_pred, normalize=normalize, labels=display_labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(
            include_values=include_values,
            cmap=cmap,
            ax=ax,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            colorbar=None,
        )
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(disp.im_, cax=cax)
        if title:
            ax.set_title(title)
        ax.grid(grid)

        if print_k_most_common:
            fix = lambda x: x.replace("\n", " ")
            counts = Counter([(fix(t), fix(p)) for t, p in zip(y_true, y_pred)])
            print("Most Common: ")
            stats = [(i, t, p, c) for i, ((t, p), c) in enumerate(counts.most_common(print_k_most_common))]
            I = max(len(str(s[0])) for s in stats)
            T = max(len(str(s[1])) for s in stats)
            P = max(len(str(s[2])) for s in stats)
            C = max(len(str(s[3])) for s in stats)
            h = f"  {'i':<{I}} |  {'True':^{T}}   {'Pred':^{P}} |  {'Count':>{C}}"
            print(h)
            print("-" * len(h))
            for (i, t, p, c) in stats:
                print(f"  {str(i):<{I}} |  {t:^{T}}   {p:^{P}} |  {str(c):>{C}}")

        return ax

    def pos_cm(self, groups_only=False, **kwargs):
        pairs = list(self.pairs.values())
        tag = (lambda x: POS_TO_GROUP[x]) if groups_only else (lambda x: x)
        y_true = [tag(w["tag"]) for pair in pairs for w in pair["g"]["words"]]
        y_pred = [tag(w["tag"]) for pair in pairs for w in pair["p"]["words"]]
        if not groups_only:
            seen_labels = set(y_true) | set(y_pred)
            print("seen labels", seen_labels)
            labels = [v for _, vs in POS_GROUPS_LIST for v in vs if v in seen_labels]
            kwargs["labels"] = labels
            ax = self.cm(y_true, y_pred, title="POS Confusion Matrix", **kwargs)
            x = -0.5
            ax.xaxis.labelpad = 30
            ax.yaxis.labelpad = 50
            for i, (k, vs) in enumerate(POS_GROUPS_LIST):
                x0 = x
                x += len([v for v in vs if v in seen_labels])
                m = x0 + (x - x0) / 2.0
                if i < len(POS_GROUPS) - 1:
                    ax.axvline(x, -1, len(seen_labels), ls="--", lw=1)
                    ax.axhline(x, -1, len(seen_labels), ls="--", lw=1)
                ax.text(m, len(seen_labels) + 2.0, k.replace(" ", "\n"), ha="center", size=10)
                ax.text(-3.0, m, k.replace(" ", "\n"), ha="right", va="center", size=10)
            return ax
        else:
            return self.cm(y_true, y_pred, title="POS Confusion Matrix", **kwargs)

    def dep_cm(self, grounded_only=False, show_groups=True, groups_only=False, **kwargs):
        pairs = list(self.pairs.values())
        y_true, y_pred = [], []
        sort = lambda arcs: sorted(arcs, key=lambda a: a["tail"])
        for pair in pairs:
            for t, p in zip(sort(pair["g"]["arcs"]), sort(pair["p"]["arcs"])):
                if groups_only:
                    t_label = DEP_TO_GROUP[t["label"]].replace(" ", "\n")
                    p_label = DEP_TO_GROUP[p["label"]].replace(" ", "\n")
                else:
                    t_label = t["label"]
                    p_label = p["label"]
                if grounded_only:
                    if t["head"] == p["head"]:
                        y_true.append(t_label)
                        y_pred.append(p_label)
                else:
                    y_true.append(t_label)
                    y_pred.append(p_label)

        if not groups_only and show_groups:
            seen_labels = set(y_true) | set(y_pred)
            labels = [v for _, vs in DEP_GROUPS_LIST for v in vs if v in seen_labels]
            kwargs["labels"] = labels
            ax = self.cm(y_true, y_pred, title="Dep Label Confusion Matrix", **kwargs)
            x = -0.5
            ax.xaxis.labelpad = 30
            ax.yaxis.labelpad = 50
            for k, vs in DEP_GROUPS_LIST[:-1]:
                x0 = x
                x += len([v for v in vs if v in seen_labels])
                m = x0 + (x - x0) / 2.0
                ax.axvline(x, -1, len(seen_labels), ls="--", lw=1)
                ax.axhline(x, -1, len(seen_labels), ls="--", lw=1)
                ax.text(m, len(seen_labels) + 2.0, k.replace(" ", "\n"), ha="center", size=10)
                ax.text(-3.0, m, k.replace(" ", "\n"), ha="right", va="center", size=10)
            return ax
        else:
            seen_labels = set(y_true) | set(y_pred)
            labels = [k.replace(" ", "\n") for k, _ in DEP_GROUPS_LIST if k.replace(" ", "\n") in seen_labels]
            kwargs["labels"] = labels
            return self.cm(y_true, y_pred, title="Dep Label Confusion Matrix", **kwargs)

    def _dep_info(self, dep, k=3):
        df = self._stats_df

        cond_df = df[df["stat_func"] == "head_tail_dep_freq|dep_freq"]
        cond_df.loc[:, "given"] = cond_df["arg"].map(lambda r: r.split("|")[-1])
        cond_df.loc[:, "head_tail"] = cond_df["arg"].map(lambda r: ",".join(r.split("|")[0].split(",")[:2]))

        print(f"\n---- {dep} ----")
        a = df[df["stat_func"] == "dep_freq"]
        p = a[a["arg"] == dep]["value"].tolist()[0]

        dep_df = cond_df[cond_df["given"] == dep]
        vals = dep_df[["head_tail", "value", "other_value"]].sort_values("value", ascending=False)

        #     print(vals[vals['value'] == 0.0]['other_value'])
        v = vals[vals["value"] == 0.0]["other_value"].clip(0.0).sum()
        print(f"gold prior prob: {100.*p:2.3f} %")
        print(f"pred prior prob: {100.*a[a['arg'] == dep]['value'].tolist()[0]} %")
        print(f"prob pred is invalid given dep: {100.*v:2.3f} %")
        print(f"joint invalid prob: {100.*p*v:2.3f} %")

        n, N = len(vals[vals["value"] > 0.0]), len(vals)
        print(f"possible gold combos: {n} / {N} = {100.*n/N:2.3f}% of args")

        if k:
            print("\nMost prevalent golds")
            print(vals[vals["value"] > 0.0][["head_tail", "value", "other_value"]].head(k).to_string(index=False))
            print("\nMost common mistakes")
            print(
                vals[vals["value"] == 0.0][["head_tail", "value", "other_value"]]
                .sort_values("other_value", ascending=False)
                .head(k)
                .to_string(index=False)
            )
        #         html(vals[vals['value'] > 0.0][['head_tail','value']].head(k))
        return p, v, p * v

    def dep_stats_report(self, k=3):
        s = ""
        with io.StringIO() as buf, redirect_stdout(buf):
            infos = dict()
            for name, group in DEP_GROUPS_LIST:
                print(f"\n\n===================\n {name} \n===================")
                for dep in group:
                    p, v, pv = self._dep_info(dep, k)
                    infos[dep] = dict(prior=p, cond=v, joint=pv)
            # details = buf.getvalue()

        with io.StringIO() as buf, redirect_stdout(buf):
            print("Impossible combo prob by category")
            ginfos = dict()
            for g, deps in DEP_GROUPS_LIST:
                p = sum(infos[dep]["prior"] for dep in deps)
                pv = sum(infos[dep]["joint"] for dep in deps)
                ginfos[g] = {"prior": p, "joint": pv, "deps": deps}
            gdf = pd.DataFrame(ginfos).transpose().sort_values("joint", ascending=False)
            print(gdf.to_string())
            print("Col totals")
            print(gdf.sum(axis=0).to_string())
            s += buf.getvalue()

        with io.StringIO() as buf, redirect_stdout(buf):
            print("\nImpossible combo prob by dependency type")
            df = pd.DataFrame(infos).transpose().sort_values("joint", ascending=False)
            print(df.to_string())
            print("Col totals")
            print(df.sum(axis=0).to_string())
            s += buf.getvalue()

        with io.StringIO() as buf, redirect_stdout(buf):
            for name in gdf.index.to_list():
                print(f"\n\n===================\n {name} \n===================")
                print(f"Impossible prob: {gdf.loc[name,'joint']:2.6f}")
                group = DEP_GROUPS[name]
                for dep in group:
                    p, v, pv = self._dep_info(dep, k)
            details = buf.getvalue()

        s += f"\n****************************\n  BREAKDOWN BY GROUP, LABEL\n\n{details}"
        return s


def heatmap(
    x,
    rowlabels=None,
    collabels=None,
    rowlabel=None,
    collabel=None,
    title=None,
    ax=None,
    show_vals=False,
    colorbar=False,
):
    ax = ax or plt.subplots(1, 1)[1]
    im = ax.imshow(x.detach().numpy(), cmap="Blues",)
    if rowlabels:
        ax.set_yticks(range(x.shape[0]))
        ax.set_yticklabels(rowlabels)
    if collabels:
        ax.set_xticks(range(x.shape[1]))
        ax.set_xticklabels(collabels, rotation=30, rotation_mode="default")
    if rowlabel:
        ax.set_ylabel(rowlabel)
    if collabel:
        ax.set_xlabel(collabel)
    if title:
        ax.set_title(title)
    if show_vals:
        n, m = x.shape
        for i in range(n):
            for j in range(m):
                ax.text(j - 0.25, i, f"{x[i,j]:2.2f}")

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return ax
