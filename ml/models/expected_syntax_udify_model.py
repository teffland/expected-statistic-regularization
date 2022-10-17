"""
The base UDify model for training and prediction

ADAPT THE UDIFY MODEL TO PERFORM THE EXTRA LOSS CALC
"""

from typing import Optional, Any, Dict, List, Tuple
from overrides import overrides
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D
import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from time import time
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_log_softmax
from allennlp.training.metrics import Average

from torch_struct import NonProjectiveDependencyCRF

from ml.udify.modules.scalar_mix import ScalarMixWithDropout
from ml.udify.models.udify_model import UdifyModel
from ml.utils.stat_funcs import stat_func_arglists, parse_argstr
from ml.utils.tensor_stat_funcs import stat_func_by_name, get_token_index

import pickle

logger = logging.getLogger(__name__)


@Model.register("expected_syntax_udify_model", exist_ok=True)
class ExpectedSyntaxUdifyModel(UdifyModel):
    """
    The UDify model base class. Applies a sequence of shared encoders before decoding in a multi-task configuration.
    Uses TagDecoder and DependencyDecoder to decode each UD task.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tasks: List[str],
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        decoders: Dict[str, Model],
        post_encoder_embedder: TextFieldEmbedder = None,
        dropout: float = 0.0,
        word_dropout: float = 0.0,
        mix_embedding: int = None,
        layer_dropout: float = 0.0,
        loss_cfg: Dict[str, Any] = None,
        loss_specs: Dict[str, Any] = None,  # deprecated: for backwards compatibility
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(ExpectedSyntaxUdifyModel, self).__init__(
            vocab,
            tasks,
            text_field_embedder,
            encoder,
            decoders,
            post_encoder_embedder,
            dropout,
            word_dropout,
            mix_embedding,
            layer_dropout,
            initializer,
            regularizer,
        )
        self.coarse_dep_tags = self.decoders["deps"].coarse_dep_tags
        self.dep_label_fine_to_coarse_reduce_matrix = self.decoders["deps"].dep_label_fine_to_coarse_reduce_matrix
        self.use_greedy_arc_ps = False  # use_greedy_arc_ps
        self.detach_arc_ps = False  # detach_arc_ps
        if loss_cfg:
            self._setup_losses(loss_cfg)
        else:
            self.losses = None
        self.metrics["_model_time"] = Average()
        self.metrics["supervised_loss"] = Average()
        self.metrics["unsupervised_loss"] = Average()
        # self.seen_ids = defaultdict(set)

    def _setup_losses(self, cfg):
        """Take superivision specs from file and create supervision loss tensors."""
        print("Setting up losses", flush=True)
        assert cfg is not None
        self.stats_df = df = pd.read_csv(cfg["stats_csv_path"])  # stats_csv_path)
        df["id"] = df["stat_func"] + "-" + df["arg"]
        df = df.set_index("id")
        self.unsup_loss_weight = cfg.get("total_weight", 1.0)
        self.losses = dict()
        self.metrics["_total_loss_time"] = Average()
        for stat_func, loss_spec in cfg["losses"].items():
            if loss_spec.get("weight", 1.0) == 0.0:
                continue
            print(f"{stat_func} spec: {loss_spec}", flush=True)
            func_domain = stat_func_arglists[stat_func]
            loss_args = defaultdict(list)

            # Maybe restrict the arglist if we pass top or worst k > 0
            if loss_spec.get("topk", 0) + loss_spec.get("worstk", 0) + bool(loss_spec.get("zeros_only", 0)):
                topks, worstks, zeros = [], [], []
                topk, worstk, zeros_only = (
                    loss_spec.get("topk", 0),
                    loss_spec.get("worstk", 0),
                    bool(loss_spec.get("zeros_only", False)),
                )
                func_df = df[df["stat_func"] == stat_func]
                n = len(func_df)
                if topk:
                    topk = int(topk * n) if type(topk) == float else topk
                    topk = max(min(topk, loss_spec.get("topk_max", int(1e8))), loss_spec.get("topk_min", 1))
                    top_df = func_df.sort_values("mean", ascending=False)  # largest mass args
                    for i in range(min(topk, len(top_df))):
                        topks.append(top_df.iloc[i]["arg"])
                        print(
                            f'Func: {stat_func} top {i} / {topk} arg: {topks[-1]} has mean: {top_df.iloc[i]["mean"]:2.6f}',
                            flush=True,
                        )
                if worstk:
                    worstk = int(worstk * n) if type(worstk) == float else worstk
                    worstk = max(min(worstk, loss_spec.get("worstk_max", int(1e8))), loss_spec.get("worstk_min"))
                    skip = set(topks)
                    worst_df = func_df.sort_values("loss_mean", ascending=False)  # largest deviations of
                    for i in range(len(worst_df)):
                        if len(worstks) >= worstk:
                            # finished when worstks is long enough or we run out of args
                            break
                        arg = worst_df.iloc[i]["arg"]
                        if arg not in skip:
                            worstks.append(arg)
                            print(
                                f'Func: {stat_func} worst {i} / {worstk} arg: {worstks[-1]} has loss value: {worst_df.iloc[i]["loss_mean"]:2.6f}'
                                f'  (gold mean:{worst_df.iloc[i]["mean"]:2.6f} , gold std:{worst_df.iloc[i]["stddev"]:2.6f} , pred:{worst_df.iloc[i]["other_mean"]:2.6f})',
                                flush=True,
                            )
                if zeros_only:
                    zero_df = func_df[func_df["mean"] == 0.0]
                    for i in range(len(zero_df)):
                        arg = zero_df.iloc[i]["arg"]
                        zeros.append(arg)
                    print(f"Func: {stat_func} the following {len(zeros)}/{len(func_domain)} args are zero")
                if zeros:
                    arglist = zeros
                else:
                    arglist = topks + worstks
            else:
                arglist = loss_spec.get("arglist", func_domain)

            print(f"Func: {stat_func} has {len(arglist)} / {len(func_domain)} final args")

            indices, targets, margins = [], [], []
            self.metrics[f"_{stat_func}_loss"] = Average()
            self.metrics[f"_{stat_func}_time"] = Average()
            for argstr in tqdm(arglist, f"Setting up args losses for {stat_func}", len(arglist)):
                if argstr == "":
                    arg_idxs = []
                else:
                    args = parse_argstr(argstr)  # [0] if "|" in stat_func else parse_argstr(argstr)
                    arg_idxs = get_token_index(args, self.vocab)

                try:
                    row = df.loc[stat_func + "-" + argstr]
                except:
                    print(f"Couldnt find id: `{stat_func}-{argstr}` in stats df")
                try:
                    target = float(loss_spec["target"])
                except:
                    target = row[loss_spec["target"]]
                if target == -1:
                    # This value is undefined (conditional where the conditioning event never happens),
                    # so omit from the losses
                    continue

                try:
                    margin = float(loss_spec["margin"])
                except:
                    if loss_spec["margin"] == "uniform":
                        margin = 1 / len(func_domain)
                    elif loss_spec["margin"] == "max(uniform,stddev)":
                        uniform = 1 / len(func_domain)
                        stddev = row["stddev"]
                        margin = max(uniform, stddev)
                    else:
                        margin = row[loss_spec["margin"]]

                indices.append(arg_idxs)
                targets.append(target)
                margins.append(margin)

            b = len(indices)
            if "|" in stat_func:
                indices = [
                    torch.LongTensor([idx[0] for idx in indices]).reshape(b, -1).unbind(dim=1),
                    torch.LongTensor([idx[1] for idx in indices]).reshape(b, -1).unbind(dim=1),
                ]
                stat_kwargs = {
                    k: (v, v)
                    for k, v in loss_spec.items()
                    if k not in ("type", "weight", "target", "margin", "arglist", "topk", "worstk")
                }
            else:
                # Gives tuple of index tensors per dimension for selecting elements from event tensors
                indices = torch.LongTensor(indices).reshape(b, -1).unbind(dim=1)
                stat_kwargs = {
                    k: v
                    for k, v in loss_spec.items()
                    if k not in ("type", "weight", "target", "margin", "arglist", "topk", "worstk")
                }
            targets = torch.FloatTensor(targets)
            margins = torch.FloatTensor(margins)

            self.losses[stat_func] = dict(
                type=loss_spec["type"],
                weight=loss_spec.get("weight", 1.0),
                indices=indices,
                targets=targets,
                margins=margins,
                kwargs=stat_kwargs,
            )

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        metadata: List[Dict[str, Any]] = None,
        **kwargs: Dict[str, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        """Run the udify model, then do all of the losses."""
        t0 = time()
        output_dict = super(ExpectedSyntaxUdifyModel, self).forward(tokens=tokens, metadata=metadata, **kwargs)
        self.metrics["_model_time"](time() - t0)
        # for m, d in zip(metadata, kwargs.get("dataset", ["val"] * len(metadata))):
        #     self.seen_ids[d].add(m["uid"])
        # print("seen ids", {k: len(vs) for k, vs in self.seen_ids.items()}, flush=True)
        try:
            for_min_risk = metadata[0].get("for_min_risk", self.losses is not None)
            # print("for_min_risk?", for_min_risk, kwargs.get("dataset", ["no dataset"])[0], flush=True)
            if self.training and for_min_risk:
                t0 = time()
                risk_loss = self.min_risk_loss(
                    tag_logits=output_dict["tag_logits"],
                    arc_logits=output_dict["arc_logits"],
                    label_logits=output_dict["label_logits"],
                    mask=output_dict["mask"],
                )
                self.metrics["_total_loss_time"](time() - t0)
                output_dict["loss"] = risk_loss
                self.metrics["unsupervised_loss"](risk_loss.detach().cpu().numpy())
            else:
                self.metrics["supervised_loss"](output_dict["loss"].detach().cpu().numpy())
        except Exception as e:
            with open("failed_data.pkl", "wb") as f:
                output_dict["tokens"] = tokens
                output_dict["metadata"] = metadata
                pickle.dump(output_dict, f)
            raise e

        return output_dict

    def min_risk_loss(self, tag_logits, arc_logits, label_logits, mask):
        """Calculate all of the specified losses."""
        og_arc_logits, og_mask = arc_logits, mask
        tag_ps, label_ps, arc_ps, arc_logits, arc_crf, mask_adj, og_arc_logits = self.parse_logits_to_probs(
            tag_logits, label_logits, arc_logits, mask
        )
        mask = mask[:, 1:].float()  # chop root off
        tensors = dict(
            pos_logits=tag_logits,
            pos_ps=tag_ps,
            arc_logits=arc_logits,
            og_arc_logits=og_arc_logits,
            arc_crf=arc_crf,
            arc_ps=arc_ps,
            label_logits=label_logits,
            label_ps=label_ps,
            mask=mask,
            mask_adj=mask_adj,
        )
        B, N, _, L = label_ps.shape

        total_loss = 0.0
        num_losses = 0
        all_losses = []
        for stat_func_name, specs in self.losses.items():
            w = specs.get("weight", 1.0)
            if w == 0.0:
                continue
            t0 = time()
            # print('stat_func_name', stat_func_name)
            idxs = specs["indices"]
            # print('idxs', len(idxs), [len(idx) for idx in idxs])
            # print('spec kwargs', specs["kwargs"])
            stats = stat_func_by_name(tensors, self.vocab, stat_func_name, *specs["indices"], **specs["kwargs"])
            losses = self.loss_func(specs["type"], stats, specs["targets"], specs["margins"])
            if specs.get("mean_after_loss", False):
                losses = losses.mean(dim=0)  # losses were left unaggregated over batch dimension
            loss = abs(w) * losses.sum()
            if w > 0.0:
                # Positive weight incorporates into loss, but a negative weight will just log it
                if torch.isnan(loss):
                    print(f"WARNING: Got nan loss for {stat_func_name} -- skipping.", flush=True)
                    print(stats.shape, torch.isnan(stats).any(), torch.isnan(specs["targets"]), stats)
                    continue
                total_loss += loss
                num_losses += w
                all_losses.append((stat_func_name, loss))

            self.metrics[f"_{stat_func_name}_time"](time() - t0)
            self.metrics[f"_{stat_func_name}_loss"](loss.detach().cpu().numpy())

        loss = total_loss / num_losses
        loss = self.unsup_loss_weight * loss
        if torch.isnan(loss):
            with open("tensors.pkl", "wb") as f:
                tensors["losses"] = all_losses
                torch.save(tensors, f)
        return loss

    def loss_func(self, dist_type, stats, targets, margins=None):
        targets = targets.to(stats.device)
        if dist_type == "l1":
            return (stats - targets).abs()

        elif dist_type == "l1-margin":
            margins = margins.to(stats.device)
            return ((stats - targets).abs() - margins).clamp(min=0.0)

        elif dist_type == "l2":
            return (stats - targets) ** 2

        elif dist_type == "xent":
            p, q = targets, stats
            return -(p * q.log())

        elif dist_type == "smoothl1":
            margins = margins.to(stats.device)
            l1 = (stats - targets).abs()
            do_l1 = (l1 >= margins).float()
            scaled_l2 = 0.5 * ((stats - targets) ** 2) / margins.clamp(1e-7)
            loss = do_l1 * l1 + (1 - do_l1) * scaled_l2
            return loss
        else:
            raise ValueError(f"Invalid dist type: {dist_type}")

    def parse_logits_to_probs(self, tag_logits, label_logits, arc_logits, mask, safe_arc_ps=True):
        """Convert the logits from udify decoders to adjacency representation used by min risk losses.

        The main thing here is to convert the label and arc logits from an (N+1)x(N+1) format where the ROOT symbol
        is the first token to an NxN format with the root scores on the diagonals that is preferred by torch_struct,
        which we use for getting marginal arc probs.

        The other thing is that we convert the label probs over fine-grained dependencies to those over coarse-grained
        dependencies since the fine-grained ones are specific to each language so for oov langs it's hard to transfer.

        The dimensional conversions are:
          * tags: (B, N, T) -> (B, N, T)
          * labels: (B, L_fine, N+1 (head), N+1 (tail)) -> (B, N (head), N (tail), L_coarse)
          * arcs: (B, N+1 (head), N+1 (child)) -> (B, N (head), N (child))

        If safe_arc_ps = True, then we renormalize the arc logits per child and clamp the root scores to |x| <= 10.
        We do this because the inverse-laplacian marginal computation can be numerically unstable otherwise. (This
        is in part due to the fact that the model is always trained greedily and we have observed empirically that
        the root scores can be a factor of 100 or more larger in magnitude than the other scores)
        """
        self.dep_label_fine_to_coarse_reduce_matrix = self.dep_label_fine_to_coarse_reduce_matrix.to(tag_logits.device)
        B, N, T = tag_logits.shape

        # Convert tag logits to per-tag dists -- just normalize along the last dimension
        tag_ps = torch.softmax(tag_logits, dim=2)
        tag_ps = mask[:, 1:].unsqueeze(2) * tag_ps

        # Convert label potentials to dist over universal labels and get marginal probs
        root_label_logits = torch.diag_embed(
            label_logits[:, :, 0, 1:]
        )  # class logits for ROOT->j along diag (shape: B, L, N, N)
        label_logits = label_logits[:, :, 1:, 1:]  # drop scores to/from ROOT token
        eye = torch.eye(N).reshape(1, 1, N, N).to(label_logits.device)
        label_logits = (1 - eye) * label_logits + eye * root_label_logits
        label_ps = torch.softmax(label_logits, dim=1).permute(0, 2, 3, 1)  # then convert to probs with labels at -1
        label_ps = label_ps.matmul(self.dep_label_fine_to_coarse_reduce_matrix)  # then sum over coarse subsets

        # Convert arc potentials to a dist and marginal probs
        # Arc potentials are converted to format used by torch-struct
        INF = 1e8
        og_arc_logits = arc_logits.clone()
        mask_adj = mask.unsqueeze(1) * mask.unsqueeze(2)  # mask out all possible edges to/from PAD
        og_mask_adj = mask_adj.clone()
        arc_logits = mask_adj * arc_logits + (1 - mask_adj) * -INF  # set edges touching pads to near -inf
        eye = torch.eye(N).reshape(1, N, N).to(arc_logits.device)

        if safe_arc_ps:
            arc_logits = masked_log_softmax(arc_logits, mask_adj, dim=1)
            root_scores = arc_logits[:, 0, 1:]
        else:
            root_scores = arc_logits[:, 0, 1:]

        root_scores = torch.diag_embed(
            root_scores
        )  # pull root scores to other words and put them on diagonal (shape: B, N, N)
        arc_logits = arc_logits[:, 1:, 1:]
        arc_logits = (1 - eye) * arc_logits + eye * root_scores
        mask_adj = mask_adj[:, 1:, 1:]  # drop root
        arc_logits = (
            mask_adj * arc_logits + (1 - mask_adj) * -INF
        )  # make sure the arc logits are masked approrpriately
        if not safe_arc_ps:
            # Set all pads to exactly -inf
            arc_logits.masked_fill_(arc_logits != 0, float("-inf"))
        lengths = mask_adj[:, 0].sum(dim=-1)
        arc_crf = NonProjectiveDependencyCRF(arc_logits, lengths)

        if self.use_greedy_arc_ps:
            arc_ps = torch.softmax(arc_logits, dim=1)
        else:
            arc_ps = arc_crf.marginals

        if self.detach_arc_ps:
            arc_ps = arc_ps.detach()

        return tag_ps, label_ps, arc_ps, arc_logits, arc_crf, mask_adj, og_arc_logits

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            name: task_metric
            for task in self.tasks
            for name, task_metric in self.decoders[task].get_metrics(reset).items()
        }
        metrics.update({k: m.get_metric(reset) for k, m in self.metrics.items()})

        # The "sum" metric summing all tracked metrics keeps a good measure of patience for early stopping and saving
        metrics_to_track = {
            "upos",
            "coarse_LAS",
        }
        sum_metrics = [
            metric
            for name, metric in metrics.items()
            if not name.startswith("_") and set(name.split("/")).intersection(metrics_to_track)
        ]
        # print("sum_metrics", sum_metrics)
        metrics[".run/.sum"] = sum(sum_metrics)

        return metrics
