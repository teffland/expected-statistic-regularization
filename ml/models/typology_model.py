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
from allennlp.modules.feedforward import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_log_softmax
from allennlp.training.metrics import Average
from allennlp.nn import Activation
from torch_struct import NonProjectiveDependencyCRF

from ml.udify.modules.scalar_mix import ScalarMixWithDropout
from ml.udify.models.udify_model import UdifyModel
from ml.utils.stat_funcs import stat_func_by_name, stat_func_arglists, parse_argstr
from scipy.spatial.distance import jensenshannon as jsd
import pickle

logger = logging.getLogger(__name__)


@Model.register("typology_model", exist_ok=True)
class TypologyModel(UdifyModel):
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
        mlp_dropout: float = 0.25,
        mlp_hidden_dim: int = 1024,
        post_encoder_embedder: TextFieldEmbedder = None,
        dropout: float = 0.0,
        word_dropout: float = 0.0,
        mix_embedding: int = None,
        layer_dropout: float = 0.0,
        loss_cfg: Dict[str, Any] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(TypologyModel, self).__init__(
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

        if loss_cfg:
            self._setup_losses(loss_cfg)
        stat_dim = sum(len(loss["args"]) for loss in self.losses.values())
        print("TOTAL STAT DIM:", stat_dim)
        self.mlp = FeedForward(
            text_field_embedder.get_output_dim(),
            2,
            [mlp_hidden_dim, stat_dim],
            [Activation.by_name("mish")(), Activation.by_name("linear")()],
            [mlp_dropout, 0.0],
        )
        # self.metrics["_model_time"] = Average()

    def _setup_losses(self, cfg):
        """Take superivision specs from file and create supervision loss tensors."""
        print("Setting up losses", flush=True)
        assert cfg is not None
        self.stats_df = df = pd.read_csv(cfg["stats_csv_path"])  # stats_csv_path)
        df["id"] = df["stat_func"] + "-" + df["arg"]
        df = df.set_index("id")
        self.losses = dict()
        self.metrics["total_jsd"] = Average()
        for stat_func, loss_spec in cfg["losses"].items():
            if loss_spec.get("weight", 1.0) == 0.0:
                continue
            print(f"{stat_func} spec: {loss_spec}", flush=True)
            func_domain = stat_func_arglists[stat_func]
            loss_args = defaultdict(list)

            # Maybe restrict the arglist if we pass top or worst k > 0
            if loss_spec.get("topk", 0) + loss_spec.get("worstk", 0):
                topks, worstks = [], []
                topk, worstk = loss_spec.get("topk", 0), loss_spec.get("worstk", 0)
                func_df = df[df["stat_func"] == stat_func]
                n = len(func_df)
                if topk:
                    topk = int(topk * n) if type(topk) == float else topk
                    top_df = func_df.sort_values("mean", ascending=False)  # largest mass args
                    for i in range(min(topk, len(top_df))):
                        topks.append(top_df.iloc[i]["arg"])
                        print(
                            f'Func: {stat_func} top {i} / {topk} arg: {topks[-1]} has mean: {top_df.iloc[i]["mean"]:2.6f}',
                            flush=True,
                        )
                if worstk:
                    worstk = int(worstk * n) if type(worstk) == float else worstk
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
                arglist = topks + worstks
            else:
                arglist = loss_spec.get("arglist", func_domain)

            print(f"Func: {stat_func} has {len(arglist)} / {len(func_domain)} final args")

            args, targets, margins = [], [], []
            self.metrics[f"_{stat_func}_loss"] = Average()
            self.metrics[f"_{stat_func}_jsd"] = Average()
            for argstr in tqdm(arglist, f"Setting up args losses for {stat_func}", len(arglist)):
                args.append(argstr)

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

                targets.append(target)
                margins.append(margin)

            if "|" in stat_func:
                stat_kwargs = {
                    k: (v, v)
                    for k, v in loss_spec.items()
                    if k not in ("type", "weight", "target", "margin", "arglist", "topk", "worstk")
                }
            else:
                # Gives tuple of index tensors per dimension for selecting elements from event tensors
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
                args=args,
                # targets=targets,
                # margins=margins,
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
        # t0 = time()
        # print("s", tokens["bert"]["bert"].shape, flush=True)
        B, N = tokens["bert"]["bert"].shape
        if N > 512:
            print("Input too long: skipping")
            print("=========\n" * 5)
            output_dict = dict(loss=torch.tensor(0.0, requires_grad=True).to(tokens["bert"]["bert"].device))
            return output_dict
        output_dict = super(TypologyModel, self).forward(tokens=tokens, metadata=metadata, **kwargs)
        rep = output_dict["encodings"][:, 0]
        scores = self.mlp(rep).sum(0)
        output_dict["scores"] = scores
        output_dict["parses"] = [m["parse"] for m in metadata]

        # self.metrics["_model_time"](time() - t0)
        try:
            if self.losses:
                # t0 = time()
                loss = self.loss(
                    **output_dict,
                )
                # self.metrics["_total_loss_time"](time() - t0)
                output_dict["loss"] = loss
        except Exception as e:
            with open("failed_data.pkl", "wb") as f:
                output_dict["tokens"] = tokens
                output_dict["metadata"] = metadata
                pickle.dump(output_dict, f)
            raise e

        return output_dict

    def loss(self, scores, parses, **kwargs):
        """Calculate all of the specified losses."""
        total_loss = 0.0
        num_losses = 0
        all_losses = []
        all_jsds = []
        s = 0

        for stat_func_name, specs in self.losses.items():
            w = specs.get("weight", 1.0)
            if w == 0.0:
                continue
            # t0 = time()
            if "|" in stat_func_name:
                counts, N = (None, None), (None, None)
            else:
                counts, N = None, None
            targets = []
            for arg in specs["args"]:
                stat, counts, N = stat_func_by_name(
                    parses, stat_func_name, arg, return_counts=True, counts=counts, N=N
                )
                targets.append(stat)
            e = s + len(targets)
            stats = scores[s:e]
            s = e
            if specs.get("normalize", True):
                stats = torch.softmax(stats, 0)
            targets = torch.FloatTensor(targets).to(scores.device)
            # print(torch.stack([targets, stats], dim=1))

            losses = self.loss_func(specs["type"], stats, targets)
            if specs.get("mean_after_loss", False):
                losses = losses.mean(dim=0)  # losses were left unaggregated over batch dimension
            loss = abs(w) * losses.sum()
            if w > 0.0:
                # Positive weight incorporates into loss, but a negative weight will just log it
                # if torch.isnan(loss):
                #     print(f"WARNING: Got nan loss for {stat_func_name} -- skipping.", flush=True)
                #     print(stats.shape, torch.isnan(stats).any(), torch.isnan(targets), stats)
                #     continue
                total_loss += loss
                num_losses += w
                all_losses.append((stat_func_name, loss))

            div = jsd(targets.detach().cpu().numpy(), stats.detach().cpu().numpy())
            all_jsds.append(div)
            self.metrics[f"_{stat_func_name}_jsd"](div)
            self.metrics[f"_{stat_func_name}_loss"](loss.detach().cpu().numpy())
        total_jsd = np.mean(all_jsds)
        self.metrics["total_jsd"](total_jsd)

        loss = total_loss / num_losses
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
