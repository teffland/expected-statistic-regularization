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
from ml.ppt.aatrn import compute_aatrn_loss

import pickle

logger = logging.getLogger(__name__)


@Model.register("ppt_udify_model", exist_ok=True)
class PPTUdifyModel(UdifyModel):
    """
    Implementation of the Udify model with an additional soft self-training constraint-based loss from
    Kurniawan et al 21 - "PPT: Parsimonious Parser Transferor Unsupervised Cross-Lingual Adaptation".

    For this model, instead of using expected syntax loss on unlabeled instances, we expect there to be a constraint
    mask on acceptable parse configurations, computed by the original model, that we use to optimize the parse tree under.
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
        unsupervised_loss_weight: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(PPTUdifyModel, self).__init__(
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
        self.metrics["_model_time"] = Average()
        self.metrics["supervised_loss"] = Average()
        self.metrics["unsupervised_loss"] = Average()
        self.metrics["ppt_tree_loss"] = Average()
        self.metrics["ppt_tag_loss"] = Average()
        self.unsupervised_loss_weight = unsupervised_loss_weight
        self.seen_ids = defaultdict(set)

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        metadata: List[Dict[str, Any]] = None,
        super_only: bool = False,
        **kwargs: Dict[str, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        """Run the udify model, then do all of the losses."""
        t0 = time()
        for_min_risk = metadata[0].get("for_min_risk", False)
        N = tokens["bert"]["mask"].shape[1]
        # print(f"N: {N}", flush=True)

        for m, d in zip(metadata, kwargs.get("dataset", ["val"] * len(metadata))):
            self.seen_ids[d].add(m["uid"])
        # print("seen ids", {k: len(vs) for k, vs in self.seen_ids.items()}, flush=True)
        # print("for min risk", for_min_risk)

        output_dict = super(PPTUdifyModel, self).forward(tokens=tokens, metadata=metadata, **kwargs)

        if super_only or N > 70:
            if N > 70:
                print(f"too long: {N}", flush=True)
            return output_dict
        self.metrics["_model_time"](time() - t0)
        try:
            for_min_risk = metadata[0].get("for_min_risk", False)
            # print("for_min_risk?", for_min_risk, kwargs.get("dataset", ["no dataset"])[0], flush=True)
            if self.training and for_min_risk and "ppt_tag_mask" in kwargs:
                t0 = time()
                ppt_loss = self.ppt_loss(
                    tag_logits=output_dict["tag_logits"],
                    arc_logits=output_dict["arc_logits"],
                    label_logits=output_dict["label_logits"],
                    mask=output_dict["mask"],
                    ppt_tag_mask=kwargs["ppt_tag_mask"],
                    ppt_tree_mask=kwargs["ppt_tree_mask"],
                )
                output_dict["loss"] = self.unsupervised_loss_weight * ppt_loss
                self.metrics["unsupervised_loss"](self.unsupervised_loss_weight * ppt_loss.detach().cpu().numpy())
            else:
                self.metrics["supervised_loss"](output_dict["loss"].detach().cpu().numpy())
        except Exception as e:
            with open("failed_data.pkl", "wb") as f:
                output_dict["tokens"] = tokens
                output_dict["metadata"] = metadata
                pickle.dump(output_dict, f)
            raise e

        return output_dict

    def ppt_loss(self, tag_logits, arc_logits, label_logits, mask, ppt_tag_mask, ppt_tree_mask):
        """Calculate the PPT self-training loss from Kurniawan et al. 2021, extended to include POS."""
        # Convert UDify outputs to format used by Kurniawan's loss code
        tree_logits, mask_adj = self.convert_logits(label_logits, arc_logits, mask)
        mask = mask[:, 1:].float()

        # Apply their loss
        tree_loss = compute_aatrn_loss(tree_logits, ppt_tree_mask, mask=mask.bool())
        self.metrics["ppt_tree_loss"](tree_loss.detach().cpu().numpy())

        # Also apply their loss to the POS tags, which is additional
        tag_Zs = mask * torch.logsumexp(tag_logits, dim=-1)
        tag_masked_Zs = mask * torch.logsumexp(tag_logits.masked_fill(~ppt_tag_mask, -1e9), dim=-1)
        tag_loss = tag_Zs.sum() - tag_masked_Zs.sum()
        self.metrics["ppt_tag_loss"](tag_loss.detach().cpu().numpy())

        loss = tree_loss + tag_loss
        return loss

    def convert_logits(self, label_logits, arc_logits, mask, safe_arc_ps=True):
        """Convert the logits from udify decoders to adjacency representation used by Kurniawan loss.

        The main thing here is to convert the label and arc logits from an (N+1)x(N+1) format where the ROOT symbol
        is the first token to an NxN format with the root scores on the diagonals that is preferred by torch_struct,
        which we use for getting marginal arc probs.

        The other thing is that we convert the label probs over fine-grained dependencies to those over coarse-grained
        dependencies since the fine-grained ones are specific to each language so for oov langs it's hard to transfer.
        We then add these to the arc potentials for a single B,N,N,L representation used by their code.

        The dimensional conversions are:
          * labels: (B, L_fine, N+1 (head), N+1 (tail)) -> (B, N (head), N (tail), L_coarse)
          * arcs: (B, N+1 (head), N+1 (child)) -> (B, N (head), N (child), L_coarse)

        If safe_arc_ps = True, then we renormalize the arc logits per child and clamp the root scores to |x| <= 10.
        We do this because the inverse-laplacian marginal computation can be numerically unstable otherwise. (This
        is in part due to the fact that the model is always trained greedily and we have observed empirically that
        the root scores can be a factor of 100 or more larger in magnitude than the other scores)
        """
        # Convert label potentials to log-normalized potentials over coarse labels
        self.dep_label_fine_to_coarse_reduce_matrix = self.dep_label_fine_to_coarse_reduce_matrix.to(
            label_logits.device
        )
        mask_adj = mask.unsqueeze(1) * mask.unsqueeze(2)  # mask out all possible edges to/from PAD

        # print("label_logits", label_logits.shape)
        # print("arc_logits", arc_logits.shape)
        root_label_logits = torch.diag_embed(
            label_logits[:, :, 0, 1:]
        )  # class logits for ROOT->j along diag (shape: B, L, N, N)
        label_logits = label_logits[:, :, 1:, 1:]  # drop scores to/from ROOT token
        B, L, _, N = label_logits.shape
        eye = torch.eye(N).reshape(1, 1, N, N).to(label_logits.device)
        label_logits = (1 - eye) * label_logits + eye * root_label_logits
        label_logits = masked_log_softmax(label_logits, mask_adj[:, 1:, 1:], dim=1)
        label_logits = label_logits.permute(0, 2, 3, 1).unsqueeze(
            -1
        )  # put label dim on the end and add broadcast coarse dim
        M = self.dep_label_fine_to_coarse_reduce_matrix.log().reshape(
            1, 1, 1, *self.dep_label_fine_to_coarse_reduce_matrix.shape
        )
        label_logits = torch.logsumexp(label_logits + M, dim=-2)  # now B, N, N, L_coarse in normalized logspace

        # Convert arc potentials to a dist and marginal probs
        # Arc potentials are converted to format used by torch-struct
        INF = 1e8
        og_arc_logits = arc_logits.clone()
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

        # Now add label logits as an extra dim to the
        tree_logits = arc_logits.unsqueeze(3) + label_logits
        return tree_logits, mask_adj

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
