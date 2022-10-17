"""
The base UDify model for training and prediction

ADAPT THE UDIFY MODEL TO PERFORM THE EXTRA LOSS CALC
"""

from typing import Optional, Any, Dict, List, Tuple
from overrides import overrides
import logging
import os
import torch
import json

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Average

from ml.udify.modules.scalar_mix import ScalarMixWithDropout

logger = logging.getLogger(__name__)


@Model.register("udify_model", exist_ok=True)
class UdifyModel(Model):
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
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(UdifyModel, self).__init__(vocab, regularizer)

        self.tasks = sorted(tasks)
        self.vocab = vocab
        print(os.getcwd())
        self.bert_vocab = BertTokenizer.from_pretrained(
            "/home/ubuntu/syntax//udify/config/archive/bert-base-multilingual-cased/vocab.txt"
        ).vocab
        self.text_field_embedder = text_field_embedder
        self.post_encoder_embedder = post_encoder_embedder
        self.shared_encoder = encoder
        self.word_dropout = word_dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        self.decoders = torch.nn.ModuleDict(decoders)

        if mix_embedding:
            self.scalar_mix = torch.nn.ModuleDict(
                {
                    task: ScalarMixWithDropout(mix_embedding, do_layer_norm=False, dropout=layer_dropout)
                    for task in self.decoders
                }
            )
        else:
            self.scalar_mix = None

        self.metrics = {}

        for task in self.tasks:
            if task not in self.decoders:
                raise ConfigurationError(f"Task {task} has no corresponding decoder. Make sure their names match.")

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        if initializer:
            initializer(self)
        self._count_params()

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        metadata: List[Dict[str, Any]] = None,
        **kwargs: Dict[str, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        # print("\nkwargs", kwargs.keys())
        # epoch_num = kwargs.pop("epoch_num", [1])[0]
        # print("Epoch", epoch_num)

        gold_tags = kwargs

        if "tokens" in self.tasks:
            # Model is predicting tokens, so add them to the gold tags
            gold_tags["tokens"] = tokens["tokens"]

        tokens = tokens["bert"]  # hack to compensate for diff between v0.9 and v2.6 token indexer outputs
        mask = tokens["mask"]
        # print("mask shape", mask.shape, flush=True)
        # print("tokens", {(k, v.shape) for k, v in tokens.items()}, "mask", mask.shape)
        self._apply_token_dropout(tokens)

        embedded_text_input = self.text_field_embedder(tokens)

        if self.post_encoder_embedder:
            post_embeddings = self.post_encoder_embedder(tokens)

        encoded_text = self.shared_encoder(embedded_text_input, mask)

        logits = {}
        class_probabilities = {}
        output_dict = {"logits": logits, "class_probabilities": class_probabilities, "encodings": encoded_text}
        loss = 0

        # Run through each of the tasks on the shared encoder and save predictions
        for task in self.tasks:
            if self.scalar_mix:
                decoder_input = self.scalar_mix[task](encoded_text, mask)
            else:
                decoder_input = encoded_text

            if self.post_encoder_embedder:
                decoder_input = decoder_input + post_embeddings

            if task == "deps":
                tag_logits = logits["upos"] if "upos" in logits else None
                pred_output = self.decoders[task](
                    decoder_input,
                    mask,
                    tag_logits,
                    gold_tags.get("head_tags", None),
                    gold_tags.get("head_indices", None),
                    metadata,
                )
                for key in [
                    "heads",
                    "head_tags",
                    "arc_loss",
                    "tag_loss",
                    "mask",
                    "head_arc_logits",
                    "head_label_logits",
                    "arc_logits",
                    "label_logits",
                ]:
                    output_dict[key] = pred_output[key]
            else:
                pred_output = self.decoders[task](decoder_input, mask, gold_tags, metadata)
                if task == "upos":
                    output_dict["tag_logits"] = pred_output["logits"]
                logits[task] = pred_output["logits"]
                class_probabilities[task] = pred_output["class_probabilities"]

            if task in gold_tags or task == "deps" and "head_tags" in gold_tags:
                # Keep track of the loss if we have the gold tags available
                loss = loss + pred_output["loss"]

        output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
            output_dict["ids"] = [x["ids"] for x in metadata if "ids" in x]
            output_dict["multiword_ids"] = [x["multiword_ids"] for x in metadata if "multiword_ids" in x]
            output_dict["multiword_forms"] = [x["multiword_forms"] for x in metadata if "multiword_forms" in x]

        output_dict.update(self.decode(output_dict))

        return output_dict

    def _apply_token_dropout(self, tokens):
        # Word dropout
        if "tokens" in tokens:
            oov_token = self.vocab.get_token_index(self.vocab._oov_token)
            ignore_tokens = [self.vocab.get_token_index(self.vocab._padding_token)]
            tokens["tokens"] = self.token_dropout(
                tokens["tokens"],
                oov_token=oov_token,
                padding_tokens=ignore_tokens,
                p=self.word_dropout,
                training=self.training,
            )

        # BERT token dropout
        if "bert" in tokens:
            oov_token = self.bert_vocab["[MASK]"]
            ignore_tokens = [self.bert_vocab["[PAD]"], self.bert_vocab["[CLS]"], self.bert_vocab["[SEP]"]]
            tokens["bert"] = self.token_dropout(
                tokens["bert"],
                oov_token=oov_token,
                padding_tokens=ignore_tokens,
                p=self.word_dropout,
                training=self.training,
            )

    @staticmethod
    def token_dropout(
        tokens: torch.LongTensor, oov_token: int, padding_tokens: List[int], p: float = 0.2, training: float = True
    ) -> torch.LongTensor:
        """
        During training, randomly replaces some of the non-padding tokens to a mask token with probability ``p``

        :param tokens: The current batch of padded sentences with word ids
        :param oov_token: The mask token
        :param padding_tokens: The tokens for padding the input batch
        :param p: The probability a word gets mapped to the unknown token
        :param training: Applies the dropout if set to ``True``
        :return: A copy of the input batch with token dropout applied
        """
        if training and p > 0:
            # Ensure that the tensors run on the same device
            device = tokens.device

            # This creates a mask that only considers unpadded tokens for mapping to oov
            padding_mask = torch.ones(tokens.size(), dtype=torch.bool).to(device)
            for pad in padding_tokens:
                padding_mask &= tokens != pad

            # Create a uniformly random mask selecting either the original words or OOV tokens
            dropout_mask = (torch.empty(tokens.size()).uniform_() < p).to(device)
            oov_mask = dropout_mask & padding_mask

            oov_fill = torch.empty(tokens.size(), dtype=torch.long).fill_(oov_token).to(device)

            result = torch.where(oov_mask, oov_fill, tokens)

            return result
        else:
            return tokens

    # @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for task in self.tasks:
            self.decoders[task].decode(output_dict)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            name: task_metric
            for task in self.tasks
            for name, task_metric in self.decoders[task].get_metrics(reset).items()
        }
        metrics.update({k: m.get_metric(reset) for k, m in self.metrics.items()})

        # The "sum" metric summing all tracked metrics keeps a good measure of patience for early stopping and saving
        metrics_to_track = {"upos", "xpos", "feats", "lemmas", "LAS", "UAS"}
        metrics[".run/.sum"] = sum(
            metric
            for name, metric in metrics.items()
            if not name.startswith("_") and set(name.split("/")).intersection(metrics_to_track)
        )

        return metrics

    def _count_params(self):
        self.total_params = sum(p.numel() for p in self.parameters())
        self.total_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Total number of parameters: {self.total_params}")
        logger.info(f"Total number of trainable parameters: {self.total_train_params}")
