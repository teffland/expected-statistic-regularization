from collections import deque, defaultdict
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union, Deque, Set
from copy import copy
from tqdm import tqdm

import torch
from collections import defaultdict

from allennlp.data import TensorDict
from allennlp.nn.util import tiny_value_of_dtype
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.util import get_train_and_validation_metrics, get_batch_size
from allennlp.data.fields.tensor_field import TensorField

from ml.ppt.aatrn import compute_ambiguous_tags_mask, compute_ambiguous_arcs_mask

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


logger = logging.getLogger(__name__)


@TrainerCallback.register("ppt_masker")
class PPTMaskCallback(TrainerCallback):
    """
    Compute the PPT masks for training with PPT loss and add them to the data
    """

    def __init__(self, serialization_dir: str, tree_threshold: float = 0.95, tag_threshold: float = 0.95,) -> None:
        super().__init__(serialization_dir)
        self.tree_threshold = tree_threshold
        self.tag_threshold = tag_threshold

    def on_start(self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs) -> None:
        """Iterate over the data, running the initial model and computing the tree and tag masks."""
        if not is_primary:
            return
        model = trainer._pytorch_model

        # Hack the data loaders to return instances and batches one at a time
        for loader in [trainer.data_loader]:  # , trainer._validation_data_loader]:
            og_batch_size = loader.batch_size
            og_batch_sampler = loader.batch_sampler
            og_shuffle = loader.shuffle
            loader.batch_size = 1
            loader.batch_sampler = None
            loader.shuffle = False

            instances = list(loader.iter_instances())
            batches = loader._instances_to_batches(instances, move_to_device=True)
            print("ON START", flush=True)
            for i, (instance, as_batch) in tqdm(enumerate(zip(instances, batches)), "Computing PPT", len(instances)):
                instance_min_risk = as_batch["metadata"][0]["for_min_risk"]
                as_batch["metadata"][0]["for_min_risk"] = True
                # print("len", len(as_batch["metadata"][0]["words"]))
                output = model(super_only=True, **as_batch)
                # print(i, len(instances), flush=True)
                # print("instance", instance)
                m = instance["metadata"]
                # print(m["for_min_risk"], flush=True)
                # print(m["parse"], flush=True)
                # print("output", output["label_logits"])

                # Compute the tree and tag masks for the ppt loss based on the current model logits
                tag_logits, arc_logits, label_logits, mask = (
                    output["tag_logits"],
                    output["arc_logits"],
                    output["label_logits"],
                    output["mask"],
                )
                instance.add_field("ppt_tag_mask", TensorField(compute_ambiguous_tags_mask(tag_logits.detach())[0]))
                tree_logits, mask_adj = model.convert_logits(label_logits, arc_logits, mask)
                instance.add_field("ppt_tree_mask", TensorField(compute_ambiguous_arcs_mask(tree_logits.detach())[0]))
                as_batch["metadata"][0]["for_min_risk"] = instance_min_risk

            loader.batch_size = og_batch_size
            loader.batch_sampler = og_batch_sampler
            loader.shuffle = og_shuffle
            model.seen_ids = defaultdict(set)
