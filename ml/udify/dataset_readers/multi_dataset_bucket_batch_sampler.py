import logging
import math
from typing import List, Iterable, Tuple, Sequence, Optional, Dict
import random
from collections import defaultdict
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers.batch_sampler import BatchSampler
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler


@BatchSampler.register("multi_dataset_bucket")
class MultiDatasetBucketBatchSampler(BucketBatchSampler):
    def __init__(self, sample_proportions: Dict[str, float], **kwargs):
        print("kwargs", kwargs)
        super(MultiDatasetBucketBatchSampler, self).__init__(**kwargs)
        self.sample_proportions = sample_proportions

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        """ Iterate over dataset instances with an upsampling scheme for smaller ones based on proportions.
        """
        # Group instances by dataset
        dataset_instances = defaultdict(list)
        for instance in instances:
            d = instance.fields["dataset"].metadata
            # print(d)
            dataset_instances[d].append(instance)

        def refresh_batches(dataset):
            instances = dataset_instances[dataset]
            indices, _ = self._argsort_by_padding(instances)
            batches = []
            for group in lazy_groups_of(indices, self.batch_size):
                batch_indices = list(group)
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue

                batches.append(batch_indices)
            if self.shuffle:
                random.shuffle(batches)
            return iter(batches)

        dataset_batches = {dataset: refresh_batches(dataset) for dataset in dataset_instances}
        longest_dataset = sorted([(k, len(vs)) for k, vs in dataset_instances.items()], key=lambda x: -x[1])[0][0]
        print(
            f"dataset batch lens: {sorted([(k, len(vs)) for k, vs in dataset_instances.items()], key=lambda x: -x[1])}"
        )

        # Decide how many of each we need per cycle
        keys = list(dataset_instances.keys())
        ps = np.array([self.sample_proportions[k] for k in keys])
        nums = ps / ps.min()
        # print("Choice nums", nums, flush=True)
        choices = [key for i, key in enumerate(keys) for _ in range(int(nums[i]))]
        random.shuffle(choices)

        # Iterate until the largest finishes, refreshing smaller datasets as needed
        stop = False
        while not stop:
            # print("Choices", choices, flush=True)
            for dataset in choices:
                # print("choice", dataset, flush=True)
                try:
                    batch = next(dataset_batches[dataset])
                    yield batch
                except StopIteration:
                    if dataset == longest_dataset:
                        stop = True
                    else:
                        print("Resetting batches for", dataset, flush=True)
                        dataset_batches[dataset] = refresh_batches(dataset)
                        batch = next(dataset_batches[dataset])
                        yield batch
            random.shuffle(choices)

