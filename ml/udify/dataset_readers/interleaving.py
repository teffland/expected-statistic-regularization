from typing import Dict, Mapping, Iterable, Union, Optional
import json

from overrides import overrides
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import (
    DatasetReader,
    PathOrStr,
    WorkerInfo,
    DistributedInfo,
)
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance

_VALID_SCHEMES = {"round_robin", "all_at_once", "sample"}


@DatasetReader.register("interleaving", exist_ok=True)
class InterleavingDatasetReader(DatasetReader):
    """
    A `DatasetReader` that wraps multiple other dataset readers,
    and interleaves their instances, adding a `MetadataField` to
    indicate the provenance of each instance.

    Unlike most of our other dataset readers, here the `file_path` passed into
    `read()` should be a JSON-serialized dictionary with one file_path
    per wrapped dataset reader (and with corresponding keys).

    Registered as a `DatasetReader` with name "interleaving".

    # Parameters

    readers : `Dict[str, DatasetReader]`
        The dataset readers to wrap. The keys of this dictionary will be used
        as the values in the MetadataField indicating provenance.
    dataset_field_name : `str`, optional (default = `"dataset"`)
        The name of the MetadataField indicating which dataset an instance came from.
    scheme : `str`, optional (default = `"round_robin"`)
        Indicates how to interleave instances. Currently the two options are "round_robin",
        which repeatedly cycles through the datasets grabbing one instance from each;
        and "all_at_once", which yields all the instances from the first dataset,
        then all the instances from the second dataset, and so on. You could imagine also
        implementing some sort of over- or under-sampling, although hasn't been done.
    sample_proportions: `Dict[str, float]`
        If scheme = 'sample', the sampling rate for each dataset.
    """

    def __init__(
        self,
        readers: Dict[str, DatasetReader],
        dataset_field_name: str = "dataset",
        scheme: str = "sample",
        sample_proportions: Dict[str, float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._readers = readers
        self._dataset_field_name = dataset_field_name
        self.sample_proportions = sample_proportions

        if scheme not in _VALID_SCHEMES:
            raise ConfigurationError(f"invalid scheme: {scheme}")
        self._scheme = scheme

    @overrides
    def _set_worker_info(self, info: Optional[WorkerInfo]) -> None:
        super()._set_worker_info(info)
        for reader in self._readers.values():
            reader._set_worker_info(info)

    @overrides
    def _set_distributed_info(self, info: Optional[DistributedInfo]) -> None:
        super()._set_distributed_info(info)
        for reader in self._readers.values():
            reader._set_distributed_info(info)

    def _read_round_robin(self, datasets: Mapping[str, Iterable[Instance]]) -> Iterable[Instance]:
        remaining = set(datasets)
        dataset_iterators = {key: iter(dataset) for key, dataset in datasets.items()}

        while remaining:
            for key, dataset in dataset_iterators.items():
                if key in remaining:
                    try:
                        instance = next(dataset)
                        instance.fields[self._dataset_field_name] = MetadataField(key)
                        yield instance
                    except StopIteration:
                        remaining.remove(key)

    def _read_sample(self, datasets: Mapping[str, Iterable[Instance]]) -> Iterable[Instance]:
        dataset_lens = {key: len(list(dataset)) for key, dataset in datasets.items()}
        dataset_iterators = {key: iter(dataset) for key, dataset in datasets.items()}
        keys = list(dataset_iterators.keys())
        ps = np.array([self.sample_proportions[k] for k in keys])
        nums = ps / ps.min()
        choices = [key for i, key in enumerate(keys) for _ in range(int(nums[i]))]

        stop = False
        while not stop:
            np.random.shuffle(choices)
            # print("Choices:", choices, flush=True)
            for choice in choices:
                dataset = dataset_iterators[choice]
                try:
                    instance = next(dataset)
                    instance.fields[self._dataset_field_name] = MetadataField(choice)
                    yield instance
                except StopIteration:
                    stop = True
                    continue

    def _read_all_at_once(self, datasets: Mapping[str, Iterable[Instance]]) -> Iterable[Instance]:
        for key, dataset in datasets.items():
            for instance in dataset:
                instance.fields[self._dataset_field_name] = MetadataField(key)
                yield instance

    @overrides
    def _read(self, file_path: Union[str, Dict[str, PathOrStr]]) -> Iterable[Instance]:
        if isinstance(file_path, str):
            try:
                file_paths = json.loads(file_path)
            except json.JSONDecodeError:
                raise ConfigurationError(
                    "the file_path for the InterleavingDatasetReader "
                    "needs to be a JSON-serialized dictionary {reader_name -> file_path}"
                )
        else:
            file_paths = file_path

        if file_paths.keys() != self._readers.keys():
            raise ConfigurationError("mismatched keys")

        # Load datasets
        datasets = {key: reader.read(file_paths[key]) for key, reader in self._readers.items()}
        # print("DATASET LENS: ", {k: len(list(insts)) for k, insts in datasets.items()})
        # quit()

        if self._scheme == "round_robin":
            yield from self._read_round_robin(datasets)
        elif self._scheme == "all_at_once":
            yield from self._read_all_at_once(datasets)
        elif self._scheme == "sample":
            yield from self._read_sample(datasets)
        else:
            raise RuntimeError("impossible to get here")

    @overrides
    def text_to_instance(self, dataset_key: str, *args, **kwargs) -> Instance:  # type: ignore
        return self._readers[dataset_key].text_to_instance(*args, **kwargs)  # type: ignore[call-arg]

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        dataset = instance.fields[self._dataset_field_name].metadata  # type: ignore[attr-defined]
        self._readers[dataset].apply_token_indexers(instance)
