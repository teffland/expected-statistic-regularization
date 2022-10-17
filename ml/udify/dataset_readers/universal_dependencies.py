"""
A Dataset Reader for Universal Dependencies, with support for multiword tokens and special handling for NULL "_" tokens
"""

from typing import *
from os import PathLike

PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]

from overrides import overrides
from ml.udify.dataset_readers.parser import parse_line, DEFAULT_FIELDS, process_multiword_tokens
from ml.utils.udp_dataset import conllu_to_spacy
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

# from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, WordSplitter
from allennlp.data.tokenizers import Token

from ml.udify.dataset_readers.lemma_edit import gen_lemma_rule
from ml.utils.stat_funcs import dep_fine_tags, clean_dep

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("udify_universal_dependencies")
class UniversalDependenciesDatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        no_labels: bool = False,
        for_min_risk: bool = False,
        skip_short: bool = False,
        limit: int = None,
        max_len: int = None,
    ) -> None:
        super().__init__(lazy)
        self.no_labels = no_labels
        self.for_min_risk = for_min_risk
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.skip_short = skip_short
        self.limit = limit
        self.max_len = max_len

    @overrides
    def read(self, file_path: DatasetReaderInput) -> Iterator[Instance]:
        """
        Returns an iterator of instances that can be read from the file path.
        """
        for instance in tqdm(self._read(file_path), "Reading instances"):  # type: ignore
            if self._worker_info is None:
                # If not running in a subprocess, it's safe to apply the token_indexers right away.
                self.apply_token_indexers(instance)
            yield instance

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        print("Reading UD instances from conllu dataset at:", file_path, flush=True)
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            print("Reading UD instances from conllu dataset at:", file_path, flush=True)
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for i, annotation in enumerate(parse_incr(conllu_file)):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and we replace these word ids with None in process_multiword_tokens.
                annotation = process_multiword_tokens(annotation)
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
                annotation = [x for x in annotation if x["id"] is not None]

                parse = conllu_to_spacy(annotation)

                if len(annotation) == 0:
                    continue

                if self.limit and i >= self.limit:
                    print(f"Reached limit of {self.limit}, breaking.")
                    break

                def get_field(tag: str, map_fn: Callable[[Any], Any] = None) -> List[Any]:
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]

                # Extract multiword token rows (not used for prediction, purely for evaluation)
                ids = [x["id"] for x in annotation]
                multiword_ids = [x["multi_id"] for x in multiword_tokens]
                multiword_forms = [x["form"] for x in multiword_tokens]

                words = get_field("form")
                lemmas = get_field("lemma")
                lemma_rules = [
                    gen_lemma_rule(word, lemma) if lemma != "_" else "_" for word, lemma in zip(words, lemmas)
                ]
                upos_tags = get_field("upostag")
                xpos_tags = None  # get_field("xpostag")
                feats = None  # get_field(
                #     "feats", lambda x: "|".join(k + "=" + v for k, v in x.items()) if hasattr(x, "items") else "_"
                # )
                heads = get_field("head")
                dep_rels = get_field("deprel")
                dependencies = list(zip(dep_rels, heads))

                if self.skip_short and len(words) <= 1:
                    print("Skipping single-word sentence")
                    continue

                yield self.text_to_instance(
                    words=words,
                    lemmas=lemmas,
                    lemma_rules=lemma_rules,
                    upos_tags=upos_tags,
                    xpos_tags=None,
                    feats=None,
                    dependencies=dependencies,
                    ids=ids,
                    multiword_ids=multiword_ids,
                    multiword_forms=multiword_forms,
                    metadata=dict(parse=parse, uid=f"{file_path.split('/')[-1]}-S{i}"),
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        lemmas: List[str] = None,
        lemma_rules: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        feats: List[str] = None,
        dependencies: List[Tuple[str, int]] = None,
        ids: List[str] = None,
        multiword_ids: List[str] = None,
        multiword_forms: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words])
        fields["tokens"] = tokens

        names = ["upos"]  # , "xpos", "feats", "lemmas"]
        all_tags = [upos_tags, xpos_tags, feats, lemma_rules]
        if not self.no_labels:
            for name, field in zip(names, all_tags):
                if field:
                    fields[name] = SequenceLabelField(field, tokens, label_namespace=name)

        if dependencies is not None and not self.no_labels:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            def map_dep(dep):
                """Some of the datasets define fine-grained deps that aren't in official UD (but the coarse are).
                Since we don't care about fine-grained anyways, just map them to the coarse tag.
                """
                if dep not in dep_fine_tags:
                    dep = clean_dep(dep)
                return dep

            fields["head_tags"] = SequenceLabelField(
                [map_dep(x[0]) for x in dependencies], tokens, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [int(x[1]) for x in dependencies], tokens, label_namespace="head_index_tags"
            )
        # print("fields", fields)

        metadata = metadata or dict()
        fields["metadata"] = MetadataField(
            {
                "words": words,
                "upos_tags": upos_tags,
                "xpos_tags": xpos_tags,
                "feats": feats,
                "lemmas": lemmas,
                "lemma_rules": lemma_rules,
                "ids": ids,
                "multiword_ids": multiword_ids,
                "multiword_forms": multiword_forms,
                "for_min_risk": self.for_min_risk,
                **metadata,
            }
        )

        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance["tokens"].token_indexers = self._token_indexers


@DatasetReader.register("udify_universal_dependencies_raw")
class UniversalDependenciesRawDatasetReader(DatasetReader):
    """Like UniversalDependenciesDatasetReader, but reads raw sentences and tokenizes them first."""

    def __init__(self, dataset_reader: DatasetReader, tokenizer: Tokenizer = None) -> None:
        super().__init__(lazy=dataset_reader.lazy)
        self.dataset_reader = dataset_reader
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SpacyTokenizer(language="xx_ent_wiki_sm")

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            for sentence in conllu_file:
                print("sent")
                if sentence:
                    words = [word.text for word in self.tokenizer.tokenize(sentence)]
                    yield self.text_to_instance(words)

    @overrides
    def text_to_instance(self, words: List[str]) -> Instance:
        return self.dataset_reader.text_to_instance(words)
