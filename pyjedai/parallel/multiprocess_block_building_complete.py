import math
import re

import nltk
from mpire import WorkerPool

from pyjedai.datamodel import Block
from pyjedai.parallel.util import merge_dicts, batchify


class SharedData:
    def __init__(self, data, ids):
        self.data = data
        self.ids = ids

class MultiprocessBlockBuildingComplete:
    def __init__(
            self, data, attributes_1: list = None, attributes_2: list = None,
            block_class_name = None, tokenize_method_params:dict = None, n_processes: int = 1
    ) -> any:
        self.n_processes = n_processes
        self.ids = data.get_ids()
        self.data = data
        self.attributes_1 = attributes_1
        self.attributes_2 = attributes_2
        self.block_class_name = block_class_name
        self.resolve_tokenize_method()
        self.tokenize_method_params = tokenize_method_params
        self.blocks = dict()
        self.parameters = []
        self.shared_data = SharedData(data, self.ids)
        self.pool = WorkerPool(n_jobs=n_processes, shared_objects=self.shared_data, start_method='fork')
        self.generate_process_parameters()

    def resolve_tokenize_method(self):
        if self.block_class_name is None or self.block_class_name == "StandardBlocking":
            self.tokenize_method = standard_blocking_tokenize_entity
        elif self.block_class_name == "QGramsBlocking":
            self.tokenize_method = qgrams_tokenize_entity
        elif self.block_class_name == "SuffixArraysBlocking":
            self.tokenize_method = suffix_arrays_tokenize_entity
        elif self.block_class_name == "ExtendedSuffixArraysBlocking":
            self.tokenize_method = extended_suffix_arrays_tokenize_entity
        elif self.block_class_name == "ExtendedQGramsBlocking":
            self.tokenize_method = extended_qgrams_tokenize_entity

    def generate_process_parameters(self):
        self.chunked_data = self.data.split(self.n_processes)

        batched_indices_1 = batchify(self.ids[0], self.n_processes)
        batched_indices_2 = batchify(self.ids[1], self.n_processes)

        parameters = dict()
        for indices_1, indices_2 in zip(batched_indices_1, batched_indices_2):
            parameters["ids_1_indices"] = indices_1
            parameters["ids_2_indices"] = indices_2
            parameters["attributes_1"] = self.attributes_1
            parameters["attributes_2"] = self.attributes_2
            parameters["last_id"] = self.ids[0].size
            parameters["tokenize_method"] = self.tokenize_method
            parameters["tokenize_method_params"] = self.tokenize_method_params
            self.parameters.append(parameters.copy())
    def run(self):
        for res in self.pool.imap_unordered(build_blocks, self.parameters):
            merge_dicts(self.blocks, res)


    def get_blocks(self):
        return self.blocks


def build_blocks(process_data, ids_1_indices, ids_2_indices, last_id,
                 attributes_1, attributes_2, tokenize_method, tokenize_method_params):

    data = process_data.data
    ids_1, ids_2 = process_data.ids

    _entities_d1 = data.dataset_1[attributes_1 if attributes_1 else data.attributes_1] \
        .apply(" ".join, axis=1) \
        .apply(tokenize_method) \
        .values.tolist()

    if not data.is_dirty_er:
        _entities_d2 = data.dataset_2[attributes_2 if attributes_2 else data.attributes_2] \
            .apply(" ".join, axis=1) \
            .apply(tokenize_method) \
            .values.tolist()


    blocks = {}

    partial_ids_1 = ids_1[ids_1_indices[0]:ids_2_indices[1]]
    for eid, entity in zip(partial_ids_1, _entities_d1):
        for token in entity:
            blocks.setdefault(token, Block())
            blocks[token].entities_D1.add(eid)

    if not data.is_dirty_er:
        partial_ids_2 = ids_2[ids_2_indices[0]:ids_2_indices[1]]
        for eid, entity in zip(partial_ids_2, _entities_d2):
            for token in entity:
                eid = eid + last_id
                blocks.setdefault(token, Block())
                blocks[token].entities_D2.add(eid)

    return blocks


def standard_blocking_tokenize_entity(entity: str) -> list:
    """Produces a list of workds of a given string

    Args:
        entity (str): String representation  of an entity

    Returns:
        list: List of words
    """
    return list(set(filter(None, re.split('[\\W_]', entity.lower()))))


def qgrams_tokenize_entity(entity, qgrams) -> set:
    keys = set()
    for token in standard_blocking_tokenize_entity(entity):
        if len(token) < qgrams:
            keys.add(token)
        else:
            keys.update(''.join(qg) for qg in nltk.ngrams(token, n=qgrams))
    return keys


def suffix_arrays_tokenize_entity(entity, suffix_length) -> set:
    keys = set()
    for token in standard_blocking_tokenize_entity(entity):
        if len(token) < suffix_length:
            keys.add(token)
        else:
            for length in range(0, len(token) - suffix_length + 1):
                keys.add(token[length:])
    return keys


def extended_suffix_arrays_tokenize_entity(entity, suffix_length) -> set:
    keys = set()
    for token in standard_blocking_tokenize_entity(entity):
        keys.add(token)
        if len(token) > suffix_length:
            for current_size in range(suffix_length, len(token)):
                for letters in list(nltk.ngrams(token, n=current_size)):
                    keys.add("".join(letters))
    return keys


def extended_qgrams_tokenize_entity(qgrams, max_qgrams, threshold, entity) -> set:
    keys = set()
    for token in super()._tokenize_entity(entity):
        if len(token) < qgrams:
            keys.add(token)
        else:
            qgrams = [''.join(qgram) for qgram in nltk.ngrams(token, n=qgrams)]
            if len(qgrams) == 1:
                keys.update(qgrams)
            else:
                if len(qgrams) > max_qgrams:
                    qgrams = qgrams[:max_qgrams]

                minimum_length = max(1, math.floor(len(qgrams) * threshold))
                for i in range(minimum_length, len(qgrams) + 1):
                    keys.update(qgrams_combinations(qgrams, i))

    return keys


def qgrams_combinations(sublists: list, sublist_length: int) -> list:
    if sublist_length == 0 or len(sublists) < sublist_length:
        return []

    remaining_elements = sublists.copy()
    last_sublist = remaining_elements.pop(len(sublists)-1)

    combinations_exclusive_x = qgrams_combinations(remaining_elements, sublist_length)
    combinations_inclusive_x = qgrams_combinations(remaining_elements, sublist_length-1)

    resulting_combinations = combinations_exclusive_x.copy() if combinations_exclusive_x else []

    if not combinations_inclusive_x: # is empty
        resulting_combinations.append(last_sublist)
    else:
        for combination in combinations_inclusive_x:
            resulting_combinations.append(combination+last_sublist)

    return resulting_combinations
