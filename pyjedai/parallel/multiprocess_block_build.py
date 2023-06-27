import math
import re
import time

import nltk
from mpire import WorkerPool

from pyjedai.datamodel import Block
from pyjedai.parallel.util import batchify, merge_dicts


class SharedData:
    def __init__(self, data, ids):
        self.data = data
        self.ids = ids


class MultiprocessBlockBuilding:

    def __init__(
            self, data, ids, blocks, parameters:dict, n_processes:int
            , progress_bar: bool = True
    ) -> any:
        self.n_processes = n_processes
        self.parameters = []
        self.blocks = blocks
        self.shared_data = SharedData(data, ids)
        self.pool = WorkerPool(n_jobs=n_processes, shared_objects=self.shared_data, start_method='fork')
        self.progress_bar = progress_bar
        self.split_data(data, parameters)
        self.start_time = time.time()

    def split_data(self, data, parameters):

        batched_indices = batchify(data, self.n_processes)

        for indices in batched_indices:
            parameters["indices"] = indices
            self.parameters.append(parameters.copy())

    def run(self) -> None:
        self.start_time = time.time()
        for res in self.pool.imap_unordered(self.build_blocks_for_entity, self.parameters):
            print(f' RESULT AT: {time.time() - self.start_time}')
            merge_dicts(self.blocks, res)
        # self.pool.stop_and_join()

    def get_logs(self):
        return self.pool.get_insights()

    def get_blocks(self):
        return self.blocks

    def build_blocks_for_entity(
            self,
            shared_data,
            indices: tuple,
            last_id: int,
            entities_id: int,
    ) -> {Block}:

        if last_id is None:
            last_id = 0

        entities = shared_data.data
        partial_entities = entities[indices[0]:indices[1]]

        ids = shared_data.ids
        partial_ids = ids[indices[0]:indices[1]]

        result = dict()

        for entity, eid in zip(partial_entities, partial_ids):
            eid = int(eid) + last_id

            for token in entity:
                result.setdefault(token, Block())
                if entities_id == 1:
                    result[token].entities_D1.add(eid)
                elif entities_id == 2:
                    result[token].entities_D2.add(eid)
                else:
                    print("YOU DID SOMETHING VERY WRONG WITH ENTITIES ID!")

        print(f' FINISHED AT: {time.time() - self.start_time}')
        return result


def standard_blocking_tokenize_entity(entity: str) -> list:
    """Produces a list of workds of a given string

    Args:
        entity (str): String representation  of an entity

    Returns:
        list: List of words
    """
    return list(set(filter(None, re.split('[\\W_]', entity.lower()))))


def qgrams_tokenize_entity(qgrams, entity) -> set:
    keys = set()
    for token in standard_blocking_tokenize_entity(entity):
        if len(token) < qgrams:
            keys.add(token)
        else:
            keys.update(''.join(qg) for qg in nltk.ngrams(token, n=qgrams))
    return keys


def suffix_arrays_tokenize_entity(suffix_length, entity) -> set:
    keys = set()
    for token in standard_blocking_tokenize_entity(entity):
        if len(token) < suffix_length:
            keys.add(token)
        else:
            for length in range(0, len(token) - suffix_length + 1):
                keys.add(token[length:])
    return keys


def extended_suffix_arrays_tokenize_entity(suffix_length, entity) -> set:
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
