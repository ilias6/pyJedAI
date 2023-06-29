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


