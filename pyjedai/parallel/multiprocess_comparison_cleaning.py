import multiprocessing
from collections import defaultdict, deque
from queue import PriorityQueue
from time import time

from mpire import WorkerPool
from tqdm import tqdm

from pyjedai.comparison_cleaning import CardinalityEdgePruning, WeightedEdgePruning
from pyjedai.parallel.util import split_dict_into_chunks, merge_dicts, split_enum_dict_into_chunks


class SharedData:
    def __init__(self, comparison_cleaning_main_object):
        self.cc_main_object = comparison_cleaning_main_object

class MultiprocessComparisonCleaning:
    def __init__(
            self, comparison_cleaning, n_processes: int = 1
    ) -> any:

        self.n_processes = n_processes
        self.chunked_data = comparison_cleaning.data.split(n_processes)
        # self.chunked_blocks = split_enum_dict_into_chunks(comparison_cleaning._entity_index, n_processes)
        self.shared_data = SharedData(comparison_cleaning)
        self.parameters = []
        self.generate_task_objects(comparison_cleaning)
        self.pool = WorkerPool(n_jobs=n_processes, shared_objects=self.shared_data, start_method='fork')
        self.blocks = dict()

    def generate_task_objects(self, comparison_cleaning):
        parameters = dict()
        for pid in range(self.n_processes):
            parameters["pid"] = pid
            cc_part_object = init_cc_class(comparison_cleaning)
            cc_part_object._num_of_processes = self.n_processes
            # entity_matching_part_object._progress_bar = entity_matching._progress_bar
            parameters["cc"] = cc_part_object
            parameters["entity_start"] = pid * self.chunked_data[0].dataset_limit
            parameters["entity_end"] = (pid+1) * self.chunked_data[0].dataset_limit
            self.parameters.append(parameters.copy())

    def run(self):
        if isinstance(self.shared_data.cc_main_object, CardinalityEdgePruning):
            self.run_CEP()
        # if isinstance(self.shared_data.cc_main_object, WeightedEdgePruning):
        #     self.run_WEP()
        else:
            for res in self.pool.imap_unordered(apply_processing, self.parameters):
                self.blocks.update(res)

    def get_blocks(self):
        return self.blocks

    def run_CEP(self):
        threshold = self.shared_data.cc_main_object._threshold
        top_k_edges = PriorityQueue(threshold * 2)
        minimum_weight = self.shared_data.cc_main_object._minimum_weight
        for res in self.pool.imap_unordered(apply_processing, self.parameters):
            for comparison in res:
                weight = comparison[0]
                if weight >= minimum_weight:
                    top_k_edges.put((weight, comparison[1], comparison[2]))
                    if threshold < top_k_edges.qsize():
                        minimum_weight = top_k_edges.get()[0]
                else:
                    break

        self.blocks = defaultdict(set)
        while not top_k_edges.empty():
            comparison = top_k_edges.get()
            self.blocks[comparison[1]].add(comparison[2])

def init_cc_class(cc_class):
    if isinstance(cc_class, CardinalityEdgePruning):
        cc_class._set_threshold()
        cc_class._top_k_edges = PriorityQueue(cc_class._threshold*2)
        return CardinalityEdgePruning(cc_class.weighting_scheme)
    if isinstance(cc_class, WeightedEdgePruning):
        # cc_class._set_threshold()
        # cc_class._top_k_edges = PriorityQueue(cc_class._threshold*2)
        return WeightedEdgePruning(cc_class.weighting_scheme)

def apply_processing(shared_data, pid, cc, entity_start, entity_end):
    cc.tqdm_disable, cc.data = shared_data.cc_main_object.tqdm_disable, shared_data.cc_main_object.data
    cc._limit = cc.data.num_of_entities \
        if cc.data.is_dirty_er or cc._node_centric else cc.data.dataset_limit
    cc._progress_bar = tqdm(
        total=cc._limit,
        desc=cc._method_name,
        disable=cc.tqdm_disable
    )

    cc._first_entity = entity_start
    cc._limit = entity_end
    cc._entity_index = shared_data.cc_main_object._entity_index
    cc._num_of_blocks = len(shared_data.cc_main_object._blocks)
    cc._blocks = shared_data.cc_main_object._blocks
    blocks = cc._apply_main_processing()

    if isinstance(cc, CardinalityEdgePruning):
        result = deque()
        while not cc._top_k_edges.empty():
            result.appendleft(cc._top_k_edges.get())
        return result

    return blocks