from mpire import WorkerPool
from networkx import Graph, union

from pyjedai.matching import EntityMatching
from pyjedai.parallel.util import batchify, split_dict_into_chunks


class SharedData:
    def __init__(self, data, blocks, chunked_blocks):
        self.data = data
        self.chunked_blocks = chunked_blocks
        self.all_blocks = list(blocks.values())

class MultiprocessEntityMatching:
    def __init__(self, entity_matching, blocks, n_processes: int = 1
    ) -> any:
        self.n_processes = n_processes
        self.chunked_blocks = split_dict_into_chunks(blocks, self.n_processes)
        self.shared_data = SharedData(entity_matching.data, blocks, self.chunked_blocks)
        self.parameters = []
        self.generate_task_objects(entity_matching, blocks)
        self.pool = WorkerPool(n_jobs=n_processes, shared_objects=self.shared_data, start_method='fork')
        self.pairs = Graph()

    def generate_task_objects(self, entity_matching, blocks):
        batched_indices = batchify(blocks, self.n_processes)
        parameters = dict()
        pid = 0
        for indices in batched_indices:
            parameters["pid"] = pid
            pid += 1
            parameters["indices"] = indices
            entity_matching_part_object = EntityMatching(
                metric=entity_matching.metric, tokenizer=entity_matching.tokenizer,
                similarity_threshold=entity_matching.similarity_threshold,
                qgram=entity_matching.qgram, tokenizer_return_set=entity_matching.tokenizer_return_set,
                attributes=entity_matching.attributes, delim_set=entity_matching.delim_set,
                padding=entity_matching.padding, prefix_pad=entity_matching.prefix_pad,
                suffix_pad=entity_matching.suffix_pad
            )
            entity_matching_part_object.pairs = Graph()
            entity_matching_part_object._progress_bar = entity_matching._progress_bar
            parameters["entity_matching"] = entity_matching_part_object
            self.parameters.append(parameters.copy())

    def run(self):
        for res in self.pool.imap(match, self.parameters):
            self.pairs = union(self.pairs, res)

    def get_pairs(self):
        return self.pairs

def match(shared_data, entity_matching, pid, indices):

    print(f'I am {pid} and I have blocks: {indices[0]} - {indices[1]}')

    entity_matching.data = shared_data.data
    blocks = shared_data.chunked_blocks[pid]
    all_blocks = shared_data.all_blocks

    if 'Block' in str(type(all_blocks[0])):
        entity_matching._predict_raw_blocks(blocks)
    elif isinstance(all_blocks[0], set):
        entity_matching._predict_prunned_blocks(blocks)
    else:
        raise AttributeError("Wrong type of Blocks")

    return entity_matching.pairs
