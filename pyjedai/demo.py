import pandas as pd
from pprint import pprint

import cProfile

from datamodel import Data

from pyjedai.workflow import WorkFlow, compare_workflows
from pyjedai.block_building import (StandardBlocking, QGramsBlocking,
                                    ExtendedQGramsBlocking, SuffixArraysBlocking,
                                    ExtendedSuffixArraysBlocking
                                    )
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
from pyjedai.comparison_cleaning import (
    WeightedEdgePruning, WeightedNodePruning, CardinalityEdgePruning, 
    CardinalityNodePruning, BLAST, ReciprocalCardinalityNodePruning, 
    ReciprocalWeightedNodePruning, ComparisonPropagation
)
from pyjedai.matching import EntityMatching
from pyjedai.clustering import ConnectedComponentsClustering


d1 = pd.read_csv("./../data/ccer/D2/abt.csv", sep="|", engine="python", na_filter=False).astype(str)
d2 = pd.read_csv("./../data/ccer/D2/buy.csv", sep='|', engine='python', na_filter=False).astype(str)
gt = pd.read_csv("./../data/ccer/D2/gt.csv", sep='|', engine='python')
# d1 = pd.read_csv("./../data/ccer/D3/amazon.csv", sep="#", engine="python", na_filter=False).astype(str)
# d2 = pd.read_csv("./../data/ccer/D3/gp.csv", sep='#', engine='python', na_filter=False).astype(str)
# gt = pd.read_csv("./../data/ccer/D3/gt.csv", sep='#', engine='python')
# d1 = pd.read_csv("./../data/ccer/D8/amazon.csv", sep="|", engine="python", na_filter=False).astype(str)
# d2 = pd.read_csv("./../data/ccer/D8/walmart.csv", sep='|', engine='python', na_filter=False).astype(str)
# gt = pd.read_csv("./../data/ccer/D8/gt.csv", sep='|', engine='python')

data = Data(
    dataset_1=d1,
    attributes_1=['id','name','description', 'price'],
    id_column_name_1='id',
    dataset_2=d2,
    attributes_2=['id','name','description', 'price'],
    id_column_name_2='id',
    ground_truth=gt,
)

# id#title#description#manufacturer#price#aggregate value
# data = Data(
#     dataset_1=d1,
#     attributes_1=['id', 'title', 'description', 'price', 'manufacturer', 'aggregate value'],
#     id_column_name_1='id',
#
#     dataset_2=d2,
#     attributes_2=['id', 'title', 'description', 'price', 'manufacturer', 'aggregate value'],
#     id_column_name_2='id',
#
#     ground_truth=gt,
# )

# id|title|modelno|price|shipweight|brand|dimensions|aggregate value
# id|title|modelno|price|shipweight|brand|dimensions|aggregate value
# data = Data(
#     dataset_1=d1,
#     attributes_1=['id', 'title', 'modelno', 'price', 'shipweight', 'brand', 'dimensions', 'aggregate value'],
#     id_column_name_1='id',
#
#     dataset_2=d2,
#     attributes_2=['id', 'title', 'modelno', 'price', 'shipweight', 'brand', 'dimensions', 'aggregate value'],
#     id_column_name_2='id',
#
#     ground_truth=gt,
# )


data.print_specs()
pprint(data.dataset_1.head(2))

w = WorkFlow(
    block_building = dict( method=QGramsBlocking, params=dict(qgrams=3) ),
    block_cleaning = [
        dict( method=BlockFiltering, params=dict(ratio=0.8) ),
        dict( method=BlockPurging, params=dict(smoothing_factor=1.025) ) ],
    comparison_cleaning = dict(method=WeightedEdgePruning),
    entity_matching = dict( method=EntityMatching, metric='sorensen_dice',
        similarity_threshold=0.5, attributes = ['description', 'name'] ),
    clustering = dict(method=ConnectedComponentsClustering),
    name="Workflow-QGramsBlocking"
)

# cProfile.run('w.run(data, workflow_tqdm_enable=True, verbose=False)')
w.run(data, workflow_tqdm_enable=True, verbose=False)


pprint(w.to_df())

w.visualize()
