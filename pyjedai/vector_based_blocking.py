'''
Contains all methods for creating embeddings from text 
and then performing NNs methods for cluster formation.
'''
import os
import pickle
import re
import sys
import warnings
from time import time
from typing import List, Tuple

import faiss
import gensim.downloader as api
import networkx as nx
import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm
from transformers import (AlbertModel, AlbertTokenizer, BertModel,
                          BertTokenizer, DistilBertModel, DistilBertTokenizer,
                          RobertaModel, RobertaTokenizer, XLNetModel,
                          XLNetTokenizer)

transformers.logging.set_verbosity_error()

from .datamodel import Data, PYJEDAIFeature
from .evaluation import Evaluation
from .utils import SubsetIndexer

EMBEDDINGS_DIR = '.embeddings'
if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)
    EMBEDDINGS_DIR = os.path.abspath(EMBEDDINGS_DIR)
    print('Created embeddings directory at: ' + EMBEDDINGS_DIR)
    
LINUX_ENV=False
# try:
#     if 'linux' in sys.platform:
#         import falconn
#         import scann
#         LINUX_ENV=True
# except:
#     warnings.warn(ImportWarning, "Can't use FALCONN/SCANN in windows environment")

class EmbeddingsNNBlockBuilding(PYJEDAIFeature):
    """Block building via creation of embeddings and a Nearest Neighbor Approach.
    """

    _method_name = "Embeddings-NN Block Building"
    _method_info = "Creates a set of candidate pais for every entity id " + \
        "based on Embeddings creariot and Similarity search among the vectors."

    _gensim_mapping_download = {
        'fasttext' : 'fasttext-wiki-news-subwords-300',
        'glove' : 'glove-wiki-gigaword-300',
        'word2vec' : 'word2vec-google-news-300'
    }
    _sentence_transformer_mapping = {
        'smpnet' : 'all-mpnet-base-v2',
        'st5' : 'gtr-t5-large',
        'sdistilroberta' : 'all-distilroberta-v1',
        'sminilm' : 'all-MiniLM-L12-v2',
        'sent_glove' : 'average_word_embeddings_glove.6B.300d'
    }

    def __init__(
            self,
            vectorizer: str,
            similarity_search: str
    ) -> None:
        super().__init__()
        self.vectorizer, self.similarity_search = vectorizer, similarity_search
        self.embeddings: np.array
        self.vectors_1: np.array
        self.vectors_2: np.array = None
        self.vector_size: int
        self.num_of_clusters: int
        self.top_k: int
        self._faiss_metric_type = None

    def _tokenize_entity(self, entity: str) -> str:
        """Produces a list of workds of a given string

        Args:
            entity (str): String representation  of an entity

        Returns:
            str: entity string
        """
        return entity.strip().lower()#' '.join(list(filter(None, re.split('[\\W_]', entity.lower()))))

    def build_blocks(self,
                     data: Data,
                     vector_size: int = 300,
                     num_of_clusters: int = 5,
                     top_k: int = 30,
                     max_word_embeddings_size: int = 256,
                     attributes_1: list = None,
                     attributes_2: list = None,
                     return_vectors: bool = False,
                     tqdm_disable: bool = False,
                     save_embeddings: bool = True,
                     load_embeddings_if_exist: bool = False,
                     with_entity_matching: bool = False
    ) -> any:
        """Main method of the vector based approach. Contains two steps. First an embedding method. \
            And afterwards a similarity search upon the vectors created in the previous step.
            Pre-trained schemes are used for the embedding process.

        Args:
            data (Data): dataset from datamodel
            vector_size (int, optional): For the Gensim vectorizers. Defaults to 300. \
                Qaution for the hugging face embeddings has no effect.
            num_of_clusters (int, optional): Number of clusters for FAISS. Defaults to 5.
            top_k (int, optional): Top K similar candidates. Defaults to 30.
            attributes_1 (list, optional): Vectorization of specific attributes for D1. \
                                            Defaults to None.
            attributes_2 (list, optional): Vectorization of specific attributes for D2. \
                                            Defaults to None.
            return_vectors (bool, optional): If true, returns the vectors created from the pretrained \
                                            embeddings instead of the blocks. Defaults to False.
            tqdm_disable (bool, optional): Disable progress bar. For experiment purposes. \
                                            Defaults to False.

        Raises:
            AttributeError: Vectorizer check
            AttributeError: Similarity Search method check.

        Returns:
            dict: Entity ids to sets of top-K candidate ids. OR
            Tuple(np.array, np.array): vectors from d1 and vectors from d2
        """
        _start_time = time()
        self.blocks = dict()
        self.with_entity_matching = with_entity_matching
        self.save_embeddings, self.load_embeddings_if_exist = save_embeddings, load_embeddings_if_exist
        self.max_word_embeddings_size = max_word_embeddings_size
        self.data, self.attributes_1, self.attributes_2, self.vector_size, self.num_of_clusters, self.top_k \
            = data, attributes_1, attributes_2, vector_size, num_of_clusters, top_k
        self._progress_bar = tqdm(total=data.num_of_entities,
                                  desc=(self._method_name + ' [' + self.vectorizer + ', ' + self.similarity_search + ']'),
                                  disable=tqdm_disable)

        self._si = SubsetIndexer(None, self.data)
        
        # print(data.attributes_1, data.attributes_2)
        self._entities_d1 = data.dataset_1[attributes_1 if attributes_1 else data.attributes_1] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist()
        
        self._entities_d2 = data.dataset_2[attributes_2 if attributes_2 else data.attributes_2] \
                    .apply(" ".join, axis=1) \
                    .apply(self._tokenize_entity) \
                    .values.tolist() if not data.is_dirty_er else None

        vectors_1 = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device selected: ", self.device)

        if self.with_entity_matching:
            self.graph = nx.Graph()
        
        if load_embeddings_if_exist:
            try:
                print("Loading embeddings from file...")
                
                p1 = os.path.join(EMBEDDINGS_DIR, self.vectorizer + '_' + self.data.dataset_name_1 \
                                                    if self.data.dataset_name_1 is not None else "d1" +'_1.npy')
                print("Loading file: ", p1)
                self.vectors_1 = vectors_1 = np.load(p1)
                self._progress_bar.update(data.num_of_entities_1)

                p2 = os.path.join(EMBEDDINGS_DIR, self.vectorizer + '_' + self.data.dataset_name_2 \
                                                    if self.data.dataset_name_2 is not None else "d2" +'_2.npy')
                print("Loading file: ", p2)
                self.vectors_2 = vectors_2 = np.load(p2)
                self._progress_bar.update(data.num_of_entities_2)
                
                print("Loading embeddings from file finished")
            except:
                print("Embeddings not found. Creating new ones.")
                raise ValueError("Embeddings not found.")
        else:
            if self.vectorizer in ['word2vec', 'fasttext', 'doc2vec', 'glove']:
                vectors_1, vectors_2 = self._create_gensim_embeddings()
            elif self.vectorizer in ['bert', 'distilbert', 'roberta', 'xlnet', 'albert']:
                vectors_1, vectors_2 = self._create_pretrained_word_embeddings()
            elif self.vectorizer in ['smpnet', 'st5', 'sent_glove', 'sdistilroberta', 'sminilm']:
                vectors_1, vectors_2 = self._create_pretrained_sentence_embeddings()
            else:
                raise AttributeError("Not available vectorizer")

        if save_embeddings:
            print("Saving embeddings...")
            
            p1 = os.path.join(EMBEDDINGS_DIR, self.vectorizer + '_' + self.data.dataset_name_1 \
                                                    if self.data.dataset_name_1 is not None else "d1" +'_1.npy')
            print("Saving file: ", p1)
            np.save(p1, self.vectors_1)
            
            p2 = os.path.join(EMBEDDINGS_DIR, self.vectorizer + '_' + self.data.dataset_name_2 \
                                                    if self.data.dataset_name_2 is not None else "d2" +'_2.npy')
            print("Saving file: ", p2)
            np.save(p2, self.vectors_2)

        if return_vectors:
            return (vectors_1, _) if data.is_dirty_er else (vectors_1, vectors_2)

        if self.similarity_search == 'faiss':
            self._faiss_metric_type = faiss.METRIC_L2
            self._similarity_search_with_FAISS()
        elif self.similarity_search == 'falconn':
            raise NotImplementedError("FALCONN")
        elif self.similarity_search == 'scann'  and LINUX_ENV:
            self._similarity_search_with_SCANN()
        else:
            raise AttributeError("Not available method")
        self._progress_bar.close()        
        self.execution_time = time() - _start_time

        if self.with_entity_matching:
            return self.blocks, self.graph
        
        return self.blocks

    def _create_gensim_embeddings(self) -> Tuple[np.array, np.array]:
        """Embeddings with Gensim. More on https://github.com/RaRe-Technologies/gensim-data

        Args:
            entities_d1 (list): Entities from D1
            entities_d2 (list, optional): Entities from D2 (CCER). Defaults to None.

        Returns:
            Tuple[np.array, np.array]: Embeddings from D1 and D2
        """
        vectors_1 = []
        vocabulary = api.load(self._gensim_mapping_download[self.vectorizer])
        for e1 in self._entities_d1:
            vectors_1.append(self._create_vector(e1, vocabulary))
            self._progress_bar.update(1)
        self.vectors_1 = np.array(vectors_1).astype('float32')

        vectors_2 = []
        if not self.data.is_dirty_er:
            for e2 in self._entities_d2:
                vectors_2.append(self._create_vector(e2, vocabulary))
                self._progress_bar.update(1)
            self.vectors_2 = np.array(vectors_2).astype('float32')

        return vectors_1, vectors_2

    def _create_pretrained_word_embeddings(self) -> Tuple[np.array, np.array]:
        if self.vectorizer == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained("bert-base-uncased")
        elif self.vectorizer == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif self.vectorizer == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained('roberta-base')
        elif self.vectorizer == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            model = XLNetModel.from_pretrained('xlnet-base-cased')
        elif self.vectorizer == 'albert':
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            model = AlbertModel.from_pretrained("albert-base-v2")

        model = model.to(self.device)
        self.vectors_1 = self._transform_entities_to_word_embeddings(self._entities_d1,
                                                                     model,
                                                                     tokenizer)
        self.vector_size = self.vectors_1[0].shape[0]
        print("Vector size: ", self.vectors_1.shape)
        self.vectors_2 = self._transform_entities_to_word_embeddings(self._entities_d2,
                                                                     model,
                                                                     tokenizer) if not self.data.is_dirty_er else None
        return self.vectors_1, self.vectors_2


    def _transform_entities_to_word_embeddings(self, entities, model, tokenizer) -> np.array:
    
        model = model.to(self.device)
        embeddings = []
        
        for entity in entities:
            encoded_input = tokenizer(entity,
                                        return_tensors='pt',
                                        truncation=True,
                                        return_attention_mask = True,
                                        max_length=self.max_word_embeddings_size,
                                        padding='max_length')

            encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}  # Move input tensors to GPU

            with torch.no_grad():
                encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}  # Move input tensors to GPU
                output = model(**encoded_input)
                vector = output.last_hidden_state[:, 0, :]
                
            vector = vector.cpu().numpy()
            embeddings.append(vector.reshape(-1))
            self._progress_bar.update(1)
        
        self.vector_size = embeddings[0].shape[0]
        return np.array(embeddings).astype('float32')

    def _create_pretrained_sentence_embeddings(self):
        model = SentenceTransformer(self._sentence_transformer_mapping[self.vectorizer],
                                    device=self.device)
        vectors_1 = []
        for e1 in self._entities_d1:
            vector = model.encode(e1)
            vectors_1.append(vector)
            self._progress_bar.update(1)
        self.vector_size = len(vectors_1[0])
        self.vectors_1 = np.array(vectors_1).astype('float32')
        vectors_2 = []
        if not self.data.is_dirty_er:
            for e2 in self._entities_d2:
                vector = model.encode(e2)
                vectors_2.append(vector)
                self._progress_bar.update(1)
            self.vector_size = len(vectors_2[0])
            self.vectors_2 = np.array(vectors_2).astype('float32')

        return vectors_1, vectors_2

    def _similarity_search_with_FAISS(self):
        index = faiss.IndexFlatL2(self.vectors_1.shape[1])
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.train(self.vectors_1)  # train on the vectors of dataset 1
        index.add(self.vectors_1)   # add the vectors and update the index

        self.distances, self.neighbors = index.search(self.vectors_1 if self.data.is_dirty_er else self.vectors_2,
                                                      self.top_k)
        
        self.blocks = dict()
        
        for i in range(0, self.neighbors.shape[0]):
            if self.data.is_dirty_er:
                entity_id_d2 = i
            else:
                entity_id_d2 = i + self.data.dataset_limit
            
            if entity_id_d2 not in self.blocks.keys():
                self.blocks[entity_id_d2] = set()            
            
            j = 0
            for entity_id_d1 in self.neighbors[i]:

                if entity_id_d1 == -1:
                    continue
                
                if entity_id_d1 not in self.blocks.keys():
                    self.blocks[entity_id_d1] = set()

                self.blocks[entity_id_d1].add(entity_id_d2)
                self.blocks[entity_id_d2].add(entity_id_d1)
                
                if self.with_entity_matching:
                    self.graph.add_edge(entity_id_d2, entity_id_d1, weight=self.distances[i][j])

                j += 1


    def _similarity_search_with_FALCONN(self):
        pass

    def _similarity_search_with_SCANN(self):
        if not LINUX_ENV:
            raise ImportError("Can't use SCANN in windows environment. Use FAISS instead.")

        searcher = scann.scann_ops_pybind.builder(self.vectors_1,
                                                    num_neighbors=self.top_k, 
                                                    distance_measure="dot_product") \
                                        .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000) \
                                        .score_ah(2, anisotropic_quantization_threshold=0.2) \
                                        .reorder(100) \
                                        .build()

        self.neighbors, self.distances = searcher.search_batched(
            self.vectors_1 if self.data.is_dirty_er else self.vectors_2,
            final_num_neighbors=self.top_k
        )
        if self.data.is_dirty_er:
            self.blocks = {
                self._si.d1_retained_ids[i] : set(x for x in self.neighbors[i] if x not in [-1, i]) \
                        for i in range(0, self.neighbors.shape[0])
            }
        else:
            self.blocks = {
                self._si.d2_retained_ids[i] : set(x for x in self.neighbors[i] if x != -1) \
                        for i in range(0, self.neighbors.shape[0])
            }

    def _create_vector(self, tokens: List[str], vocabulary) -> np.array:
        num_of_tokens = 0
        vector = np.zeros(self.vector_size)
        for token in tokens.split():
            if token in vocabulary:
                vector += vocabulary[token]
                num_of_tokens += 1
        if num_of_tokens > 0:
            vector /= num_of_tokens

        return vector

    def evaluate(self,
                 prediction,
                 export_to_df: bool = False,
                 export_to_dict: bool = False,
                 with_classification_report: bool = False,
                 verbose: bool = True,
                 with_stats: bool = False) -> any:

        if self.data is None:
            raise AttributeError("Can not proceed to evaluation without data object.")

        if self.data.ground_truth is None:
            raise AttributeError("Can not proceed to evaluation without a ground-truth file. " +
                    "Data object has not been initialized with the ground-truth file")

        eval_obj = Evaluation(self.data)
        true_positives = 0
        total_matching_pairs = sum([len(block) for block in prediction.values()])
        for _, (id1, id2) in self.data.ground_truth.iterrows():
            id1 = self.data._ids_mapping_1[id1]
            id2 = self.data._ids_mapping_1[id2] if self.data.is_dirty_er \
                                                else self.data._ids_mapping_2[id2]
            if (id1 in prediction and id2 in prediction[id1]) or   \
                (id2 in prediction and id1 in prediction[id2]):
                true_positives += 1

        eval_obj.calculate_scores(true_positives=true_positives, 
                                  total_matching_pairs=total_matching_pairs)
        evaluation = eval_obj.report(self.method_configuration(),
                                export_to_df,
                                export_to_dict,
                                with_classification_report,
                                verbose)

        if with_stats:
            self.stats()

        return evaluation

    def _configuration(self) -> dict:
        return {
            "Vectorizer" : self.vectorizer,
            "Similarity-Search" : self.similarity_search,
            "Top-K" : self.top_k,
            "Vector size": self.vector_size
        }
    
    def stats(self) -> None:
        print("Statistics:")
        if self.similarity_search == 'faiss':
            print(" FAISS:" +
                # "\n\tNumber of entries in each list:  " + str(self._faiss_num_lists) + 
                "\n\tIndices shape returned after search: " + str(self.neighbors.shape)
            )
        elif self.similarity_search == 'falconn':           
            pass
        elif self.similarity_search == 'scann'  and LINUX_ENV:
            pass
        
        print(u'\u2500' * 123)
        
class PREmbeddingsNNBlockBuilding(EmbeddingsNNBlockBuilding):
    """Block building via creation of embeddings and a Nearest Neighbor Approach with specified budget
    """

    _method_name = "Progressive Embeddings-NN Block Building"
    _method_info = "Creates a set of candidate pairs for every entity id " + \
        "based on Progresssive Similarity Search over the entity embeddings neighborhood"

    def __init__(
            self,
            vectorizer: str,
            similarity_search: str,
            budget: int
    ) -> None:
        super().__init__(vectorizer, similarity_search)
        self._budget = budget
        self.vectorizer, self.similarity_search = vectorizer, similarity_search
        self.embeddings: np.array
        self.vectors_1: np.array
        self.vectors_2: np.array = None
        self.vector_size: int
        self.num_of_clusters: int
        self.top_k: int

    def precomputed_vectors(self,
                    data: Data,
                    vectors_1: np.array = None,
                    vectors_2: np.array = None
    ) -> bool:

        if(not vectors_1):
            return False

        if(not data.is_dirty_er):
            return vectors_2 is not None

        return True

    def build_blocks(self,
                     data: Data,
                     cc_blocks: dict = None,
                     vectors_1: np.array = None,
                     vectors_2: np.array = None, 
                     vector_size: int = 300,
                     num_of_clusters: int = 5,
                     top_k: int = 30,
                     max_word_embeddings_size: int = 256,
                     attributes_1: list = None,
                     attributes_2: list = None,
                     return_vectors: bool = False,
                     tqdm_disable: bool = True
    ) -> any:
        """Retrieves the entities retained after the last step, produces/retrieves their embeddings and applies popular NN neighbor techniques
           to produce pair candidates
        Args:
            data (Data): Data Module
            cc_blocks (dict, optional): Blocks, subset of the initial dataset retrieved from the last CC step. Defaults to None.
            vectors_1 (np.array, optional): Precalculated embeddings of entities of dataset 1. Defaults to None.
            vectors_2 (np.array, optional): Precalculated embeddings of entities of dataset 2. Defaults to None.
            vector_size (int, optional): The size of the embeddings. Defaults to 300.
            num_of_clusters (int, optional): The number of clusters. Defaults to 5.
            top_k (int, optional): The number of candidates that will be produced per entity. Defaults to 30.
            attributes_1 (list, optional): Attributes of entities in dataset 1 that we want to take into consideration. Defaults to None.
            attributes_2 (list, optional): Attributes of entities in dataset 2 that we want to take into consideration. Defaults to None.
            return_vectors (bool, optional): If true, returns the entities vector embeddings. Defaults to True.
            tqdm_disable (bool, optional): Disable progress bar. For experiment purposes. Defaults to False.
        Returns:
            any: Blocks produced by Vector Based BB, vector distances and entity embeddings
        """
        _start_time = time()
        self.blocks = dict()
        self.max_word_embeddings_size = max_word_embeddings_size
        self.data, self.attributes_1, self.attributes_2, self.vector_size, self.num_of_clusters, self.top_k \
            = data, attributes_1, attributes_2, vector_size, num_of_clusters, top_k
        self._progress_bar = tqdm(total=data.num_of_entities,
                                  desc=self._method_name,
                                  disable=tqdm_disable)

        self._si = SubsetIndexer(cc_blocks, self.data)
        self._d1_valid_indices: list[int] = self._si.d1_retained_ids
        self._d2_valid_indices: list[int] = [x - self.data.dataset_limit for x in self._si.d2_retained_ids]       


        self._entities_d1 = data.dataset_1[attributes_1 if attributes_1 else data.attributes_1] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist()
        self._entities_d1 = [self._entities_d1[x] for x in self._d1_valid_indices]
                            
                            
        self._entities_d2 = data.dataset_2[attributes_2 if attributes_2 else data.attributes_2] \
                            .apply(" ".join, axis=1) \
                            .apply(self._tokenize_entity) \
                            .values.tolist() if not data.is_dirty_er else None
        self._entities_d2 = [self._entities_d2[x] for x in self._d2_valid_indices] if not data.is_dirty_er else None
                                               

        self.vectors_1 = vectors_1
        self.vectors_2 = vectors_2
        self.create_vectors()
        if(return_vectors): return self.vectors_1, self.vectors_2       
        self.blocks = self.create_blocks()
        self._progress_bar.close()
        self.execution_time = time() - _start_time

        return self.blocks

    def create_vectors(self):      

        if(self.precomputed_vectors(data = self.data)):
            self.vectors_1 = self.vectors_1[self._d1_valid_indices]

            if(not self.data.is_dirty_er):
                self.vectors_2 = self.vectors_2[self._d2_valid_indices]
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Device selected: ", self.device)
            
            if self.vectorizer in ['word2vec', 'fasttext', 'doc2vec', 'glove']:
                vectors_1, vectors_2 = self._create_gensim_embeddings()
            elif self.vectorizer in ['bert', 'distilbert', 'roberta', 'xlnet', 'albert']:
                vectors_1, vectors_2 = self._create_pretrained_word_embeddings()
            elif self.vectorizer in ['smpnet', 'st5', 'glove', 'sdistilroberta', 'sminilm']:
                vectors_1, vectors_2 = self._create_pretrained_sentence_embeddings()
            else:
                raise AttributeError("Not available vectorizer")
        
        self.vectors_1 = vectors_1
        self.vectors_2 = vectors_2

    def create_blocks(self):
        if self.similarity_search == 'faiss':
            self._similarity_search_with_FAISS()
        elif self.similarity_search == 'falconn':
            raise NotImplementedError("FALCONN")
        elif self.similarity_search == 'scann'  and LINUX_ENV:
            self._similarity_search_with_SCANN()
        else:
            raise AttributeError("Not available method")
        return self.blocks
