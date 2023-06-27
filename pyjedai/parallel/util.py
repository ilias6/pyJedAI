import warnings
from itertools import islice


def batch(iterable, size):
    iterator = iter(iterable)
    result = []
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return result
        result.append(batch)


def batchify(data, n):
    if n <= 0:
        return []

    chunks_indices = []

    len_1 = len(data)

    chunk_size_1 = len_1 // n
    remainder_1 = len_1 % n

    chunk_start_1 = 0
    for i in range(n):

        chunk_end_1 = chunk_start_1+chunk_size_1
        if remainder_1 > 0:
            chunk_end_1 += 1
            remainder_1 -= 1
        if chunk_end_1 > len_1:
            chunk_end_1 = len_1

        chunks_indices.append((chunk_start_1, chunk_end_1))
        chunk_start_1 = chunk_end_1

    return chunks_indices

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def count_unique(l):

    unique_keys = set()
    for s in l:
        if isinstance(s, set):
            unique_keys |= s
        else:
            unique_keys |= set(s.keys())

    count = len(unique_keys)
    return count

def find_different_keys(list1, list2):
    merged_list = []
    merged_list.extend(list1)
    merged_list.extend(list2)

    common_keys = set()
    different_keys = set()

    for s in merged_list:
        common_keys |= s
        different_keys |= s

    different_keys -= common_keys

    return different_keys

def merge_dicts(dict1, dict2):
    """
    Be careful!
    The value object must have a concat method implemented.
    """

    for key in dict2.keys():
        try:
            dict1[key].concat(dict2[key])
        except KeyError:
            dict1[key] = dict2[key]
