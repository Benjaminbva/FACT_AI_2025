from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np

from .boundary_analysis import boundary_analysis


def triangular_boundary_analysis(model, dataset_list, key='embeds_last', k=100, n=100, threshold = 0.6):
    c = len(dataset_list)
    adjacent_triplets = dict()
    for i, ii, iii in combinations_with_replacement(range(c), 3):
        if i == ii or ii == iii or i == iii:
            continue

        result_1 = boundary_analysis(
            model, dataset_list[i], dataset_list[ii],
            key=key, k=k, n=n
        )
        result_2 = boundary_analysis(
            model, dataset_list[ii], dataset_list[iii],
            key=key, k=k, n=n
        )
        result_3 = boundary_analysis(
            model, dataset_list[i], dataset_list[iii],
            key=key, k=k, n=n
        )

        if result_1.adj_ratio > threshold and result_2.adj_ratio > threshold and result_3.adj_ratio > threshold:
            # adjacent_triplets.append([i, ii, iii])
            adjacent_triplets[i, ii, iii] = [result_1.adj_ratio, result_2.adj_ratio, result_3.adj_ratio]


    return adjacent_triplets