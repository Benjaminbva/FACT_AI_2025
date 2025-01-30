import torch
import numpy as np
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_style("white")
sns.set(font_scale=1.6)

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def extract_adjacent_classes(adj_ratio_mat, threshold):
    adjacent_classes = list()
    n_classes = len(adj_ratio_mat)
    for i in range(n_classes):
        for j in range (n_classes):
            if i != j and adj_ratio_mat[i][j] > threshold:
                adjacent_classes.append(frozenset([i,j]))

    adjacent_classes = list(set(adjacent_classes))

    for i, adj_class in enumerate(adjacent_classes):
        adjacent_classes[i] = list(adj_class)

    return adjacent_classes

def draw_matrix_colorless(matrix, names, fmt='.2f', vmin=0, vmax=None,
                annotsize=20, labelsize=18, xlabel='Predicted', ylabel='Actual'):
    ax = sns.heatmap(
        matrix,
        annot=True, annot_kws=dict(size=annotsize), fmt=fmt, vmin=vmin, vmax=vmax, linewidth=1,
        cmap=sns.color_palette("light:grey", as_cmap=False),
        xticklabels=names,
        yticklabels=names,
    )
    ax.set_facecolor('white')
    ax.tick_params(axis='x', labelsize=labelsize, rotation=0)
    ax.tick_params(axis='y', labelsize=labelsize, rotation=0)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()
