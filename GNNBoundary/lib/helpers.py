import torch
import numpy as np
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
import json
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

def save_json(dataset_name, data):
    formatted_data = format_results(data)

    with open(f"logs/{dataset_name}.json", 'w', encoding='utf-8') as f: 
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

def load_json(dataset_name):
    with open(f"logs/{dataset_name}.json", 'r') as f:
        loaded = json.load(f)
    
    results = format_loaded(loaded)

    return results

def format_results(results):
    results_formatted = dict()
    for k, v in results.items():
        results_formatted[f"{k[0]},{k[1]}"] = v.tolist()

    return results_formatted

def format_loaded(loaded):
    results = dict()
    for k,v in loaded.items():
        results[(k[0],k[2])] = np.array(v)
    
    return results

def print_results(results):
    for k, v in results.items():
        print(f"c1 = {k[0]} c2 = {k[1]}")
        print(f'gnnboundary: c1_mean = {v[0]} c1_std = {v[1]} c2_mean = {v[2]} c2_std = {v[3]}')
        print(f'baseline: c1_mean = {v[4]} c1_std = {v[5]} c2_mean = {v[6]} c2_std = {v[7]}')

def print_results_3way(results_3way):
    for k, v in results_3way.items():
        print(f"c1 = {k[0]} c2 = {k[1]} c3 = {k[2]}")
        print(f'gnnboundary: c1_mean = {v[0]} c1_std = {v[1]} c2_mean = {v[2]} c2_std = {v[3]} c3_mean = {v[4]} c3_std = {v[5]}')
        print(f'baseline: c1_mean = {v[6]} c1_std = {v[7]} c2_mean = {v[8]} c2_std = {v[9]} c3_mean = {v[10]} c3_std = {v[11]}')
        print(f"convergence rate = {v[12]}")