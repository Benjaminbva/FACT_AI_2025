import sys
import os
sys.path.append(os.path.abspath('../'))

from gnnboundary import *
import torch
import argparse
from lib import *
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def normalGenerate2Way(dataset, model, mean_embeds, num_graphs, num_iter, cls_1, cls_2, show_progress):
    iteration = 0
    while True:
        trainer = NewTrainer(
            sampler=(s := GraphSampler(
                max_nodes=20,
                temperature=0.15,
                num_node_cls=len(dataset.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(
                    classes=[cls_1, cls_2], alpha=1, beta=1
                ), weight=25), # main reg function
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                # dict(key="logits", criterion=MeanPenalty(), weight=2),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1), # L1 reg
                dict(key="omega", criterion=NormPenalty(order=2), weight=1), # L2 reg 
                # dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=dataset,
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1), # budget reg
        )

        if trainer.train(
            iterations=num_iter,
            target_probs={cls_1: (0.45, 0.55), cls_2: (0.45, 0.55)},
            target_size=60,
            w_budget_init=1,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=32,
            show_progress=show_progress
        ):
            res = trainer.quantitative(sample_size=num_graphs)
            base_res = trainer.quantitative_baseline()

            # print(f'c1 = {cls_1}')
            # print(f'mean = {res["mean"][cls_1]} std = {res["std"][cls_2]}')
            # print(f'c2 = {cls_2}')
            # print(f'mean = {res["mean"][cls_2]} std = {res["std"][cls_2]}\n')

            return {
                "gnnboundary": [res["mean"][cls_1], res["std"][cls_1], res["mean"][cls_2], res["std"][cls_2]], 
                "baseline": [base_res["mean"][cls_1], base_res["std"][cls_1], base_res["mean"][cls_2], base_res["std"][cls_2]]
            }
        
        else:
            iteration += 1
            if iteration == 3:
                res = trainer.quantitative(sample_size=num_graphs)
                base_res = trainer.quantitative_baseline()

                return {
                    "gnnboundary": [res["mean"][cls_1], res["std"][cls_1], res["mean"][cls_2], res["std"][cls_2]], 
                    "baseline": [base_res["mean"][cls_1], base_res["std"][cls_1], base_res["mean"][cls_2], base_res["std"][cls_2]]
                }

def gpuGenerate2Way(dataset, model, mean_embeds, num_graphs, num_iter, cls_1, cls_2, show_progress):
    iteration = 0
    while True:
        trainer = GPUTrainer(
            sampler=(s := GraphSampler(
                max_nodes=20,
                temperature=0.15,
                num_node_cls=len(dataset.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(
                    classes=[cls_1, cls_2], alpha=5, beta=1
                ), weight=25),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=5),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                # dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=1),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=dataset,
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
        )

        if trainer.train(
            iterations=num_iter,
            target_probs={cls_1: (0.45, 0.55), cls_2: (0.45, 0.55)},
            target_size=60,
            w_budget_init=1,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=32,
            show_progress=show_progress
        ):
            res = trainer.quantitative(sample_size=num_graphs)
            base_res = trainer.quantitative_baseline()

            return {
                "gnnboundary": [res["mean"][cls_1].item(), res["std"][cls_1].item(), 
                                res["mean"][cls_2].item(), res["std"][cls_2].item()], 
                "baseline": [base_res["mean"][cls_1].item(), base_res["std"][cls_1].item(), 
                             base_res["mean"][cls_2].item(), base_res["std"][cls_2].item()]
            }
        
        else:
            iteration += 1
            if iteration == 3:
                res = trainer.quantitative(sample_size=num_graphs)
                base_res = trainer.quantitative_baseline()

                return {
                    "gnnboundary": [res["mean"][cls_1].item(), res["std"][cls_1].item(), 
                                    res["mean"][cls_2].item(), res["std"][cls_2].item()], 
                    "baseline": [base_res["mean"][cls_1].item(), base_res["std"][cls_1].item(), 
                                base_res["mean"][cls_2].item(), base_res["std"][cls_2].item()]
                }

def get_results_2way(model, adjacent_classes, dataset_name, seeds, show_progress=True, num_iter=2000, num_graphs=1000):
    """
    returns dictionary for each adjacent class pair

    the key values are class 1 and class 2
    the item values is an array with:
    [ gnn_mean_c1, gnn_std_c1, gnn_mean_c2, gnn_std_c2, base_mean_c1, _base_std_c1, base_mean_c2, base_std_c2 ]
    """
    results = dict()

    for seed in seeds:
        print(f"using seed {seed}")
        set_seed(seed)

        if dataset_name == "mrsc9":
            dataset = MSRCDataset(seed=seed)
            func = gpuGenerate2Way
        if dataset_name == "collab":
            dataset = CollabDataset(seed=seed)
            func = normalGenerate2Way
        if dataset_name == "enzymes":
            dataset = ENZYMESDataset(seed=seed)
            func = normalGenerate2Way
        if dataset_name == "motif":
            dataset = MotifDataset(seed=seed)
            func = normalGenerate2Way

        dataset_list_gt = dataset.split_by_class()
        mean_embeds = [d.model_transform(model, key="embeds").mean(dim=0) for d in dataset_list_gt]

        for adjacent_pair in adjacent_classes:
            generations = func(
                dataset, model, mean_embeds, num_graphs, num_iter, adjacent_pair[0], adjacent_pair[1], show_progress
            )


            if tuple(adjacent_pair) in results:
                results[tuple(adjacent_pair)] += np.array([generations["gnnboundary"][0], generations["gnnboundary"][1], 
                    generations["gnnboundary"][2], generations["gnnboundary"][3], generations["baseline"][0], 
                    generations["baseline"][1], generations["baseline"][2], generations["baseline"][3]]) / len(seeds)
            else:
                results[tuple(adjacent_pair)] = np.array([generations["gnnboundary"][0], generations["gnnboundary"][1], 
                    generations["gnnboundary"][2], generations["gnnboundary"][3], generations["baseline"][0], 
                    generations["baseline"][1], generations["baseline"][2], generations["baseline"][3]]) / len(seeds)

    return results


def normalGenerate3Way(dataset, model, mean_embeds, num_graphs, num_iter, cls_1, cls_2, cls_3, show_progress):
    lower_bound = 0.3
    upper_bound = 0.4
    target_size = 40
    iteration = 0
    while True:
        trainer = NewTrainer(
            sampler=(s := GraphSampler(
                max_nodes=20,
                temperature=0.15,
                num_node_cls=len(dataset.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(
                    classes=[cls_1, cls_2, cls_3], alpha=1, beta=1
                ), weight=25),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_3]), weight=0),
                # dict(key="logits", criterion=MeanPenalty(), weight=2),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                # dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=1),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=dataset,
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
            min_iteration=500
        )

        if trainer.train(
            iterations=2000,
            target_probs={cls_1: (lower_bound, upper_bound), cls_2: (lower_bound, upper_bound), cls_3: (lower_bound, upper_bound)},
            target_size=target_size,
            w_budget_init=1,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=32,
            show_progress=show_progress
        ):
            res = trainer.quantitative(sample_size=num_graphs)
            base_res = trainer.quantitative_baseline()

            return {
                "gnnboundary": [res["mean"][cls_1], res["std"][cls_1], res["mean"][cls_2], res["std"][cls_2], 
                                res["mean"][cls_3], res["std"][cls_3]], 
                "baseline": [base_res["mean"][cls_1], base_res["std"][cls_1], base_res["mean"][cls_2], 
                             base_res["std"][cls_2], base_res["mean"][cls_3], base_res["std"][cls_3]],
                "converged": True
            }
        
        else:
            lower_bound -= 0.033
            target_size += 10
            iteration += 1

            if iteration == 3:
                res = trainer.quantitative(sample_size=num_graphs)
                base_res = trainer.quantitative_baseline()

                return {
                    "gnnboundary": [res["mean"][cls_1], res["std"][cls_1], res["mean"][cls_2], res["std"][cls_2], 
                                    res["mean"][cls_3], res["std"][cls_3]], 
                    "baseline": [base_res["mean"][cls_1], base_res["std"][cls_1], base_res["mean"][cls_2], 
                                base_res["std"][cls_2], base_res["mean"][cls_3], base_res["std"][cls_3]],
                    "converged": True
                }

def gpuGenerate3Way(dataset, model, mean_embeds, num_graphs, num_iter, cls_1, cls_2, cls_3, show_progress):
    lower_bound = 0.3
    upper_bound = 0.4
    target_size = 40
    iteration = 0
    while True:
        trainer = GPUTrainer(
            sampler=(s := GraphSampler(
                max_nodes=20,
                temperature=0.15,
                num_node_cls=len(dataset.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(
                    classes=[cls_1, cls_2, cls_3], alpha=1, beta=1
                ), weight=25),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                # dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_3]), weight=0),
                # dict(key="logits", criterion=MeanPenalty(), weight=2),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                # dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=1),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=dataset,
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
            min_iteration=500
        )

        if trainer.train(
            iterations=2000,
            target_probs={cls_1: (lower_bound, upper_bound), cls_2: (lower_bound, upper_bound), cls_3: (lower_bound, upper_bound)},
            target_size=target_size,
            w_budget_init=1,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=32,
            show_progress=show_progress
        ):
            res = trainer.quantitative(sample_size=num_graphs)
            base_res = trainer.quantitative_baseline()

            return {
                "gnnboundary": [res["mean"][cls_1].item(), res["std"][cls_1].item(), res["mean"][cls_2].item(), res["std"][cls_2].item(), 
                                res["mean"][cls_3].item(), res["std"][cls_3].item()], 
                "baseline": [base_res["mean"][cls_1].item(), base_res["std"][cls_1].item(), base_res["mean"][cls_2].item(), 
                             base_res["std"][cls_2].item(), base_res["mean"][cls_3].item(), base_res["std"][cls_3].item()],
                "converged": True
            }
        
        else:
            lower_bound -= 0.033
            target_size += 10
            iteration += 1

            if iteration == 3:
                res = trainer.quantitative(sample_size=num_graphs)
                base_res = trainer.quantitative_baseline()

                return {
                    "gnnboundary": [res["mean"][cls_1].item(), res["std"][cls_1].item(), res["mean"][cls_2].item(), res["std"][cls_2].item(), 
                                res["mean"][cls_3].item(), res["std"][cls_3].item()], 
                    "baseline": [base_res["mean"][cls_1].item(), base_res["std"][cls_1].item(), base_res["mean"][cls_2].item(), 
                                base_res["std"][cls_2].item(), base_res["mean"][cls_3].item(), base_res["std"][cls_3].item()],
                    "converged": False
                }


def get_results_3way(model, adjacent_classes, dataset_name, seeds, show_progress=True, num_iter=2000, num_graphs=1000):
    """
    returns dictionary for each adjacent class pair

    the key values are class 1, class 2 and class 3
    the item values is an array with:

    [ gnn_mean_c1, gnn_std_c1, gnn_mean_c2, gnn_std_c2, gnn_mean_c3, gnn_std_c3, base_mean_c1, 
    base_std_c1, base_mean_c2, base_std_c2, base_mean_c3, base_std_c3, convergence_rate ]
    """
    results = dict()

    for seed in seeds:
        print(f"using seed {seed}")
        set_seed(seed)

        if dataset_name == "mrsc9":
            dataset = MSRCDataset(seed=seed)
            func = gpuGenerate3Way
        if dataset_name == "collab":
            dataset = CollabDataset(seed=seed)
            func = normalGenerate3Way
        if dataset_name == "enzymes":
            dataset = ENZYMESDataset(seed=seed)
            func = normalGenerate3Way
        if dataset_name == "motif":
            dataset = MotifDataset(seed=seed)
            func = normalGenerate3Way

        dataset_list_gt = dataset.split_by_class()
        mean_embeds = [d.model_transform(model, key="embeds").mean(dim=0) for d in dataset_list_gt]

        for adjacent_pair in adjacent_classes:
            generations = func(
                dataset, model, mean_embeds, num_graphs, num_iter, adjacent_pair[0], adjacent_pair[1], adjacent_pair[2], show_progress
            )


            if tuple(adjacent_pair) in results:
                results[tuple(adjacent_pair)] += np.array([generations["gnnboundary"][0], generations["gnnboundary"][1], 
                    generations["gnnboundary"][2], generations["gnnboundary"][3], generations["gnnboundary"][4], generations["gnnboundary"][5], 
                    generations["baseline"][0], generations["baseline"][1], generations["baseline"][2], generations["baseline"][3], 
                    generations["baseline"][4], generations["baseline"][5], int(generations["converged"])]) / len(seeds)
            else:
                results[tuple(adjacent_pair)] = np.array([generations["gnnboundary"][0], generations["gnnboundary"][1], 
                    generations["gnnboundary"][2], generations["gnnboundary"][3], generations["gnnboundary"][4], generations["gnnboundary"][5], 
                    generations["baseline"][0], generations["baseline"][1], generations["baseline"][2], generations["baseline"][3], 
                    generations["baseline"][4], generations["baseline"][5], int(generations["converged"])]) / len(seeds)

    return results
