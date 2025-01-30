import sys
import os
sys.path.append(os.path.abspath('../'))

from gnnboundary import *
import torch
import argparse
from lib.trainer import GPUTrainer
from lib.gcn_classifier import MultiGCNClassifier
from gnnboundary.datasets.msrc_dataset import MSRCDataset
import numpy as np

import warnings
warnings.filterwarnings("ignore")


DATASETS = ["enzymes","motif","collab","mrsc9"]
THRESHOLD = 0.8



def generateBoundaryGraphs(dataset, model, mean_embeds, num_graphs, num_iter, cls_1, cls_2, cls_3):
    lower_bound = 0.266
    upper_bound = 0.4
    target_size = 40
    iteration = 0
    while True:
        trainer = GPUTrainer(
            sampler=(s := GraphSampler(
                max_nodes=25,
                temperature=0.15,
                num_node_cls=len(dataset.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(
                    classes=[cls_1, cls_2], alpha=1, beta=1
                ), weight=5),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_3]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=1),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=1),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=dataset,
            budget_penalty=BudgetPenalty(budget=15, order=2, beta=1),
            min_iteration=500
        )

        if trainer.train(
            iterations=2000,
            target_probs={cls_1: (lower_bound, upper_bound), cls_2: (lower_bound, upper_bound), cls_3: (lower_bound, upper_bound)},
            target_size=target_size,
            w_budget_init=1,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=32
        ):
            res = trainer.quantitative(sample_size=num_graphs)
            base_res = trainer.quantitative_baseline()

            G, probs = trainer.evaluate(threshold=0.5, show=False, return_probs=True)
            print(probs)

            return {
                "gnnboundary": [res["mean"][cls_1], res["std"][cls_1], res["mean"][cls_2], res["std"][cls_2], res["mean"][cls_3], res["std"][cls_3]], 
                "baseline": [base_res["mean"][cls_1], base_res["std"][cls_1], base_res["mean"][cls_2], 
                             base_res["std"][cls_2], base_res["mean"][cls_3], base_res["std"][cls_3]],
                "converged": True
            }
        
        else:
            lower_bound -= 0.033
            target_size += 5
            iteration += 1
            print(lower_bound, target_size, iteration)

            if iteration == 3:
                res = trainer.quantitative(sample_size=num_graphs)
                base_res = trainer.quantitative_baseline()

                return {
                    "gnnboundary": [res["mean"][cls_1], res["std"][cls_1], res["mean"][cls_2], res["std"][cls_2], res["mean"][cls_3], res["std"][cls_3]], 
                    "baseline": [base_res["mean"][cls_1], base_res["std"][cls_1], base_res["mean"][cls_2], 
                                base_res["std"][cls_2], base_res["mean"][cls_3], base_res["std"][cls_3]],
                    "converged": False
            }
        
def evaluate(args): 
    datasets = DATASETS if "all" in args.dataset else args.dataset
    for dataset_name in datasets:
        print(f'loading dataset {dataset_name}...')

        if dataset_name == "mrsc9":
            dataset = MSRCDataset(seed=12345)

            model = MultiGCNClassifier(node_features=len(dataset.NODE_CLS),
                num_classes=len(dataset.GRAPH_CLS),
                hidden_channels=16,
                num_layers=5)
            
            model.load_state_dict(torch.load('ckpts/msrc_9.pt'))

            adjacent_classes = [[5, 6, 7], [0, 2, 4], [3, 5, 7], [1, 2, 6], [1, 2, 3], [2, 3, 7],
                                [1, 2, 7], [2, 5, 7], [0, 4, 6], [1, 3, 7], [0, 2, 7]]

        print("done loading")

        dataset_list_gt = dataset.split_by_class()
        dataset_list_pred = dataset.split_by_pred(model)
        mean_embeds = [d.model_transform(model, key="embeds").mean(dim=0) for d in dataset_list_gt]


        for adjacent_pair in adjacent_classes:
            adjacent_pair = list(adjacent_pair)

            generations = generateBoundaryGraphs(
                dataset, model, mean_embeds, args.num_graphs, args.num_iter, adjacent_pair[0], adjacent_pair[1], adjacent_pair[2]
            )

            print(f'c1 = {adjacent_pair[0]}')
            print(f'mean = {generations["gnnboundary"][0].item()} std = {generations["gnnboundary"][1].item()}')
            print(f'c2 = {adjacent_pair[1]}')
            print(f'mean = {generations["gnnboundary"][2].item()} std = {generations["gnnboundary"][3].item()}')
            print(f'c2 = {adjacent_pair[2]}')
            print(f'mean = {generations["gnnboundary"][4].item()} std = {generations["gnnboundary"][5].item()}\n')

            f = open(f"logs/quantitative_{dataset_name}.txt", "a")
            f.write(f'converged: {generations["converged"]}\n')
            f.write(f'c1 = {adjacent_pair[0]}\n')
            f.write(f'gnnboundary mean = {generations["gnnboundary"][0].item()} std = {generations["gnnboundary"][1].item()}\n')
            f.write(f'baseline mean = {generations["baseline"][0].item()} std = {generations["baseline"][1].item()}\n')
            f.write(f'c2 = {adjacent_pair[1]}\n')
            f.write(f'gnnboundary mean = {generations["gnnboundary"][2].item()} std = {generations["gnnboundary"][3].item()}\n')
            f.write(f'baseline mean = {generations["baseline"][2].item()} std = {generations["baseline"][3].item()}\n')
            f.write(f'c3 = {adjacent_pair[2]}\n')
            f.write(f'gnnboundary mean = {generations["gnnboundary"][4].item()} std = {generations["gnnboundary"][5].item()}\n')
            f.write(f'baseline mean = {generations["baseline"][4].item()} std = {generations["baseline"][5].item()}\n\n')
            f.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--num_iter', default=2000, type=int,
                        help='Number of iterations')
    parser.add_argument('--num_graphs', default=500, type=int,
                        help='Number of graphs to be generated')
    parser.add_argument('--dataset', nargs='+', choices=["mrsc9"], default=["mrsc9"]
                        , help='List of datasets to use')

    args = parser.parse_args()

    evaluate(args)
