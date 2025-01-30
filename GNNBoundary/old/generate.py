import sys
import os
sys.path.append(os.path.abspath('../'))

from gnnboundary import *
import torch
import argparse
from lib.trainer import NewTrainer
import numpy as np

import warnings
warnings.filterwarnings("ignore")


DATASETS = ["enzymes","motif","collab"]
THRESHOLD = 0.8



def generateBoundaryGraphs(dataset, model, mean_embeds, num_graphs, num_iter, cls_1, cls_2):
    while True:
        trainer = NewTrainer(
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
                dict(key="logits", criterion=MeanPenalty(), weight=1),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=1),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=dataset,
            budget_penalty=BudgetPenalty(budget=15, order=2, beta=1),
        )

        if trainer.train(
            iterations=2000,
            target_probs={cls_1: (0.45, 0.55), cls_2: (0.45, 0.55)},
            target_size=40,
            w_budget_init=1,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=32
        ):
            res = trainer.quantitative(sample_size=num_graphs)
            base_res = trainer.quantitative_baseline()

            G, probs = trainer.evaluate(threshold=0.5, show=False, return_probs=True)

            return {
                "gnnboundary": [res["mean"][cls_1], res["std"][cls_1], res["mean"][cls_2], res["std"][cls_2]], 
                "baseline": [base_res["mean"][cls_1], base_res["std"][cls_1], base_res["mean"][cls_2], base_res["std"][cls_2]]
            }
        
        G, probs = trainer.evaluate(threshold=0.5, show=False, return_probs=True)
        


def evaluate(args): 
    datasets = DATASETS if "all" in args.dataset else args.dataset
    for dataset_name in datasets:
        print(f'loading dataset {dataset_name}...')

        if dataset_name == "collab":
            dataset = CollabDataset(seed=12345)

            model = GCNClassifier(node_features=len(dataset.NODE_CLS),
                num_classes=len(dataset.GRAPH_CLS),
                hidden_channels=64,
                num_layers=5)
            
            model.load_state_dict(torch.load('ckpts/collab.pt'))

            adjacent_classes = [[0,1],[0,2]]
        if dataset_name == "enzymes":
            dataset = ENZYMESDataset(seed=12345)

            model = GCNClassifier(node_features=len(dataset.NODE_CLS),
                num_classes=len(dataset.GRAPH_CLS),
                hidden_channels=32,
                num_layers=3)
            
            model.load_state_dict(torch.load('ckpts/enzymes.pt'))

            adjacent_classes = [[0,3],[0,4],[0,5],[1,2],[3,4],[4,5]]
        if dataset_name == "motif":
            dataset = MotifDataset(seed=12345)

            model = GCNClassifier(node_features=len(dataset.NODE_CLS),
                num_classes=len(dataset.GRAPH_CLS),
                hidden_channels=6,
                num_layers=3)
            
            model.load_state_dict(torch.load('ckpts/motif.pt'))

            adjacent_classes = [[0,1],[0,2],[1,3]]

        print("done loading")

        # evaluation = dataset.model_evaluate(model)
        # np.savetxt(f'logs/confusion_{dataset_name}.txt', evaluation['cm'], fmt='%d')

        dataset_list_gt = dataset.split_by_class()
        dataset_list_pred = dataset.split_by_pred(model)
        mean_embeds = [d.model_transform(model, key="embeds").mean(dim=0) for d in dataset_list_gt]

        adj_ratio_mat, boundary_info = pairwise_boundary_analysis(model, dataset_list_pred)
        n_classes = len(adj_ratio_mat)
        # adjacent_classes = list()

        # adjacency matrix heavily dependant of the random seed. It can give different resulting adjacent pairs
        # hence the same adjacent pairs as the paper are used
        # np.savetxt(f'logs/adjacency_{dataset_name}.txt', adj_ratio_mat, fmt='%f')

        # for i in range(n_classes):
        #     for j in range (n_classes):
        #         if i != j and adj_ratio_mat[i][j] > THRESHOLD:
        #             adjacent_classes.append(frozenset([i,j]))

        # adjacent_classes = list(set(adjacent_classes))

        # print(adjacent_classes)

        for adjacent_pair in adjacent_classes:
            adjacent_pair = list(adjacent_pair)

            generations = generateBoundaryGraphs(
                dataset, model, mean_embeds, args.num_graphs, args.num_iter, adjacent_pair[0], adjacent_pair[1]
            )

            print(f'c1 = {adjacent_pair[0]}')
            print(f'mean = {generations["gnnboundary"][0].item()} std = {generations["gnnboundary"][1].item()}')
            print(f'c2 = {adjacent_pair[1]}')
            print(f'mean = {generations["gnnboundary"][2].item()} std = {generations["gnnboundary"][3].item()}\n')

            f = open(f"logs/quantitative_{dataset_name}.txt", "a")
            f.write(f'c1 = {adjacent_pair[0]}\n')
            f.write(f'gnnboundary mean = {generations["gnnboundary"][0].item()} std = {generations["gnnboundary"][1].item()}\n')
            f.write(f'baseline mean = {generations["baseline"][0].item()} std = {generations["baseline"][1].item()}\n')
            f.write(f'c2 = {adjacent_pair[1]}\n')
            f.write(f'gnnboundary mean = {generations["gnnboundary"][2].item()} std = {generations["gnnboundary"][3].item()}\n')
            f.write(f'baseline mean = {generations["baseline"][2].item()} std = {generations["baseline"][3].item()}\n\n')
            f.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--num_iter', default=2000, type=int,
                        help='Number of iterations')
    parser.add_argument('--num_graphs', default=500, type=int,
                        help='Number of graphs to be generated')
    parser.add_argument('--dataset', nargs='+', choices=["motif", "enzymes", "collab", "all"], default=["all"]
                        , help='List of datasets to use')

    args = parser.parse_args()

    evaluate(args)
