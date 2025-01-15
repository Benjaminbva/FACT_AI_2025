import sys
import os
sys.path.append(os.path.abspath('../'))

from gnnboundary import *
import torch
import argparse
from lib.trainer import TrainerGPU

import warnings
warnings.filterwarnings("ignore")


DATASETS = ["collab","enzymes","motif"]
# not actually sure what the threshold is
THRESHOLD = 0.75



def generateBoundaryGraphs(dataset, model, mean_embeds, num_graphs, num_iter, cls_1, cls_2):
    c1 = []
    c2 = []

    for _ in range(num_graphs):
        trainer = {}
        sampler = {}

        trainer[cls_1, cls_2] = TrainerGPU(
            sampler=(s := GraphSampler(
                max_nodes=25,
                temperature=0.2,
                num_node_cls=len(dataset.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(
                    classes=[cls_1, cls_2], alpha=1, beta=2
                ), weight=25),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=1),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                # dict(key="xi", criterion=NormPenalty(order=1), weight=0),
                # dict(key="xi", criterion=NormPenalty(order=2), weight=0),
                # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
                # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=dataset,
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
        )

        trainer[cls_1, cls_2].train(
            iterations=num_iter,
            target_probs={cls_1: (0.4, 0.6), cls_2: (0.4, 0.6)},
            target_size=40,
            w_budget_init=1,
            w_budget_inc=1.1,
            w_budget_dec=0.95,
            k_samples=16
        )

        Graph, probs = trainer[cls_1, cls_2].evaluate(threshold=0.5, show=False, return_probs=True)
        c1.append(probs[cls_1])
        c2.append(probs[cls_2])

    c1 = torch.FloatTensor(c1)
    c2 = torch.FloatTensor(c2)

    return torch.mean(c1), torch.mean(c2), torch.std(c1), torch.std(c2)


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
        if dataset_name == "enzymes":
            dataset = ENZYMESDataset(seed=12345)

            model = GCNClassifier(node_features=len(dataset.NODE_CLS),
                num_classes=len(dataset.GRAPH_CLS),
                hidden_channels=32,
                num_layers=3)
        if dataset_name == "motif":
            dataset = MotifDataset(seed=12345)

            model = GCNClassifier(node_features=len(dataset.NODE_CLS),
                num_classes=len(dataset.GRAPH_CLS),
                hidden_channels=6,
                num_layers=3)

        print("done loading")

        
        print(dataset_name)
        model.load_state_dict(torch.load(f'ckpts/{dataset_name}.pt'))

        dataset_list_gt = dataset.split_by_class()
        dataset_list_pred = dataset.split_by_pred(model)
        mean_embeds = [d.model_transform(model, key="embeds").mean(dim=0) for d in dataset_list_gt]

        adj_ratio_mat, boundary_info = pairwise_boundary_analysis(model, dataset_list_pred)
        n_classes = len(adj_ratio_mat)
        adjacent_classes = list()

        for i in range(n_classes):
            for j in range (n_classes):
                if i != j and adj_ratio_mat[i][j] > THRESHOLD:
                    adjacent_classes.append(frozenset([i,j]))

        adjacent_classes = list(set(adjacent_classes))

        for adjacent_pair in adjacent_classes:
            adjacent_pair = list(adjacent_pair)

            mean_c1, mean_c2, std_c1, std_c2 = generateBoundaryGraphs(
                dataset, model, mean_embeds, args.num_graphs, args.num_iter, adjacent_pair[0], adjacent_pair[1]
            )

            print(f'c1 = {adjacent_pair[0]}')
            print(f'mean = {mean_c1.item()} std = {std_c2.item()}')
            print(f'c2 = {adjacent_pair[1]}')
            print(f'mean = {mean_c2.item()} std = {std_c2.item()}\n')

            f = open(f"logs/{dataset_name}.txt", "a")
            f.write(f'c1 = {adjacent_pair[0]}')
            f.write(f'mean = {mean_c1.item()} std = {std_c2.item()}')
            f.write(f'c2 = {adjacent_pair[1]}')
            f.write(f'mean = {mean_c2.item()} std = {std_c2.item()}\n')
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
