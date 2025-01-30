import torch
import numpy as np

def boundary_thickness(graph_embedding,
                       boundary_graph_embedding,
                       model_scoring_function,
                       c1,
                       c2,
                       gamma=0.75,
                       num_points=50):

    """
    Args:
        graph_embedding (torch.Tensor): (embedding_dimension * num_graphs) output embedding from graph pooling layer
        boundary_graph_embedding (torch.Tensor): (embedding_dimension * num_graphs) output embedding from graph pooling
                                                 layer of boundary graph
        class_pair_idx (tuple(int)) : tuple of class pair idx
        model_scoring_function () : MLP layer after embedding layer
        gamma (float) : hyperparameter
        num_points (int) : number of points used for interpolation

    Returns:
        thickness (float) : boundary thickness margin
    """

    #shuffle data around
    num_samples = min(graph_embedding.shape[1], boundary_graph_embedding.shape[1])

    graph_embedding = graph_embedding[:, torch.randperm(graph_embedding.shape[1])]
    boundary_graph_embedding = boundary_graph_embedding[:, torch.randperm(boundary_graph_embedding.shape[1])]
    thickness = []

    for idx in range(num_samples):

        g1 = graph_embedding[:, idx]

        if boundary_graph_embedding.shape[1] > 1:
            g2 = boundary_graph_embedding[:, idx]
        else:
            g2 = boundary_graph_embedding[:, 0]

        ## We use l2 norm to measure distance
        dist = torch.norm(g1 - g2, p=2)

        new_batch = []

        ## Sample some points from each segment
        ## This number can be changed to get better precision

        for lmbd in np.linspace(0, 1.0, num=num_points):
            new_batch.append(g1 * lmbd + g2 * (1 - lmbd))
        new_batch = torch.stack(new_batch)

        y_new_batch = model_scoring_function(embeds=new_batch)['probs'].T

        #assuming that y_new_batch is off dimension (num_classes * num_samples)

        thickness.append(dist.item() * len(np.where(gamma > y_new_batch[c1, :] - y_new_batch[c2, :])[0]) / num_points)

    return np.mean(thickness)