from collections import namedtuple
import random
import numpy as np
from itertools import combinations
import numpy as np
from itertools import combinations


def triplet_boundary_analysis(model, dataset_list, key="embeds_last", k=100, n=20):
    """
    Analyzes boundaries between triplets of classes.

    Args:
        model: The model to analyze
        dataset_list: List of datasets for each class
        key: Key for embeddings
        k: Number of random triplet samples
        n: Number of interpolation points per edge
    """
    c = len(dataset_list)
    # Create a 3D tensor to store triplet adjacency ratios
    adj_ratio_tensor = np.zeros((c, c, c))
    result_dict = {}

    # Analyze all possible triplets of classes
    for i, j, l in combinations(range(c), 3):
        result = three_way_boundary_analysis(
            model, dataset_list[i], dataset_list[j], dataset_list[l], key=key, k=k, n=n
        )
        # Store results for each triplet
        result_dict[(i, j, l)] = result
        adj_ratio_tensor[i, j, l] = result.adj_ratio

    return adj_ratio_tensor, result_dict


def three_way_boundary_analysis(
    model, dataset_1, dataset_2, dataset_3, key="embeds_last", k=100, n=20
):
    """
    Analyzes the boundary region between three classes using barycentric interpolation.
    """
    ThreeWayBoundaryResult = namedtuple(
        "ThreeWayBoundaryResult",
        ["adj_ratio", "interp_points", "bound_results", "is_three_way_connected"],
    )

    is_three_way_connected = []
    interp_points = []
    bound_results = []

    # Get embeddings for each class
    embeds_1 = dataset_1.model_transform(model, key=key)
    embeds_2 = dataset_2.model_transform(model, key=key)
    embeds_3 = dataset_3.model_transform(model, key=key)

    for _ in range(k):
        # Sample one point from each class
        sample1 = random.choice(embeds_1)
        sample2 = random.choice(embeds_2)
        sample3 = random.choice(embeds_3)

        # Create a triangular grid of interpolation points using barycentric coordinates
        batch_points = []
        min_diff = 1
        best_bound_result = None

        for i in range(n):
            for j in range(n - i):
                # Barycentric coordinates
                a = i / (n - 1)
                b = j / (n - 1)
                c = 1 - a - b

                # Interpolated point
                point = a * sample1 + b * sample2 + c * sample3
                result = model(**{key: point})

                # Check if point is near three-way boundary
                top_probs = result["probs"].sort(descending=True)[0][:3]
                prob_diffs = top_probs.diff().abs()
                max_diff = prob_diffs.max().item()

                if max_diff < min_diff:
                    min_diff = max_diff
                    best_bound_result = result

                batch_points.append(result["logits"].argmax().item())

        # Check if there's a region where all three classes meet
        unique_classes = np.unique(batch_points)
        is_three_way = len(unique_classes) == 3

        interp_points.append(batch_points)
        bound_results.append(best_bound_result)
        is_three_way_connected.append(is_three_way)

    return ThreeWayBoundaryResult(
        adj_ratio=np.mean(is_three_way_connected),
        interp_points=interp_points,
        bound_results=bound_results,
        is_three_way_connected=is_three_way_connected,
    )


def find_top_triplets(adj_tensor, threshold=0.3):
    """
    Finds and prints high-scoring class triplets from adjacency tensor in a single line.

    Args:
        adj_tensor: 3D numpy array containing class adjacency scores
        threshold: Minimum score to consider (default: 0.3)
    """
    n_classes = adj_tensor.shape[0]
    triplet_scores = []

    # Check all possible triplet combinations
    for i, j, k in combinations(range(n_classes), 3):
        # Get scores between pairs in the triplet
        scores = [
            adj_tensor[i][j][k],
            adj_tensor[i][k][j],
            adj_tensor[j][i][k],
            adj_tensor[j][k][i],
            adj_tensor[k][i][j],
            adj_tensor[k][j][i],
        ]
        score = max(scores)  # Take max since these are alternative paths

        if score >= threshold:
            triplet_scores.append({"classes": (i, j, k), "score": score})

    # Sort by score in descending order
    triplet_scores.sort(key=lambda x: x["score"], reverse=True)

    # Print in single line with desired format
    output = " ".join(
        f"Classes {entry['classes']}: Score {entry['score']:.3f}"
        for entry in triplet_scores
    )

    print(output)
    return triplet_scores
