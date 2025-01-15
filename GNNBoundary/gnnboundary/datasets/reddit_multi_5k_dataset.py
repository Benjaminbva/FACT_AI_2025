import networkx as nx
import pandas as pd
import torch_geometric as pyg
import os
from gnn_xai_common.datasets import BaseGraphDataset
from gnn_xai_common.datasets.utils import default_ax, unpack_G


class RedditMultiDataset(BaseGraphDataset):
    NODE_CLS = {0: "user"}  # Changed to reflect that nodes are Reddit users
    # Classes are subreddits
    GRAPH_CLS = {
        0: "worldnews",
        1: "videos",
        2: "AdviceAnimals",
        3: "aww",
        4: "mildlyinteresting",
    }

    def __init__(
        self,
        *,
        name="REDDIT-MULTI-5K",
        url="https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/REDDIT-MULTI-5K.zip",
        **kwargs,
    ):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        return [
            "REDDIT-MULTI-5K/REDDIT-MULTI-5K_A.txt",
            "REDDIT-MULTI-5K/REDDIT-MULTI-5K_graph_indicator.txt",
            "REDDIT-MULTI-5K/REDDIT-MULTI-5K_graph_labels.txt",
        ]

    def download(self):
        # Skip downloading but use downloaded version
        zip_path = f"{self.raw_dir}/REDDIT-MULTI-5K.zip"
        print("zip_path: ", zip_path)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Expected dataset ZIP file at {zip_path}.")
        pyg.data.extract_zip(zip_path, self.raw_dir)

    def generate(self):
        edges = pd.read_csv(self.raw_paths[0], header=None).to_numpy(dtype=int) - 1
        graph_idx = (
            pd.read_csv(self.raw_paths[1], header=None)[0].to_numpy(dtype=int) - 1
        )
        graph_labels = (
            pd.read_csv(self.raw_paths[2], header=None)[0].to_numpy(dtype=int) - 1
        )

        super_G = nx.Graph(edges.tolist(), label=graph_labels)
        nx.set_node_attributes(super_G, 0, name="label")
        nx.set_node_attributes(super_G, dict(enumerate(graph_idx)), name="graph")
        return unpack_G(super_G)

    def analyze_graph_properties(self):
        """Analyzes structural properties of graphs for each genre."""
        import networkx as nx

        properties = {
            "worldnews": [],
            "videos": [],
            "AdviceAnimals": [],
            "aww": [],
            "mildlyinteresting": [],
        }

        # Iterate through the dataset using PyG's structure
        for i in range(len(self)):
            data = self.get(i)  # Get the data object for this graph

            # Convert PyG graph to NetworkX for analysis
            edge_index = data.edge_index.numpy()
            edges = list(zip(edge_index[0], edge_index[1]))
            G = nx.Graph(edges)

            # Get the label (subreddit)
            label = data.y.item()  # Convert tensor to integer
            subreddit = list(self.GRAPH_CLS.values())[label]

            # Calculate network metrics
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G)
            avg_degree = sum(dict(G.degree()).values()) / n_nodes
            try:
                clustering_coef = nx.average_clustering(G)
            except:
                clustering_coef = (
                    0  # Handle cases where clustering coefficient can't be computed
                )

            properties[subreddit].append(
                {
                    "nodes": n_nodes,
                    "edges": n_edges,
                    "density": density,
                    "avg_degree": avg_degree,
                    "clustering_coefficient": clustering_coef,
                }
            )

        return properties

    @default_ax
    def draw(self, G, pos=None, ax=None):
        pos = pos or nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=G.nodes, node_size=500)
        nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=6)

    def process(self):
        super().process()
