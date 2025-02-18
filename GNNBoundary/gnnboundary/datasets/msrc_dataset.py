import networkx as nx
import pandas as pd
import torch_geometric as pyg

from gnn_xai_common.datasets import BaseGraphDataset
from gnn_xai_common.datasets.utils import default_ax, unpack_G

from lib.datasets.base_graph_dataset import GPUBaseGraphDataset

class MSRCDataset(GPUBaseGraphDataset):

    NODE_CLS = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
    }

    NODE_COLOR = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'yellow',
        4: 'purple',
        5: 'orange',
        6: 'pink',
        7: 'brown',
        8: 'cyan',
        9: 'magenta',
    }


    GRAPH_CLS = {
        0: 'C1',
        1: 'C2',
        2: 'C3',
        3: "C4",
        4: "C5",
        5: "C6",
        6: "C7",
        7: "C8",
    }

    def __init__(self, *,
                 name='MSRC_9',
                # url="https://www.chrsmrrs.com/graphkerneldatasets/COLLAB.zip",
                url="https://www.chrsmrrs.com/graphkerneldatasets/MSRC_9.zip",
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        return ["MSRC_9/MSRC_9_A.txt",
                "MSRC_9/MSRC_9_graph_indicator.txt",
                "MSRC_9/MSRC_9_graph_labels.txt",
                "MSRC_9/MSRC_9_node_attributes.txt",
                "MSRC_9/MSRC_9_node_labels.txt"]

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_zip(f'{self.raw_dir}/MSRC_9.zip', self.raw_dir)

    def generate(self):
        edges = pd.read_csv(self.raw_paths[0], header=None).to_numpy(dtype=int) - 1
        graph_idx = pd.read_csv(self.raw_paths[1], header=None)[0].to_numpy(dtype=int) - 1
        graph_labels = pd.read_csv(self.raw_paths[2], header=None)[0].to_numpy(dtype=int) - 1
        node_labels = pd.read_csv(self.raw_paths[4], header=None)[0].to_numpy(dtype=int) - 1
        super_G = nx.Graph(edges.tolist(), label=graph_labels)
        nx.set_node_attributes(super_G, dict(enumerate(node_labels)), name='label')
        nx.set_node_attributes(super_G, dict(enumerate(graph_idx)), name='graph')
        return unpack_G(super_G)

    # TODO: use EDGE_WIDTH
    @default_ax
    def draw(self, G, pos=None, label=False, ax=None):
        pos = pos or nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               ax=ax,
                               nodelist=G.nodes,
                               node_size=500,
                               node_color=[
                                   self.NODE_COLOR[G.nodes[v]['label']]
                                   for v in G.nodes
                               ])
        if label:
            nx.draw_networkx_labels(G, pos,
                                    ax=ax,
                                    labels={
                                        v: self.NODE_CLS[G.nodes[v]['label']]
                                        for v in G.nodes
                                    })
        nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=6)


    def process(self):
        super().process()