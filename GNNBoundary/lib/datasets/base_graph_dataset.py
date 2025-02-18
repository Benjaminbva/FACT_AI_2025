from abc import ABC, abstractmethod
import random
import copy

import networkx as nx
import numpy as np
import torch
from torch import nn
import torch_geometric as pyg
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
from tqdm.auto import tqdm

from .utils import default_ax


class GPUBaseGraphDataset(pyg.data.InMemoryDataset, ABC):

    DATA_ROOT = 'data'

    def __init__(self, *,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 seed=None):
        self.name = name
        self._seed_all(seed)
        super().__init__(root=f'{self.DATA_ROOT}/{name}',
                         transform=transform,
                         pre_transform=pre_transform,
                         pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _seed_all(self, seed):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def train_test_split(self, k=10):
        train = self[len(self)//k:]
        test = self[:len(self)//k]
        return train, test

    def split_by_class(self):
        return [self[self.data.y == i] for i in range(len(self.GRAPH_CLS))]

    @torch.no_grad()
    def split_by_pred(self, model, batch_size=32):
        model.eval()
        preds = []
        for batch in self.loader(batch_size=batch_size, shuffle=False):
            preds.append(model(batch)['logits'].argmax(dim=-1))
        preds = torch.cat(preds, dim=0)
        return [self[preds == i] for i in range(len(self.GRAPH_CLS))]

    def loader(self, *args, **kwargs):
        return pyg.data.DataLoader(self, *args, **kwargs)

    @property
    def processed_file_names(self):
        return [f'data_{self.seed}.pt' if self.seed is not None else 'data.pt']

    def convert(self, G, generate_label=False):
        if isinstance(G, list):
            return pyg.data.Batch.from_data_list([self.convert(g) for g in G])
        G = nx.convert_node_labels_to_integers(G)
        node_labels = [G.nodes[i]['label']
                       if 'label' in G.nodes[i] or not generate_label
                       else random.choice(list(self.NODE_CLS))
                       for i in G.nodes]
        if G.number_of_edges() > 0:
            if hasattr(self, "EDGE_CLS"):
                edge_labels = [G.edges[e]['label']
                               if 'label' in G.edges[e] or not generate_label
                               else random.choice(list(self.EDGE_CLS))
                               for e in G.edges]
                edge_index, edge_attr = pyg.utils.to_undirected(
                    torch.tensor(list(G.edges)).T,
                    torch.eye(len(self.EDGE_CLS))[edge_labels].float(),
                )
            else:
                edge_index, edge_attr = pyg.utils.to_undirected(
                    torch.tensor(list(G.edges)).T,
                ), None
        else:
            if hasattr(self, "EDGE_CLS"):
                edge_index, edge_attr = torch.empty(2, 0).long(), torch.empty(0, len(self.EDGE_CLS))
            else:
                edge_index, edge_attr = torch.empty(2, 0).long(), None
        return pyg.data.Data(
            G=G,
            x=torch.eye(len(self.NODE_CLS))[node_labels].float(),
            y=torch.tensor(G.graph['label'] if "label" in G.graph else -1).long(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    @abstractmethod
    def generate(self):
        raise NotImplementedError

    def spawn_from_G_list(self, G_list):
        data_list = list(map(self.convert, G_list))
        new_dataset = copy.copy(self)
        new_dataset.data, new_dataset.slices = self.collate(data_list)
        return new_dataset

    def process(self):
        data_list = list(map(self.convert, self.generate()))
        if self.seed is not None:
            random.shuffle(data_list)

        if self.pre_filter is not None:
            data_list = filter(self.pre_filter, data_list)

        if self.pre_transform is not None:
            data_list = map(self.pre_transform, data_list)

        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])

    @default_ax
    def draw(self, G, pos=None, ax=None):
        nx.draw(G, pos=pos or nx.kamada_kawai_layout(G), ax=ax)

    def show(self, idx, ax=None, **kwargs):
        data = self[idx]
        print(f"data: {data}")
        print(f"class: {self.GRAPH_CLS[data.G.graph['label']]}")
        self.draw(data.G, ax=ax, **kwargs)

    def describe(self):
        n = [data.G.number_of_nodes() for data in self]
        m = [data.G.number_of_edges() for data in self]
        return dict(mean_n=np.mean(n), mean_m=np.mean(m), std_n=np.std(n), std_m=np.std(m))

    def model_fit(self, model, batch_size=32, lr=0.01):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model.train()
        losses = []
        for batch in self.loader(batch_size=batch_size, shuffle=True):
            model.zero_grad()  # Clear gradients.
            out = model(batch)  # Perform a single forward pass.
            loss = criterion(out['logits'], batch.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def model_evaluate(self, model, batch_size=32):
        acc = Accuracy(task="multiclass", num_classes=len(self.GRAPH_CLS)).to(self.device)
        cm = ConfusionMatrix(task='multiclass', num_classes=len(self.GRAPH_CLS)).to(self.device)
        f1 = F1Score(task="multiclass", num_classes=len(self.GRAPH_CLS), average=None).to(self.device)
        model.eval()
        for batch in self.loader(batch_size=batch_size, shuffle=False):
            acc(model(batch)['logits'].to(self.device), batch.y.to(self.device))
            cm(model(batch)['logits'].to(self.device), batch.y.to(self.device))
            f1(model(batch)['logits'].to(self.device), batch.y.to(self.device))
        return {
            'acc': acc.compute().item(),
            'cm': cm.compute().cpu().numpy(),
            'f1': dict(zip(self.GRAPH_CLS.values(), f1.compute().tolist()))
        }

    @torch.no_grad()
    def model_transform(self, model, batch_size=32, key='logits'):
        embeds = []
        model.eval()
        for batch in self.loader(batch_size=batch_size, shuffle=False):
            embeds.append(model(batch)[key])
        return torch.cat(embeds, dim=0)
