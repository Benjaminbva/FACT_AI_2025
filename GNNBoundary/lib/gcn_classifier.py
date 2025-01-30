import torch
from torch import nn
import torch_geometric as pyg
import torch.nn.functional as F

from .nn.functional import smooth_maximum_weight_propagation, global_sum_pool_weighted, global_mean_pool_weighted


class MultiGCNClassifier(nn.Module):
    def __init__(self, hidden_channels, node_features, num_classes, num_layers=3, dropout=0):
        super().__init__()
        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1 = pyg.nn.GCN(in_channels=node_features,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               act=nn.LeakyReLU(inplace=True),
                               dropout=dropout).to(self.dev)
        self.conv2 = pyg.nn.GCN(in_channels=hidden_channels,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               act=nn.LeakyReLU(inplace=True),
                               dropout=dropout).to(self.dev)
        self.drop = nn.Dropout(p=dropout).to(self.dev)
        self.lin = pyg.nn.Linear(hidden_channels*2, hidden_channels).to(self.dev)
        self.out = pyg.nn.Linear(hidden_channels, num_classes).to(self.dev)


    def forward(self, batch=None, embeds=None, embeds_last=None, edge_weight=None, temperature=0.05):
        if embeds_last is None:
            if embeds is None:
                node_weight = (None if edge_weight is None
                               else smooth_maximum_weight_propagation(batch.edge_index, edge_weight,
                                                                      size=len(batch.x),
                                                                      temperature=temperature))        
    
                # 1. Obtain node embeddings
                h = self.conv1(batch.x.to(self.dev), batch.edge_index.to(self.dev), edge_weight=edge_weight)
                # h = self.conv2(h,batch.edge_index.to(self.dev), edge_weight=edge_weight)

                # 2. Readout layer
                embeds = torch.cat([
                    global_sum_pool_weighted(h, batch=batch.batch.to(self.dev), node_weight=node_weight),
                    global_mean_pool_weighted(h, batch=batch.batch.to(self.dev), node_weight=node_weight),
                ], dim=1)

            h = self.drop(embeds)
            h = self.lin(h)
            embeds_last = h.relu()

        h = self.out(embeds_last)
        x = dict(logits=h, probs=F.softmax(h, dim=-1), embeds=embeds, embeds_last=embeds_last)

        return x
