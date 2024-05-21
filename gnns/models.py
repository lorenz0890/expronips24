import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP ,GINConv #GCNConv
from torch_geometric.nn import global_add_pool

from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm

from gnns.convs import GCNConv



class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4, activation = F.sigmoid):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()  # Use ModuleList to hold GINConv layers

	# Add GIN layers
        for _ in range(num_layers):
            mlp = MLP([num_features, num_features, num_features], batch_norm=False, bias=False, plain_last=False, act=activation)# plain_last = False for logging!
            self.convs.append(GINConv(mlp))

        self.fc1 = torch.nn.Linear(num_features, num_classes) #*num_layers
        #self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, graph):
        x, edge_index, batch = graph.x.cuda(), graph.edge_index.cuda(), graph.batch


        # Apply each GINConv layer in a loop
        for conv in self.convs:
            x = conv(x, edge_index)#.relu()
        xr = global_add_pool(x, batch)  # Pooling for graph-level classification
        x = self.fc1(xr)
        #x = self.drop(x)
        return F.log_softmax(x, dim=1), xr


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=4, activation=F.relu):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Add GCN layers
        for _ in range(num_layers):
            self.convs.append(GCNConv(num_features, num_features, activation=activation, bias=False))

        self.fc = torch.nn.Linear(num_features, num_classes)
        #self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x.cuda(), data.edge_index.cuda()

        for conv in self.convs:
            x = conv(x, edge_index) #Relu included in layer for metrics
        xr = global_add_pool(x, data.batch)  # Pooling for graph-level classification
        #x = F.dropout(xr, p=0.5, training=self.training)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), xr
