import math
import random

import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected, add_self_loops, degree, one_hot


class AddBinaryEncodedDegree(BaseTransform):
    def __init__(self, max_degree=None):
        self.max_degree = max_degree

    def degree_to_binary(self, degree, max_length):
        """Converts a degree value to a binary tensor with padding."""
        binary = torch.tensor([int(x) for x in bin(degree)[2:]], dtype=torch.float)
        # Pad the binary representation to have the same length
        if len(binary) < max_length:
            padding = torch.zeros(max_length - len(binary), dtype=torch.float)
            binary = torch.cat((padding, binary), dim=0)
        return binary

    def __call__(self, data):
        # Compute the degree of each node
        d = degree(data.edge_index[0], dtype=torch.long)

        # Determine the max length for binary representation
        if self.max_degree is None:
            self.max_degree = d.max().item()
        max_length = len(bin(self.max_degree)[2:])

        # Convert and pad each degree to binary
        binary_degrees = torch.stack([self.degree_to_binary(deg, max_length) for deg in d])

        # Concatenate with existing node features, if they exist
        data.x = torch.cat([data.x, binary_degrees], dim=1) if data.x is not None else binary_degrees
        return data

    def __repr__(self):
        return '{}(max_degree={})'.format(self.__class__.__name__, self.max_degree)

