import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import to_networkx
from gnns.utils import binary_to_int



def partition_complexity(dataset, iterations=1):
    total_complexity = {i: [] for i in range(iterations)}

    for data in dataset:
        colors = binary_to_int(data.x)
        old = len(set(list(colors.numpy())))
        wl = torch_geometric.nn.conv.WLConv()
        for i in range(iterations):
            colors = wl(colors, data.edge_index)
            new = len(set(list(colors.numpy())))

            total_complexity[i].append(new / old if old > 0 else 0)
            old=new

    for key in total_complexity:
        total_complexity[key] = np.array(total_complexity[key]).mean()
    return total_complexity

'''
def weisfeiler_leman_1wl(G, iterations=3):
    """Refined WL algorithm."""
    # Initial labeling based on node features
    labels = {node: G.nodes[node]['feature'] for node in G.nodes()}

    # Initialize complexity tracking
    complexity = {}#{1: 1.0}  # Start with a baseline complexity ratio of 1.0

    old = len(set(labels.values()))  # Initial count of unique labels

    for i in range(iterations):
        # Relabel nodes based on neighbors' labels
        new_labels = {}
        for node in G.nodes():
            neighbors_labels = tuple(sorted([labels[neighbor] for neighbor in G.neighbors(node)]))
            new_labels[node] = hash((labels[node], neighbors_labels))

        labels = new_labels

        # Update complexity tracking
        new = len(set(labels.values()))
        complexity[i] = new / old if old > 0 else 0
        old = new  # Update old for the next iteration

    return labels, complexity


def assign_unique_ids(dataset):
    unique_vectors = {}
    next_id = 0
    for data in dataset:
        for feature_vector in data.x:
            vector_key = tuple(feature_vector.tolist())
            if vector_key not in unique_vectors:
                unique_vectors[vector_key] = next_id
                next_id += 1
    return unique_vectors


def partition_complexity(dataset, iterations=1):
    unique_vectors = assign_unique_ids(dataset)
    total_complexity = {i: [] for i in range(iterations)}

    for data in dataset:
        # Convert PyG data to NetworkX graph
        G = to_networkx(data, node_attrs=['x'])  # Ensure 'x' is included if you want to keep node features

        # Correctly assign unique IDs based on original features
        for node in G.nodes(data=True):
            original_feature = tuple(node[1]['x'])  # Extract original feature vector converted to tuple
            color = unique_vectors[original_feature]
            G.nodes[node[0]]['feature'] = color  # Assign unique ID based on original feature

        labels, complexity = weisfeiler_leman_1wl(G, iterations)
        for i in range(iterations):
            total_complexity[i].append(complexity[i])

    for key in total_complexity:
        total_complexity[key] = np.array(total_complexity[key]).mean()
    return total_complexity
'''