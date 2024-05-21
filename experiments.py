import argparse
import json

import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform, OneHotDegree, Compose, RemoveIsolatedNodes

from gnns.utils import eval
from utils.bitflips import flip_bits_across_layers
from utils.io import generate_filename
from utils.isotests import partition_complexity
from gnns.models import GIN, GCN

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from itertools import combinations
import numpy as np
import networkx as nx

import torch.nn.functional as F


from operator import itemgetter
import copy
from gnns.transforms import *

import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from torch.nn.functional import one_hot




def main(args):
    target_bit_map = {'sign' : 1, 'exponent' : 9, 'mantissa': 22, 'all' : 32} #number of bits
    models = {'GIN' : GIN, 'GCN' : GCN}
    activations = {'sigmoid' : F.sigmoid, 'relu' : F.relu, 'silu' : F.silu, 'tanh' : F.tanh}
    featureless = ['IMDB-BINARY']
    if args.dataset in featureless:
        transform = T.Compose([T.ToUndirected(), T.RemoveIsolatedNodes(), AddBinaryEncodedDegree(max_degree=128)])
    else:
        transform = T.Compose([T.ToUndirected(), T.RemoveIsolatedNodes()])  # AddBinaryEncodedDegree(max_degree=128) --> for featureless graphs only
    dataset = TUDataset(root='/tmp/{}'.format(args.dataset), name=args.dataset, use_node_attr=False,
                        transform=transform, cleaned=True)


    num_features = dataset[0].num_node_features
    num_classes = len(torch.unique(torch.cat([g.y.unsqueeze(0) for g in dataset if g.y is not None], dim=0))) #len(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth = args.depth

    stats = {}
    stats ['meta'] = vars(args)

    complexity = list(partition_complexity(dataset, depth).values())
    stats['1wl_iter_subdiv_ratios'] = complexity
    stats['1wl_iter_subdiv_ratios_prod'] = np.prod(complexity)

    for model_i in range(args.num_clean): #num_models
        clean_key = 'model{}_clean'.format(model_i)
        stats[clean_key] = {}

        model = models[args.model](num_features, num_classes, depth, activations[args.activation])
        model = model.to(device)
        loss_fn = F.nll_loss

        results = eval(model, device, loss_fn, dataset, depth=depth)
        metrics = results[-3]

        total_bits = 0
        total_target_bits = 0
        sparsities = {}
        eps = torch.finfo(torch.float32).eps
        for name, parameter in model.named_parameters():
            if 'conv' in name:
                total_bits += parameter.shape[0] * parameter.shape[1] * target_bit_map['all']
                sparsities[name] = (1 - torch.count_nonzero(torch.clamp(parameter, min=parameter.min() + eps, max=parameter.max() - eps)) / parameter.numel()).detach().cpu().item()
            if 'nn.lins' in name or 'lin' in name:
                total_target_bits += parameter.shape[0] * parameter.shape[1] * target_bit_map[args.target]
        num_target_bits = int(total_target_bits * args.bit_flip_percentage)

        stats['meta']['homophily'] = results[-1]

        stats[clean_key]['total_bits'] = total_bits
        stats[clean_key]['total_target_bits'] = total_target_bits
        stats[clean_key]['num_target_bits'] = num_target_bits
        stats[clean_key]['eps'] = eps
        stats[clean_key]['accuracy'] = results[0]
        stats[clean_key]['distinguishable'] = results[-2]
        stats[clean_key]['sparsities'] = sparsities
        conv_subdiv_ratios = [metrics[key]['io_refinement'] for key in metrics if '01' in key]
        stats[clean_key]['conv_subdiv_ratios'] = conv_subdiv_ratios
        stats[clean_key]['conv_subdiv_ratios_prod'] = np.prod(conv_subdiv_ratios)
        stats[clean_key]['conv_out_ranks'] = [metrics[key]['o_rank'] for key in metrics if '01' in key]
        stats[clean_key]['conv_out_vars'] = [metrics[key]['o_var'] for key in metrics if '01' in key]
        
        stats[clean_key]['comb_mappings'] = [metrics[key]['io_refinement'] for key in metrics if '10' in key]
        stats[clean_key]['comb_out_ranks'] = [metrics[key]['o_rank'] for key in metrics if '10' in key]
        stats[clean_key]['comb_out_vars'] = [metrics[key]['o_var'] for key in metrics if '10' in key]

        stats[clean_key]['agg_out_ranks'] = [metrics[key]['o_rank'] for key in metrics if '00' in key]
        stats[clean_key]['agg_out_vars'] = [metrics[key]['o_var'] for key in metrics if '00' in key]

        clean_model = copy.deepcopy(model)

        conv_names = ['convs.{}'.format(d) for d in range(depth)]
        combos = []

        for r in range(1, min(args.combo_length+1, len(conv_names) + 1)):
            for combo in combinations(conv_names, r):
                combos.append(combo)
        for attack_i in range(args.num_dirty): #How many times to attack for each model
            dirty_key = 'model{}_dirty{}'.format(model_i, attack_i)
            stats[dirty_key] = {}
            for combo in combos:
                combo_key = ''.join(combo)
                stats[dirty_key][combo_key] = {}
                model = copy.deepcopy(clean_model)
                flipped_bits = flip_bits_across_layers(model, combo, num_target_bits, part=args.target)

                results = eval(model, device, loss_fn, dataset, depth=depth)
                metrics = results[-3]

                sparsities = {}
                eps = torch.finfo(torch.float32).eps
                for name, param in model.named_parameters():
                    if 'conv' in name:
                        with torch.no_grad():
                            sparsities[name] = (1 - torch.count_nonzero(torch.clamp(param, min=param.min() + eps, max=param.max() - eps)) / param.numel()).detach().cpu().item()

                stats[dirty_key][combo_key]['flipped_bits'] = flipped_bits
                stats[dirty_key][combo_key]['accuracy'] = results[0]
                stats[dirty_key][combo_key]['distinguishable'] = results[-2]
                stats[dirty_key][combo_key]['sparsities'] = sparsities
                conv_subdiv_ratios = [metrics[key]['io_refinement'] for key in metrics if '01' in key]
                stats[dirty_key][combo_key]['conv_subdiv_ratios'] = conv_subdiv_ratios
                stats[dirty_key][combo_key]['conv_subdiv_ratios_prod'] = np.prod(conv_subdiv_ratios)
                stats[dirty_key][combo_key]['conv_out_ranks'] = [metrics[key]['o_rank'] for key in metrics if '01' in key]
                stats[dirty_key][combo_key]['conv_out_vars'] = [metrics[key]['o_var'] for key in metrics if '01' in key]

                stats[dirty_key][combo_key]['comb_mappings'] = [metrics[key]['io_refinement'] for key in metrics if '10' in key]
                stats[dirty_key][combo_key]['comb_out_ranks'] = [metrics[key]['o_rank'] for key in metrics if '10' in key]
                stats[dirty_key][combo_key]['comb_out_vars'] = [metrics[key]['o_var'] for key in metrics if '10' in key]

                stats[dirty_key][combo_key]['agg_out_ranks'] = [metrics[key]['o_rank'] for key in metrics if '00' in key]
                stats[dirty_key][combo_key]['agg_out_vars'] = [metrics[key]['o_var'] for key in metrics if '00' in key]

    filename = generate_filename(args)

    # Writing JSON data
    with open('./results/'+filename, 'w') as f:
        json.dump(stats, f, indent=4)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script that takes command line arguments.")

    # Add arguments
    parser.add_argument("-d", "--depth", type=int, default=3, help="Depth of the model (default: 3).")
    parser.add_argument("-m", "--model", type=str, default="GIN", help="Name of the model to use (GIN, GCN).")
    parser.add_argument("-a", "--activation", type=str, default="relu",  help="Name of the activation to use (relu, sigmoid, silu, tanh).")
    parser.add_argument("-nc", "--num_clean", type=int, default=1, help="Number of clean models.")
    parser.add_argument("-nd", "--num_dirty", type=int, default=2, help="Number of dirty models per clean model.")
    parser.add_argument("-t", "--target", type=str, default='sign', help="Where flips should occur (sign, mantissa, exponent, all)")
    parser.add_argument("-cl", "--combo_length", type=int, default=1, help="Max length of layer combo (< depth)")
    parser.add_argument("-bfpc", "--bit_flip_percentage", type=float, default=0.25, help="Percentage of bits to flip in target combo [0.0, 1.0]")
    parser.add_argument("-ds", "--dataset", type=str, default="MUTAG", help="Name of the dataset to use.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
