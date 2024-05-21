import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import Aggregation, MeanAggregation, SumAggregation

def binary_to_int(binary_vector):
    # Assuming binary_vectors is a 2D tensor with shape [num_nodes, feature_size]
    # Calculate powers of 2 for each bit position
    num_bits = binary_vector.size(1)
    powers_of_two = 2 ** torch.arange(num_bits - 1, -1, -1).to(binary_vector.device).to(torch.float32)

    # Use matmul to perform the binary to integer conversion for the whole batch
    integer_values = torch.matmul(binary_vector.to(torch.float32), powers_of_two).to(torch.long)
    return integer_values


class LayerMetrics:
    def __init__(self):
        self.metrics = {'o_rank' : [], 'io_refinement': [], 'o_var' : []}

    def __call__(self, module, input, output):
        with torch.no_grad():

            out_rank = torch.linalg.matrix_rank(output).detach().cpu().item()

            unique_rows, counts = torch.unique(output, dim=0, return_counts=True)
            num_unique_out_rows = unique_rows.shape[0]

            unique_rows, counts = torch.unique(input[0], dim=0, return_counts=True)
            num_unique_in_rows = unique_rows.shape[0]


            out_var = torch.var(output).detach().cpu().item()


            self.metrics['io_refinement'].append(num_unique_out_rows/ num_unique_in_rows)
            ratio = out_var
            self.metrics['o_var'].append(ratio)
            ratio = out_rank
            self.metrics['o_rank'].append(ratio)

    def average_metric(self):
        # Calculate the average rank
        for key in self.metrics:
            self.metrics[key] = np.array(self.metrics[key]).mean()
        return self.metrics


def encode_features_to_labels(node_features):
    unique_features, labels = np.unique(node_features, axis=0, return_inverse=True)
    return torch.tensor(labels, dtype=torch.long)


def get_feature_based_label_homophily_score(dataset):
    homophily_scores = []
    for data in dataset:
        node_features = data.x.cpu().numpy()
        labels = encode_features_to_labels(node_features)
        edge_index = data.edge_index.cpu()
        source_labels = labels[edge_index[0]]
        target_labels = labels[edge_index[1]]

        similarity_scores = (source_labels == target_labels).float()
        homophily_score = similarity_scores.mean().item()
        homophily_scores.append(homophily_score)

    return np.mean(homophily_scores), np.std(homophily_scores)


def count_distinguishable_tensors(tensor_list):
    #epsilon = torch.finfo(tensor_list[0].dtype).eps  # Machine epsilon for the tensor's dtype
    n = len(tensor_list)
    distinguishable = [True] * n  # Initially assume all tensors are distinguishable

    for i in range(n):
        for j in range(i + 1, n):
            if distinguishable[i] or distinguishable[j]:
                if torch.any(torch.abs(tensor_list[i] - tensor_list[j]) > 0):
                    continue
                else:
                    # Mark tensors as indistinguishable
                    distinguishable[i] = distinguishable[j] = False

    # Count the number of true entries in distinguishable
    return sum(distinguishable)

def eval(model, device, loss_fn, eval_data, depth=5):
    hooks = {}
    hook_handles = []
    for name, module in model.named_modules():
        #print(name)
        is_conv = name in ['convs.{}'.format(i) for i in range(0, depth)]
        is_mlp = name in ['convs.{}.nn'.format(i) for i in range(0, depth)]
        is_linear = name in ['convs.{}.lin'.format(i) for i in range(0, depth)]
        is_aggr = name in ['convs.{}.aggr_module'.format(i) for i in range(0, depth)]
        #print(name) #measure propagate/aggregation variance?
        if is_conv or is_mlp or is_linear or is_aggr:
            hook = LayerMetrics()
            hook_handles.append(module.register_forward_hook(hook))
            hooks['{}_{}{}'.format(name, int(is_mlp or is_linear), int(is_conv))] = hook

    model.eval()

    # Evaluating
    total_loss = []
    correct_predictions = 0
    total_graphs = 0
    embeddings, homophily = [],[]
    for data in eval_data:
        data = data.to(device)
        
        node_features = data.x.cpu().numpy()
        labels = encode_features_to_labels(node_features)
        edge_index = data.edge_index.cpu()
        source_labels = labels[edge_index[0]]
        target_labels = labels[edge_index[1]]
        similarity_scores = (source_labels == target_labels).float()
        homophily_score = similarity_scores.mean().item()
        homophily.append(homophily_score)

        # Forward pass
        with torch.no_grad():
            out, xr = model(data)  # Assuming individual graphs (not batches)
        out = out.unsqueeze(0) if out.dim() == 1 else out  # Ensure out has batch dimension
        data.y = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y  # Ensure data.y has batch dimension

        loss = loss_fn(out, data.y)
        total_loss.append(loss.item())

        embeddings.append(xr)
        # Compute accuracy
        _, predicted = torch.max(out, dim=1)
        correct_predictions += (predicted == data.y).sum().item()
        total_graphs += data.y.size(0)
    total_loss = np.array(total_loss).mean()

    tensor = torch.stack(embeddings)
    tensor_cpu = tensor.cpu()

    num_unique_vectors = count_distinguishable_tensors([t.squeeze() for t in tensor_cpu])


    test_accuracy = correct_predictions / total_graphs

    metrics = {name: hook.average_metric() for name, hook in hooks.items()}
    for hook in hook_handles: hook.remove()

    return test_accuracy, total_loss, metrics, num_unique_vectors/len(eval_data), sum(homophily)/len(homophily)
