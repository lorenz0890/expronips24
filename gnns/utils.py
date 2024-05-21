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

def train(model, optimizer, device, loss_fn, epochs, train_data):
    total_loss = []
    for epoch in range(epochs):
        epoch_loss = []
        model.train()
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(data)[0]  # Assuming individual graphs (not batches)
            out = out.unsqueeze(0) if out.dim() == 1 else out  # Ensure out has batch dimension
            data.y = data.y.unsqueeze(0) if data.y.dim() == 0 else data.y  # Ensure data.y has batch dimension

            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        total_loss.append(np.array(epoch_loss).mean())
    return total_loss


class LayerMetrics:
    def __init__(self):
        self.metrics = {'o_rank' : [], 'io_refinement': [], 'o_var' : []}

    def __call__(self, module, input, output):
        with torch.no_grad():
            #in_rank = torch.linalg.matrix_rank(input[0]).detach().cpu().item()
            out_rank = torch.linalg.matrix_rank(output).detach().cpu().item()
            #if out_rank == 0:
            #    print(out_rank, output)
            unique_rows, counts = torch.unique(output, dim=0, return_counts=True)
            num_unique_out_rows = unique_rows.shape[0]

            unique_rows, counts = torch.unique(input[0], dim=0, return_counts=True)
            num_unique_in_rows = unique_rows.shape[0]

            #in_var = torch.var(input[0]).detach().cpu().item()
            out_var = torch.var(output).detach().cpu().item()

            #if output.shape[0] == input[0].shape[0]:
            #    i = torch.distributions.normal.Normal(input[0].mean(), input[0].var())
            #    o = torch.distributions.normal.Normal(output.mean(), output.var())
            #    out_var = torch.distributions.kl.kl_divergence(i, o).detach().cpu().item()
            #else: out_var = 0
            self.metrics['io_refinement'].append(num_unique_out_rows/ num_unique_in_rows)
            ratio = out_var#out_var / in_var #if in_var > 0 else 0
            self.metrics['o_var'].append(ratio)
            ratio = out_rank# / in_rank# if in_rank > 0 else 0
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
    epsilon = torch.finfo(tensor_list[0].dtype).eps  # Machine epsilon for the tensor's dtype
    n = len(tensor_list)
    distinguishable = [True] * n  # Initially assume all tensors are distinguishable

    for i in range(n):
        for j in range(i + 1, n):
            if distinguishable[i] or distinguishable[j]:
                # Check if the absolute difference exceeds epsilon for any element
                if torch.any(torch.abs(tensor_list[i] - tensor_list[j]) > 0):
                    continue
                else:
                    # Mark tensors as indistinguishable if no element differs by more than epsilon
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
    #print(hooks.keys(), flush=True)
    #exit(-1)
    model.eval()

    # Evaluating
    total_loss = []
    correct_predictions = 0
    total_graphs = 0
    embeddings, homophily = [],[]
    for data in eval_data:
        data = data.to(device)

        # Homophily (feature based - double check the meaning of this!)
        #edge_index = data.edge_index.cpu()
        #node_features = data.x.cpu()
        #source_features = node_features[edge_index[0]].long()
        #target_features = node_features[edge_index[1]].long()
        #similarity_scores = (source_features & target_features).float().sum(dim=1)
        #homophily_score = (similarity_scores > 0).float().mean().item()

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

    #eps = torch.finfo(torch.float32).eps
    #serialized_tensors = [torch.clamp(t, min=t.min()+eps, max=t.max()-eps).numpy().tobytes() for t in tensor_cpu]
    #unique_serialized = set(serialized_tensors)
    #num_unique_vectors = len(unique_serialized)
    #print(tensor_cpu.shape)
    num_unique_vectors = count_distinguishable_tensors([t.squeeze() for t in tensor_cpu])

    #print(num_unique_vectors, len(eval_data))

    test_accuracy = correct_predictions / total_graphs
    #print([(hook.average_metric()) for name, hook in hooks.items()])
    metrics = {name: hook.average_metric() for name, hook in hooks.items()}
    for hook in hook_handles: hook.remove()

    return test_accuracy, total_loss, metrics, num_unique_vectors/len(eval_data), sum(homophily)/len(homophily)