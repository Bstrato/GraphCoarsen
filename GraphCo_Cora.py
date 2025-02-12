import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from collections import defaultdict
import random
import itertools
from datetime import datetime
import json
from sklearn.metrics import precision_recall_fscore_support


class GraphCoarsenLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_samples, aggregation='mean', coarsen_aggregation='mean'):
        super().__init__()
        self.num_samples = num_samples
        self.aggregation = aggregation
        self.coarsen_aggregation = coarsen_aggregation


        self.weight_self = nn.Linear(in_channels, out_channels)
        self.weight_neigh = nn.Linear(in_channels, out_channels)
        self.weight_coarsen = nn.Linear(in_channels, out_channels)

    def aggregate(self, x, indices, method):
        if len(indices) == 0:
            return torch.zeros(1, x.size(1), device=x.device)

        features = x[indices]
        if method == 'mean':
            return torch.mean(features, dim=0, keepdim=True)
        elif method == 'sum':
            return torch.sum(features, dim=0, keepdim=True)
        elif method == 'max':
            return torch.max(features, dim=0, keepdim=True)[0]
        elif method == 'min':
            return torch.min(features, dim=0, keepdim=True)[0]
        elif method == 'median':
            return torch.median(features, dim=0, keepdim=True)[0]
        else:
            raise ValueError(f"Unsupported aggregation: {method}")

    def forward(self, x, adj_dict):
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.weight_self.out_features, device=x.device)

        for node in range(num_nodes):
            neighbors = adj_dict[node]
            if not neighbors:
                out[node] = self.weight_self(x[node:node + 1]).squeeze(0)
                continue

            if len(neighbors) > self.num_samples:
                sampled_neighbors = random.sample(neighbors, self.num_samples)
            else:
                sampled_neighbors = neighbors

            unsampled_neighbors = list(set(neighbors) - set(sampled_neighbors))

            self_emb = self.weight_self(x[node:node + 1])
            sampled_agg = self.aggregate(x, sampled_neighbors, self.aggregation)
            neigh_emb = self.weight_neigh(sampled_agg)
            coarsened_agg = self.aggregate(x, unsampled_neighbors, self.coarsen_aggregation)
            coarsen_emb = self.weight_coarsen(coarsened_agg)

            out[node] = (self_emb + neigh_emb + coarsen_emb).squeeze(0)

        return out


class GraphCoarsen(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, num_layers, num_samples,
                 aggregation='mean', coarsen_aggregation='mean', dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.aggregation = aggregation
        self.coarsen_aggregation = coarsen_aggregation

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.layers.append(
                GraphCoarsenLayer(in_dim, out_dim, num_samples, aggregation, coarsen_aggregation)
            )

        self.layer_norms = nn.ModuleList([
                                             nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
                                         ] + [nn.LayerNorm(output_dim)])

    def to_adj_dict(self, edge_index):
        adj_dict = defaultdict(list)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            adj_dict[src].append(dst)
            adj_dict[dst].append(src)
        return adj_dict

    def forward(self, x, edge_index):
        adj_dict = self.to_adj_dict(edge_index)

        for i, layer in enumerate(self.layers):
            x = layer(x, adj_dict)
            x = self.layer_norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return F.log_softmax(x, dim=1)


def calculate_metrics(pred, true, mask):
    """Calculate accuracy, precision, and recall for the given predictions"""
    pred_masked = pred[mask].cpu().numpy()
    true_masked = true[mask].cpu().numpy()

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_masked, pred_masked, average='macro', zero_division=0
    )


    accuracy = (pred_masked == true_masked).mean()

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def train_and_evaluate(model, data, optimizer, criterion, epochs=100):
    device = data.x.device
    model = model.to(device)
    best_val_acc = 0
    best_model = None
    training_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)

                # Calculate comprehensive metrics
                train_metrics = calculate_metrics(pred, data.y, data.train_mask)
                val_metrics = calculate_metrics(pred, data.y, data.val_mask)

                epoch_results = {
                    'epoch': epoch,
                    'loss': float(loss.item()),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                training_history.append(epoch_results)

                print(f'Epoch {epoch:03d}:')
                print(f'Loss: {loss:.4f}')
                print(f'Train - Acc: {train_metrics["accuracy"]:.4f}, '
                      f'Prec: {train_metrics["precision"]:.4f}, '
                      f'Rec: {train_metrics["recall"]:.4f}, '
                      f'F1: {train_metrics["f1"]:.4f}')
                print(f'Val   - Acc: {val_metrics["accuracy"]:.4f}, '
                      f'Prec: {val_metrics["precision"]:.4f}, '
                      f'Rec: {val_metrics["recall"]:.4f}, '
                      f'F1: {val_metrics["f1"]:.4f}')

                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_model = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_model is not None:
        model.load_state_dict(best_model)
        model = model.to(device)


    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_metrics = calculate_metrics(pred, data.y, data.test_mask)

    return test_metrics, best_val_acc, training_history


def run_aggregation_experiments():
    # Load dataset
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]

    # Model parameters
    feature_dim = data.x.size(1)
    hidden_dim = 64
    output_dim = dataset.num_classes
    num_layers = 2
    num_samples = 10

    # Aggregation methods to test
    agg_methods = ['mean', 'sum', 'max', 'min', 'median']

    # Results storage
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Test all combinations of aggregation methods
    for sampled_agg, coarsen_agg in itertools.product(agg_methods, agg_methods):
        print(f"\nTesting combination - Sampled: {sampled_agg}, Coarsened: {coarsen_agg}")
        print("-" * 50)

        model = GraphCoarsen(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_samples=num_samples,
            aggregation=sampled_agg,
            coarsen_aggregation=coarsen_agg,
            dropout=0.5
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()

        test_metrics, best_val_acc, history = train_and_evaluate(model, data, optimizer, criterion)

        result = {
            'sampled_aggregation': sampled_agg,
            'coarsen_aggregation': coarsen_agg,
            'test_metrics': test_metrics,
            'best_val_accuracy': best_val_acc,
            'training_history': history
        }
        results.append(result)

        print(f"\nFinal Test Results for {sampled_agg}-{coarsen_agg}:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")


        with open(f'aggregation_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)


    print("\nFinal Results Summary:")
    print("----------------------")
    for result in results:
        print(f"Sampled: {result['sampled_aggregation']}, "
              f"Coarsened: {result['coarsen_aggregation']}")
        metrics = result['test_metrics']
        print(f"Test Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print("----------------------")

    return results


if __name__ == "__main__":
    print("Starting aggregation experiments...")
    results = run_aggregation_experiments()