import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

from torch_geometric.datasets import TUDataset
import torchmetrics

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


def degree_centrality(G):
    """
    Compute the degree centrality for nodes.
    """
    _, columns = G.edge_index
    degree = torch.bincount(columns, minlength=G.num_nodes)
    return degree / G.num_nodes


def to_undirected(edge_index):
    """
    Returns the undirected edge_index
    [[0, 1], [1, 0]] will result in [[0], [1]]
    """
    edge_index = edge_index.sort(dim=0)[0]
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def min_max_graph_size(graph_dataset):
    if len(graph_dataset) == 0:
        return None, None

    max_num_nodes = float("-inf")
    min_num_nodes = float("inf")

    for G in graph_dataset:
        num_nodes = G.num_nodes
        max_num_nodes = max(max_num_nodes, num_nodes)
        min_num_nodes = min(min_num_nodes, num_nodes)

    return min_num_nodes, max_num_nodes


class Encoder(nn.Module):
    def __init__(self, out_features, size):
        super(Encoder, self).__init__()
        self.out_features = out_features
        self.node_ids = embeddings.Random(size, out_features)

    def forward(self, x):
        pr = degree_centrality(x)
        _, pr_argsort = pr.sort()

        node_id_hvs = torch.zeros((x.num_nodes, self.out_features), device=device)
        node_id_hvs[pr_argsort] = self.node_ids.weight[: x.num_nodes]

        row, col = to_undirected(x.edge_index)

        hvs = torchhd.bind(node_id_hvs[row], node_id_hvs[col])
        return torchhd.multiset(hvs)


DIMENSIONS = 10000

dataset = "MUTAG"
print(f"Testing {dataset}")

graphs = TUDataset("../data", dataset)
train_size = int(0.7 * len(graphs))
test_size = len(graphs) - train_size

train_ld, test_ld = torch.utils.data.random_split(graphs, [train_size, test_size])

min_graph_size, max_graph_size = min_max_graph_size(graphs)

encoder = Encoder(DIMENSIONS, max_graph_size)
encoder = encoder.to(device)

model = Centroid(DIMENSIONS, graphs.num_classes)
model = model.to(device)

start = time.time()

with torch.no_grad():
    for samples in tqdm(train_ld, desc="Training"):
        samples.edge_index = samples.edge_index.to(device)
        samples.y = samples.y.to(device)

        test_samples_hv = encoder(samples).unsqueeze(0)
        model.add(test_samples_hv, samples.y)


f1score = torchmetrics.F1Score("multiclass", num_classes=graphs.num_classes)

# with torch.no_grad():
model.normalize()
matrix = torch.zeros((len(test_ld) + model.weight.size(0), DIMENSIONS))
incorrect_indices = []
correct_indices = []

for index in range(model.weight.size(0)):
    matrix[index] = model.weight[index]

for index, samples in enumerate(tqdm(test_ld, desc="Testing_")):
    samples.edge_index = samples.edge_index.to(device)

    test_samples_hv = encoder(samples).unsqueeze(0)
    matrix[index + model.weight.size(0) - 1] = test_samples_hv
    test_outputs = model(test_samples_hv, dot=True)

    f1score.update(test_outputs.cpu(), samples.y)
    if test_outputs.cpu().argmax() != samples.y:
        incorrect_indices.append(index)
    else:
        correct_indices.append(index)

end = time.time()

print(f"Test: f1-score of {f1score.compute().item() * 100:.3f}%")
print(f"Time: {(end - start):.3f}s")


tsne = TSNE(n_components=2, random_state=42, init="random", metric="cosine", perplexity=25)
vis_dims = tsne.fit_transform(matrix)
colors = ["blue", "green"]
x = [x for x, y in vis_dims][model.weight.size(0) :]
y = [y for x, y in vis_dims][model.weight.size(0) :]
color_indices = graphs.data.y[test_ld.indices]

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.4)

colors = ["orange", "darkgreen"]
colormap = matplotlib.colors.ListedColormap(colors)

plt.scatter(x[: model.weight.size(0)], y[: model.weight.size(0)], c=[0, 1], cmap=colormap)

plt.title("Level embeddings visualized using t-SNE")
plt.show(block=True)


# for score in [0, 1]:
#     # avg_x = np.array(x)[df.Score - 1 == score].mean()
#     # avg_y = np.array(y)[df.Score - 1 == score].mean()
#     color = colors[score]
#     # plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)
