# GraphHD

**GraphHD** implementation in python with numpy, networkx and sklearn

**Abstract.** GraphHD presents a novel baseline approach for graph classification with HDC. It uses PageRank centrality rank to map vertices from different graphs into the same HV. The classification algorithm is simply comparison similarity search for the given query-vector with the classes HVs.

**Abstract.** Hyperdimensional Computing (HDC) developed by Kanerva is a computational model for machine learning inspired by neuroscience. HDC exploits characteristics of biological neural systems such as high-dimensionality, randomness and a holographic representation of information to achieve a good balance between accuracy, efficiency and robustness. HDC models have already been proven to be useful in different learning applications, especially in resource-limited settings such as the increas- ingly popular Internet of Things (IoT). One class of learning tasks that is missing from the current body of work on HDC is graph classification. Graphs are among the most important forms of information representation, yet, to this day, HDC algorithms have not been applied to the graph learning problem in a general sense. Moreover, graph learning in IoT and sensor networks, with limited compute capabilities, introduce challenges to the overall design methodology. In this paper, we present GraphHD — a baseline approach for graph classification with HDC. We evaluate GraphHD on real-world graph classification problems. Our results show that when compared to the state-of-the-art Graph Neural Networks (GNNs) the proposed model achieves comparable accuracy, while training and inference times are on average 14.6× and 2.0× faster, respectively.

### Dependencies

```python
import numpy as np
import functools as ft
import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin
from datasets import load_dataset
from sklearn.model_selection import cross_val_score
```

### Basic HDC operations

```python
def hdv(d):
    return np.random.choice([-1, 1], d)


def bind(xs):
    return ft.reduce(lambda x, y: x * y, xs)


def bundle(xs):
    return np.sign(ft.reduce(lambda x, y: x + y, xs))


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    if norm_A == 0 or norm_B == 0:
        return 0

    return dot_product / (norm_A * norm_B)
```

### Item Memory

```python
class ItemMemory:
    def __init__(self, vectors=[]):
        self.vectors = vectors

    def addVector(self, label, V):
        self.vectors.append((label, V))

    def count(self):
        return len(self.vectors)

    def cleanup(self, V):
        return max(self.vectors, key=lambda x: cosine_similarity(V, x[1]))
```

### Process Dataset

```python
def processDataset(dataset):
    graphs = []
    labels = []

    for graph in dataset:
        G = nx.Graph()
        G.add_edges_from(zip(graph["edge_index"][0], graph["edge_index"][1]))
        graphs.append(G)
        labels.append(graph["y"][0])

    return (graphs, labels)
```

### Calculate PageRank

```python
def transform(X, alpha, digits):
    graphs = []
    for graph in X:
        gpr = nx.pagerank(graph, alpha)
        nodes = dict()
        for key, value in gpr.items():
            nodes[key] = str(round(value, digits))
        H = nx.relabel_nodes(graph, nodes)
        graphs.append(H)
    return graphs
```

### Encode Graph

```python
def encodeGraph(graph, vertices, dimensions):
    for node in graph.nodes:
        if node not in vertices:
            vertices[node] = hdv(dimensions)

    Edges = []

    for edge in graph.edges:
        v1 = vertices[edge[0]]
        v2 = vertices[edge[1]]
        E = bind([v1, v2])
        Edges.append(E)

    Graph = bundle(Edges)

    return Graph
```

### Scikit-learn Classifier

```python
class GraphHD(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.45, digits=4, dimensions=10000, step=20):
        self.alpha = alpha
        self.digits = digits
        self.dimensions = dimensions
        self.step = step

    def fit(self, X, y):
        self.memory = ItemMemory([])
        self.vertices = dict()
        self.labels = list(set(y))
        dictLabels = dict()

        graphs = transform(X, self.alpha, self.digits)

        for label in self.labels:
            dictLabels[label] = []

        for i in range(len(graphs)):
            Graph = encodeGraph(graphs[i], self.vertices, self.dimensions)
            dictLabels[y[i]].append(Graph)

        for key, value in dictLabels.items():
            for i in range(0, len(value), self.step):
                H = bundle(value[i : i + self.step])
                self.memory.addVector(str(key), H)

        return self

    def predict(self, X):
        graphs = transform(X, self.alpha, self.digits)

        p = []
        for testGraph in graphs:
            queryVector = encodeGraph(testGraph, self.vertices, self.dimensions)
            cleanVector = self.memory.cleanup(queryVector)
            p.append(int(cleanVector[0]))

        return p
```
