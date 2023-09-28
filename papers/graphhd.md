# GraphHD

**GraphHD** implementation in python with numpy, networkx and sklearn

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
