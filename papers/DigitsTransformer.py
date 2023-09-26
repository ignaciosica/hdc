import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin

class DigitsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, digits=5):
        self.digits = digits

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        graphs = []

        for graph, gpr in X:
            nodes = dict()
            for key, value in gpr.items():
                nodes[key] = str(round(value, self.digits))
            H = nx.relabel_nodes(graph, nodes)
            graphs.append(H)

        return graphs