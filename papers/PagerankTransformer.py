import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin

class PagerankTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.55):
        self.alpha = alpha

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        graphs = []

        for graph in X:
            gpr = nx.pagerank(graph, self.alpha)
            graphs.append((graph, gpr))

        return graphs