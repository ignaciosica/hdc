# Dependencies

import numpy as np
import functools as ft
import networkx as nx

from memory_profiler import profile

# Graph pre-processing


def process_dataset(dataset):
    graphs, labels = [], []

    for graph in dataset:
        G = nx.Graph()
        G.add_edges_from(
            zip(graph["edge_index"][0], graph["edge_index"][1]), nodetype=int
        )
        if G.number_of_nodes() > 100:
            continue
        graphs.append(G)
        labels.append(graph["y"][0])

    return (graphs, labels)


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


def centrality(X, rank):
    graphs = []
    for graph in X:
        graphs.append(nx.relabel_nodes(graph, rank(graph)))

    return graphs


# def centrality(X, rank):
#     for graph in X:
#         yield nx.relabel_nodes(graph, rank(graph))
