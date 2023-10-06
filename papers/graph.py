# Dependencies

import numpy as np
import functools as ft
import networkx as nx

# Graph pre-processing


def processDataset(dataset):
    graphs, labels = [], []

    for graph in dataset:
        G = nx.Graph()
        G.add_edges_from(zip(graph["edge_index"][0], graph["edge_index"][1]))
        nx.set_node_attributes(G, graph["node_feat"])
        # set_node_attributes(G, graph["node_feat"])
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
