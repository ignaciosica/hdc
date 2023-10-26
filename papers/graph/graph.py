# Dependencies

import numpy as np
import functools as ft
import networkx as nx


# Graph pre-processing


def process_dataset(dataset):
    graphs, labels = [], []

    for graph in dataset:
        G = nx.Graph()
        G.add_edges_from(
            zip(graph["edge_index"][0], graph["edge_index"][1]), nodetype=int
        )
        graphs.append(G)
        labels.append(graph["y"][0])

    return (graphs, labels)


def transform(X, alpha, digits):
    graphs = []
    for graph in X:
        gpr = nx.pagerank(graph, alpha)
        # ecr = nx.eigenvector_centrality_numpy(graph)
        # kcr = nx.katz_centrality_numpy(graph, alpha)
        # dcr = nx.degree_centrality(graph)
        nodes = dict()
        for key, value in gpr.items():
            nodes[key] = str(round(value, digits))
        H = nx.relabel_nodes(graph, nodes)
        graphs.append(H)
    return graphs


def centrality(X, rank):
    graphs = []
    for graph in X:
        cr = rank(graph)
        H = nx.relabel_nodes(graph, cr)
        graphs.append(H)
    return graphs
    #     nodes = dict()
    #     for key, value in cr.items():
    #         nodes[key] = str(round(value, digits))
    #     H = nx.relabel_nodes(graph, nodes)
    #     graphs.append(H)
    # return graphs
