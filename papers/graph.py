def processDataset(dataset):
    graphs = []
    labels = []

    for graph in dataset:
        G = nx.Graph()
        G.add_edges_from(zip(graph["edge_index"][0], graph["edge_index"][1]))
        graphs.append(G)
        labels.append(graph["y"][0])

    return (graphs, labels)

