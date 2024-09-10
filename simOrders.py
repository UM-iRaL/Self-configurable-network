import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def randomGraph(n, p):
    graph = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(n, p)
    return graph


def getTime(graph):
    n = graph.number_of_nodes()
    paths = dict(nx.all_pairs_shortest_path_length(graph))
    return sum([paths[i][i + 1] for i in range(n - 1)])


def relabelGraph(graph, relabel):
    nodes = list(graph.nodes)
    relabelMap = dict(zip(nodes, relabel))
    return nx.relabel_nodes(graph, relabelMap)


def shuffleGraph(graph):
    from random import shuffle
    shuffled = list(graph.nodes)
    shuffle(shuffled)
    gshuffled = relabelGraph(graph, shuffled)
    return gshuffled


def randomOrder(graph):
    gshuffled = shuffleGraph(graph)
    return getTime(gshuffled)


def DFSOrder(graph):
    def init(v, w):
        graph.nodes[v]['done'] = True
        graph.nodes[v]['parent'] = w
        message(v)

    def message(v):
        nonlocal t
        order = len([w for w in graph.nodes if graph.nodes[w]['done']])
        nodes = [w for w in graph[v] if not graph.nodes[w]['done']]
        par = graph.nodes[v]['parent']
        if nodes:
            t += 1
            w = nodes[0]
            init(w, v)
        elif order != graph.number_of_nodes():
            t += 1
            message(par)

    for i in range(graph.number_of_nodes()):
        graph.nodes[i]['done'] = False
    t = 0
    init(0, -1)
    return t


def bestOrder(graph):
    from itertools import permutations
    n = graph.number_of_nodes()
    perm = list(permutations(range(n)))
    T = 2*n
    for p in perm:
        graphRe = relabelGraph(graph, p)
        Tp = getTime(graphRe)
        if Tp < T:
            T = Tp
    return T


def main():
    n = 40
    p = .05
    N = 300
    graphs = [randomGraph(n, p) for _ in range(N)]
    randomRes = np.array([randomOrder(gr) for gr in graphs])
    dfsRes = np.array([DFSOrder(gr) for gr in graphs])
    # bestRes = np.array([bestOrder(gr) for gr in graphs])
    print(randomRes)
    print(dfsRes)
    # print(bestRes)
    kwargs = dict(histtype='stepfilled', alpha=0.6, density=True, ec="k")
    plt.hist((randomRes, dfsRes), bins=50)
    # plt.hist(dfsRes, **kwargs)
    # plt.hist(bestRes, **kwargs)
    plt.show()





if __name__ == '__main__':
    # graph = nx.Graph()
    # graph.add_nodes_from([0, 1, 2, 3, 4, 5])
    # graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    # print(bestOrder(graph))
    main()
