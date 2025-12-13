import numpy as np
import networkx as nx
import sys
import os
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import register_component


class Graph:
    """Class for working with graphs in RQAOA."""

    def __init__(self, n, d=3, G=None):
        self.n = n
        self.d = d
        if G is None:
            G = nx.generators.random_graphs.random_regular_graph(d, n)
            rs = np.random.RandomState(42)
            for e in G.edges():
                G[e[0]][e[1]]['weight'] = 2 * rs.randint(2) - 1
        self.G = G
        self.G0 = G.copy()

    def reset(self):
        """Reset graph to initial state."""
        self.G = self.G0.copy()

    def get_G_numpy(self, nodelist=None):
        """Get adjacency matrix of the graph."""
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_numpy_array(self.G, dtype=np.float64, nodelist=nodelist)

    def get_G_sparse(self, nodelist=None):
        """Get sparse adjacency matrix of the graph."""
        if nodelist is None:
            nodelist = range(self.n)
        return nx.to_scipy_sparse_matrix(self.G, dtype=np.float64, nodelist=nodelist)

    def eliminate(self, edge, sign):
        """Eliminate vertex from graph."""
        rmv_edges = []
        updt_edges = []
        add_edges = []
        for neighb in self.G.neighbors(edge[1]):
            rmv_edges += [(edge[1], neighb)]
            if neighb not in edge:
                if (edge[0], neighb) in self.G.edges():
                    self.G[edge[0]][neighb]['weight'] += sign * self.G[edge[1]][neighb]['weight']
                    if self.G[edge[0]][neighb]['weight'] == 0:
                        self.G.remove_edge(edge[0], neighb)
                        rmv_edges += [(edge[0], neighb)]
                    else:
                        updt_edges += [(edge[0], neighb)]
                else:
                    self.G.add_edge(edge[0], neighb, weight=sign * self.G[edge[1]][neighb]['weight'])
                    add_edges += [(edge[0], neighb)]
        for e in self.G.edges(edge[0]):
            if e not in updt_edges + rmv_edges + add_edges:
                updt_edges += [e]
        for neighb in self.G.neighbors(edge[1]):
            for e in self.G.edges(neighb):
                if e not in updt_edges + rmv_edges + add_edges:
                    updt_edges += [e]
        self.G.remove_node(edge[1])
        return rmv_edges, updt_edges, add_edges


# Register component in interconnect
register_component('graph', Graph)