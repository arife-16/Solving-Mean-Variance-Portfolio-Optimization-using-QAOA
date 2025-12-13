# Smart QAOA short example by ⚛️ Sigma PublishinQ Team ⚛️  
# https://www.linkedin.com/company/sigma-publishinq/about/


import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Optional


class ErdosRenyiFabric(nx.Graph):
    """
    Generates nx graph from adjacency matrix vector.
    """
    def __init__(self, vec_adjacancy: np.ndarray, num_nodes: int) -> None:
        super().__init__()
        self._vec_adj = vec_adjacancy if isinstance(vec_adjacancy, np.ndarray) else np.array(vec_adjacancy)
        self._n = num_nodes
        self._weighted_bool = not np.array_equal(self._vec_adj, self._vec_adj.astype(bool))
        self._poss_edges = self._edge_list()
        self._edges_list = self._graph_edges_n_weights()
        self.add_nodes_from(range(self._n))
        if self._weighted_bool:
            self.add_weighted_edges_from(self._edges_list)
        else:
            self.add_edges_from(self._edges_list)

    def _edge_list(self) -> List[Tuple]:
        """Returns all possible edges between nodes."""
        edge_list = []
        for i in range(self._n):
            for j in range(i + 1, self._n):
                edge_list.append((i, j))
        return edge_list

    def _graph_edges_n_weights(self) -> List[Tuple]:
        """Generates list of edges and weights for graph creation."""
        assert len(self._vec_adj) == len(self._poss_edges)
        configuration = []
        
        for edge_idx in range(len(self._vec_adj)):
            if self._vec_adj[edge_idx] > 0.0:
                if self._weighted_bool:
                    configuration.append(self._poss_edges[edge_idx] + (self._vec_adj[edge_idx],))
                else:
                    configuration.append(self._poss_edges[edge_idx])
        return configuration

    @property
    def possible_edges(self):
        return self._poss_edges

    @staticmethod
    def generate_vector_adjacency_elements(
        Data: pd.DataFrame,
        edge_prob: Tuple[float, float],
        weighted_bool: bool,
        num_possible_edges: int,
        adjacency_vec: List[str]
    ) -> np.ndarray:
        """
        Generates adjacency vector for random graph.
        """
        if not Data.empty:
            previous_graphs = Data[adjacency_vec].values
        else:
            previous_graphs = np.empty((0, len(adjacency_vec)))
            
        graph_edge_prob = np.random.uniform(low=edge_prob[0], high=edge_prob[1])
        
        if weighted_bool:
            vector_edges = np.array([
                np.random.rand() if np.random.rand() < graph_edge_prob else 0 
                for _ in range(num_possible_edges)
            ])
            while (len(previous_graphs) > 0 and 
                   any(np.equal(previous_graphs, vector_edges).all(1))) or np.sum(vector_edges) == 0.0:
                vector_edges = np.array([
                    np.random.rand() if np.random.rand() < graph_edge_prob else 0 
                    for _ in range(num_possible_edges)
                ])
        else:
            vector_edges = np.array([
                1.0 if np.random.rand() < graph_edge_prob else 0.0 
                for _ in range(num_possible_edges)
            ])
            while (len(previous_graphs) > 0 and 
                   any(np.equal(previous_graphs, vector_edges).all(1))) or np.sum(vector_edges) == 0.0:
                vector_edges = np.array([
                    1.0 if np.random.rand() < graph_edge_prob else 0.0 
                    for _ in range(num_possible_edges)
                ])
        
        return vector_edges