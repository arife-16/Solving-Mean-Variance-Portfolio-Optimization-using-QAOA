import numpy as np
import os
import networkx as nx
from typing import Optional

np.set_printoptions(suppress=True)


def random_weights(graph: nx.Graph,
                   rs: Optional[np.random.RandomState] = None,
                   type: str = 'bimodal'):
    """Create graph with random weights.
    
    Args:
        graph: Graph to add weights to
        rs: RandomState for reproducibility
        type: Weight distribution type - 'bimodal', 'gaussian', 'one'
    """
    if rs is None:
        rs = np.random
    elif not isinstance(rs, np.random.RandomState):
        raise ValueError("Invalid random state: {}".format(rs))

    problem_graph = nx.Graph()
    for n1, n2 in graph.edges:
        if type == 'bimodal':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-1, 1]))
        elif type == 'gaussian':
            problem_graph.add_edge(n1, n2, weight=rs.randn())
        elif type == 'one':
            problem_graph.add_edge(n1, n2, weight=1)
        else:
            raise ValueError(f"Unsupported weight type: {type}")
    
    return problem_graph


def safe_open_w(path):
    """Safe file opening for writing with directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'wb')


# Interconnect integration
try:
    import sys
    sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
    from interconnect import register_component
    
    class UtilsComponent:
        @staticmethod
        def random_weights(graph, rs=None, type='bimodal'):
            return random_weights(graph, rs, type)
        
        @staticmethod
        def safe_open_w(path):
            return safe_open_w(path)
    
    register_component('utils', UtilsComponent)
except ImportError:
    pass