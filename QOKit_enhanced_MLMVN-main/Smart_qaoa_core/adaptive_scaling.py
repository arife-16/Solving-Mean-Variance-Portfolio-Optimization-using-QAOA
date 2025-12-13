import numpy as np
import networkx as nx
from typing import Optional, Dict, Any
import sys
import os
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import route_request

class HybridAdaptiveScaling:
    """Hybrid adaptive scaling with centralized cache."""
    
    def __init__(self, config: Optional[Any] = None, cache_manager: Optional[Any] = None):
        self.config = config or route_request('config', 'get_instance')['data']
        self.cache = cache_manager or route_request('cachemanager', 'get_instance', {'method_params': {'config': self.config}})['data']
    
    def compute_entropy_scale(self, features: np.ndarray) -> float:
        """Entropy-based scaling."""
        features_positive = np.abs(features) + self.config.FREQUENCY_THRESHOLD
        features_norm = features_positive / np.sum(features_positive)
        
        entropy = -np.sum(features_norm * np.log(features_norm + self.config.FREQUENCY_THRESHOLD))
        max_entropy = np.log(len(features_norm))
        
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        scale_factor = (0.4 + normalized_entropy * 1.4) * np.pi
        return np.clip(scale_factor, self.config.SCALE_MIN, self.config.SCALE_MAX)
    
    def compute_topology_scale(self, graph: nx.Graph) -> float:
        """Topology-based scaling with caching."""
        # Cache check
        cached_scale = self.cache.get_topology_scale(graph)
        if cached_scale is not None:
            return cached_scale
        
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        
        if n_nodes <= 1:
            return self.config.FALLBACK_SCALE
        
        try:
            max_edges = n_nodes * (n_nodes - 1) / 2
            density = n_edges / max_edges if max_edges > 0 else 0
            
            degrees = [d for _, d in graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            normalized_avg_degree = avg_degree / (n_nodes - 1) if n_nodes > 1 else 0
            
            try:
                clustering = nx.average_clustering(graph)
            except (ZeroDivisionError, nx.NetworkXError):
                clustering = 0.0
            
            degree_variance = np.var(degrees) if len(degrees) > 1 else 0
            max_degree_variance = ((n_nodes - 1) ** 2) / 4
            normalized_degree_variance = degree_variance / max_degree_variance if max_degree_variance > 0 else 0
            
            try:
                normalized_diameter = (nx.diameter(graph) / (n_nodes - 1) 
                                     if nx.is_connected(graph) and n_nodes > 1 else 0.5)
            except (nx.NetworkXError, ValueError):
                normalized_diameter = 0.5
            
            topology_complexity = (
                self.config.DENSITY_WEIGHT * density +
                self.config.DEGREE_WEIGHT * normalized_avg_degree +
                self.config.CLUSTERING_WEIGHT * clustering +
                self.config.VARIANCE_WEIGHT * normalized_degree_variance +
                self.config.DIAMETER_WEIGHT * normalized_diameter
            )
            
            scale_factor = (self.config.SCALE_BASE + topology_complexity * self.config.SCALE_RANGE) * np.pi
            result = np.clip(scale_factor, self.config.SCALE_MIN, self.config.SCALE_MAX)
            
            # Cache result
            self.cache.set_topology_scale(graph, result)
            return result
            
        except Exception:
            return self.config.FALLBACK_SCALE
    
    def compute_hybrid_scale(self, graph: nx.Graph, features: np.ndarray) -> float:
        """Hybrid scaling."""
        entropy_scale = self.compute_entropy_scale(features)
        topology_scale = self.compute_topology_scale(graph)
        
        return (self.config.ENTROPY_WEIGHT * entropy_scale + 
                self.config.TOPOLOGY_WEIGHT * topology_scale)
    
    def get_scaling_info(self, graph: nx.Graph, features: np.ndarray) -> Dict[str, Any]:
        """Scaling information."""
        entropy_scale = self.compute_entropy_scale(features)
        topology_scale = self.compute_topology_scale(graph)
        hybrid_scale = (self.config.ENTROPY_WEIGHT * entropy_scale + 
                       self.config.TOPOLOGY_WEIGHT * topology_scale)
        
        # Basic characteristics
        features_positive = np.abs(features) + self.config.FREQUENCY_THRESHOLD
        features_norm = features_positive / np.sum(features_positive)
        entropy = -np.sum(features_norm * np.log(features_norm + self.config.FREQUENCY_THRESHOLD))
        max_entropy = np.log(len(features_norm))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
        density = n_edges / max_edges if max_edges > 0 else 0
        
        try:
            clustering = nx.average_clustering(graph)
        except (ZeroDivisionError, nx.NetworkXError):
            clustering = 0.0
        
        return {
            'entropy_scale': entropy_scale,
            'topology_scale': topology_scale,
            'hybrid_scale': hybrid_scale,
            'normalized_entropy': normalized_entropy,
            'graph_density': density,
            'average_clustering': clustering,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'entropy_contribution': self.config.ENTROPY_WEIGHT * entropy_scale,
            'topology_contribution': self.config.TOPOLOGY_WEIGHT * topology_scale
        }