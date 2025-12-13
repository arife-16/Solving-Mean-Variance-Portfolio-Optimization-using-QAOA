"""
Centralized cache manager for spectral computations
"""

import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple
from interconnect import register_component, route_request

class CacheManager:
    """Single cache for all spectral operations."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self._topology_cache: Dict[Tuple, float] = {}
        self._spectrum_cache: Dict[Tuple, np.ndarray] = {}
        self._importance_cache: Dict[str, np.ndarray] = {}
    
    def _get_default_config(self) -> Dict:
        """Get configuration via interconnect."""
        try:
            result = route_request('config', 'get_config')
            return result.get('data', {}) if result['status'] == 'ok' else {'CACHE_MAX_SIZE': 1000}
        except:
            return {'CACHE_MAX_SIZE': 1000}
    
    def _get_graph_key(self, graph: nx.Graph) -> Tuple:
        """Create key for graph caching."""
        return (len(graph.nodes()), len(graph.edges()), tuple(sorted(graph.edges())))
    
    def get_topology_scale(self, graph: nx.Graph) -> Optional[float]:
        """Get topological scale from cache."""
        key = self._get_graph_key(graph)
        return self._topology_cache.get(key)
    
    def set_topology_scale(self, graph: nx.Graph, scale: float) -> None:
        """Save topological scale to cache."""
        if len(self._topology_cache) >= self.config.get('CACHE_MAX_SIZE', 1000):
            # Remove oldest element
            oldest_key = next(iter(self._topology_cache))
            del self._topology_cache[oldest_key]
        
        key = self._get_graph_key(graph)
        self._topology_cache[key] = scale
    
    def get_spectrum(self, graph: nx.Graph, spectrum_type: str) -> Optional[np.ndarray]:
        """Get spectrum from cache."""
        key = (*self._get_graph_key(graph), spectrum_type)
        return self._spectrum_cache.get(key)
    
    def set_spectrum(self, graph: nx.Graph, spectrum_type: str, spectrum: np.ndarray) -> None:
        """Save spectrum to cache."""
        if len(self._spectrum_cache) >= self.config.get('CACHE_MAX_SIZE', 1000):
            oldest_key = next(iter(self._spectrum_cache))
            del self._spectrum_cache[oldest_key]
        
        key = (*self._get_graph_key(graph), spectrum_type)
        self._spectrum_cache[key] = spectrum.copy()
    
    def get_importance_weights(self, key: str) -> Optional[np.ndarray]:
        """Get importance weights from cache."""
        weights = self._importance_cache.get(key)
        return weights.copy() if weights is not None else None
    
    def set_importance_weights(self, key: str, weights: np.ndarray) -> None:
        """Save importance weights to cache."""
        if len(self._importance_cache) >= self.config.get('CACHE_MAX_SIZE', 1000):
            oldest_key = next(iter(self._importance_cache))
            del self._importance_cache[oldest_key]
        
        self._importance_cache[key] = weights.copy()
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self._topology_cache.clear()
        self._spectrum_cache.clear()
        self._importance_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Cache usage statistics."""
        return {
            'topology_entries': len(self._topology_cache),
            'spectrum_entries': len(self._spectrum_cache),
            'importance_entries': len(self._importance_cache)
        }

# Register component in interconnect
register_component('cachemanager', CacheManager)