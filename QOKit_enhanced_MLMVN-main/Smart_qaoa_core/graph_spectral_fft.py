import numpy as np
import networkx as nx
from typing import Optional, Union
import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import route_request, register_component

class GraphSpectralFFT:
    """FFT-encoding of spectral characteristics with centralized management."""
    
    def __init__(self, spectral_core: Optional = None, 
                 cache_manager: Optional = None,
                 config: Optional = None):
        self.config = config or route_request('spectralconfig', 'get_default_config')['data']
        self.spectral_core = spectral_core or route_request('spectralcore', 'create_instance', {'init_params': {'config': self.config}})['data']
        self.cache = cache_manager or route_request('cachemanager', 'create_instance', {'init_params': {'config': self.config}})['data']
        self._current_importance_key = None
    
    def encode_graph_fft(self, graph: nx.Graph) -> np.ndarray:
        """Main FFT encoding of graph."""
        if not isinstance(graph, nx.Graph):
            raise TypeError("Expected networkx.Graph object")
        
        try:
            laplacian_spectrum = route_request('spectralcore', 'compute_laplacian_spectrum', {'method_params': {'graph': graph}, 'instance_id': id(self.spectral_core)})['data']
            adjacency_spectrum = route_request('spectralcore', 'compute_adjacency_spectrum', {'method_params': {'graph': graph}, 'instance_id': id(self.spectral_core)})['data']
            
            combined_spectrum = np.concatenate([laplacian_spectrum, adjacency_spectrum])
            result = route_request('spectralcore', 'pad_to_power_of_2', {'method_params': {'spectrum': combined_spectrum}, 'instance_id': id(self.spectral_core)})['data']
            padded_spectrum, size = result
            fft_spectrum = np.fft.fft(padded_spectrum)
            
            half_size = size // 2
            importance_weights = route_request('spectralcore', 'extract_importance_weights', {'method_params': {'spectrum': fft_spectrum[:half_size]}, 'instance_id': id(self.spectral_core)})['data']
            
            # Caching importance weights
            self._current_importance_key = f"graph_fft_{hash(str(sorted(graph.edges())))}"
            route_request('cachemanager', 'set_importance_weights', {'method_params': {'key': self._current_importance_key, 'weights': importance_weights}, 'instance_id': id(self.cache)})
            
            normalized_fft = route_request('spectralcore', 'normalize_to_unit_circle', {'method_params': {'spectrum': fft_spectrum}, 'instance_id': id(self.spectral_core)})['data']
            return normalized_fft[:half_size]
            
        except Exception:
            return np.array([self.config.FALLBACK_COMPLEX])
    
    def encode_multiscale_fft(self, graph: nx.Graph) -> np.ndarray:
        """Multiscale FFT encoding."""
        if not isinstance(graph, nx.Graph):
            raise TypeError("Expected networkx.Graph object")
        
        features_list = []
        importance_list = []
        
        try:
            # Laplacian spectrum
            laplacian_spectrum = route_request('spectralcore', 'compute_laplacian_spectrum', {'method_params': {'graph': graph}, 'instance_id': id(self.spectral_core)})['data']
            if len(laplacian_spectrum) > 1:
                result = route_request('spectralcore', 'pad_to_power_of_2', {'method_params': {'spectrum': laplacian_spectrum}, 'instance_id': id(self.spectral_core)})['data']
                padded_lap, size_lap = result
                fft_lap = np.fft.fft(padded_lap)
                half_size_lap = size_lap // 2
                norm_lap = route_request('spectralcore', 'normalize_to_unit_circle', {'method_params': {'spectrum': fft_lap[:half_size_lap]}, 'instance_id': id(self.spectral_core)})['data']
                features_list.extend(norm_lap)
                importance_list.extend(route_request('spectralcore', 'extract_importance_weights', {'method_params': {'spectrum': fft_lap[:half_size_lap]}, 'instance_id': id(self.spectral_core)})['data'])
            
            # Node degrees
            try:
                degrees = np.array([d for _, d in graph.degree()], dtype=float)
                if len(degrees) > 0:
                    result = route_request('spectralcore', 'pad_to_power_of_2', {'method_params': {'spectrum': degrees}, 'instance_id': id(self.spectral_core)})['data']
                    padded_deg, size_deg = result
                    fft_deg = np.fft.fft(padded_deg)
                    half_size_deg = size_deg // 2
                    norm_deg = route_request('spectralcore', 'normalize_to_unit_circle', {'method_params': {'spectrum': fft_deg[:half_size_deg]}, 'instance_id': id(self.spectral_core)})['data']
                    features_list.extend(norm_deg)
                    importance_list.extend(route_request('spectralcore', 'extract_importance_weights', {'method_params': {'spectrum': fft_deg[:half_size_deg]}, 'instance_id': id(self.spectral_core)})['data'])
            except Exception:
                pass
            
            # Clustering coefficients
            try:
                clustering_coeffs = np.array(list(nx.clustering(graph).values()), dtype=float)
                if len(clustering_coeffs) > 1:
                    result = route_request('spectralcore', 'pad_to_power_of_2', {'method_params': {'spectrum': clustering_coeffs}, 'instance_id': id(self.spectral_core)})['data']
                    padded_clust, size_clust = result
                    fft_clust = np.fft.fft(padded_clust)
                    half_size_clust = size_clust // 2
                    norm_clust = route_request('spectralcore', 'normalize_to_unit_circle', {'method_params': {'spectrum': fft_clust[:half_size_clust]}, 'instance_id': id(self.spectral_core)})['data']
                    features_list.extend(norm_clust)
                    importance_list.extend(route_request('spectralcore', 'extract_importance_weights', {'method_params': {'spectrum': fft_clust[:half_size_clust]}, 'instance_id': id(self.spectral_core)})['data'])
            except Exception:
                pass
            
            # Caching importance weights
            if importance_list:
                self._current_importance_key = f"multiscale_fft_{hash(str(sorted(graph.edges())))}"
                route_request('cachemanager', 'set_importance_weights', {'method_params': {'key': self._current_importance_key, 'weights': np.array(importance_list)}, 'instance_id': id(self.cache)})
            
            return np.array(features_list, dtype=complex) if features_list else np.array([self.config.FALLBACK_COMPLEX])
                
        except Exception:
            return np.array([self.config.FALLBACK_COMPLEX])
    
    def get_feature_importance_weights(self) -> Optional[np.ndarray]:
        """Getting importance weights from cache."""
        if self._current_importance_key:
            return route_request('cachemanager', 'get_importance_weights', {'method_params': {'key': self._current_importance_key}, 'instance_id': id(self.cache)})['data']
        return None

class AdaptiveSpectralFFT(GraphSpectralFFT):
    """Adaptive spectral FFT with topology weights."""
    
    def _compute_adaptive_weights(self, graph: nx.Graph, spectrum: np.ndarray) -> np.ndarray:
        """Computing adaptive weights."""
        if not isinstance(graph, nx.Graph) or len(spectrum) == 0:
            return np.ones(1, dtype=float)
        
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        
        if n_nodes <= 1:
            return np.ones_like(spectrum, dtype=float)
        
        try:
            max_edges = n_nodes * (n_nodes - 1) / 2
            density = n_edges / max_edges if max_edges > 0 else 0
            
            try:
                clustering = nx.average_clustering(graph)
            except (ZeroDivisionError, ValueError, nx.NetworkXError):
                clustering = 0.0
            
            base_weight = 0.5 + 0.3 * density + 0.2 * clustering
            adaptive_weights = np.full_like(spectrum, base_weight, dtype=float)
            
            max_spectrum = np.max(np.abs(spectrum))
            if max_spectrum > self.config.FREQUENCY_THRESHOLD:
                spectrum_normalized = spectrum / max_spectrum
                importance_mask = np.abs(spectrum_normalized) > 0.1
                adaptive_weights[importance_mask] *= 1.2
            
            return adaptive_weights
            
        except Exception:
            return np.ones_like(spectrum, dtype=float)
    
    def encode_adaptive_fft(self, graph: nx.Graph) -> np.ndarray:
        """FFT encoding with adaptive weights."""
        if not isinstance(graph, nx.Graph):
            raise TypeError("Expected networkx.Graph object")
        
        try:
            laplacian_spectrum = route_request('spectralcore', 'compute_laplacian_spectrum', {'method_params': {'graph': graph}, 'instance_id': id(self.spectral_core)})['data']
            adaptive_weights = self._compute_adaptive_weights(graph, laplacian_spectrum)
            weighted_spectrum = laplacian_spectrum * adaptive_weights
            
            result = route_request('spectralcore', 'pad_to_power_of_2', {'method_params': {'spectrum': weighted_spectrum}, 'instance_id': id(self.spectral_core)})['data']
            padded_spectrum, size = result
            fft_spectrum = np.fft.fft(padded_spectrum)
            half_size = size // 2
            
            # Combined importance weights
            base_importance = route_request('spectralcore', 'extract_importance_weights', {'method_params': {'spectrum': fft_spectrum[:half_size]}, 'instance_id': id(self.spectral_core)})['data']
            weights_len = min(len(adaptive_weights), len(base_importance))
            combined_importance = base_importance[:weights_len] * adaptive_weights[:weights_len]
            
            # Caching importance weights
            self._current_importance_key = f"adaptive_fft_{hash(str(sorted(graph.edges())))}"
            route_request('cachemanager', 'set_importance_weights', {'method_params': {'key': self._current_importance_key, 'weights': combined_importance}, 'instance_id': id(self.cache)})
            
            normalized_fft = route_request('spectralcore', 'normalize_to_unit_circle', {'method_params': {'spectrum': fft_spectrum}, 'instance_id': id(self.spectral_core)})['data']
            return normalized_fft[:half_size]
            
        except Exception:
            return np.array([self.config.FALLBACK_COMPLEX])

# Register components in interconnect
register_component('graphspectralfft', GraphSpectralFFT)
register_component('adaptivespectralfft', AdaptiveSpectralFFT)