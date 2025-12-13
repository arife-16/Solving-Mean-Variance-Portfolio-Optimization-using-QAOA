import numpy as np
import networkx as nx
from typing import Tuple, Optional, Dict, Any
import sys
import os
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import route_request, register_component

# Import config from same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from config import SpectralConfig

class SpectralCore:
    """Basic spectral operations with unified interface."""
    
    def __init__(self, config: Optional[SpectralConfig] = None):
        self.config = config or SpectralConfig()
        if not self.config.validate():
            raise ValueError("Invalid configuration")
    
    def compute_laplacian_spectrum(self, graph: nx.Graph) -> np.ndarray:
        """Compute Laplacian spectrum."""
        if len(graph.nodes()) == 0:
            return np.array([self.config.FALLBACK_REAL])
        
        try:
            laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
            eigenvals = np.real(np.linalg.eigvals(laplacian))
            return np.sort(eigenvals)
        except (np.linalg.LinAlgError, ValueError, nx.NetworkXError):
            return np.array([self.config.FALLBACK_REAL])
    
    def compute_adjacency_spectrum(self, graph: nx.Graph) -> np.ndarray:
        """Compute adjacency matrix spectrum."""
        if len(graph.nodes()) == 0:
            return np.array([self.config.FALLBACK_REAL])
        
        try:
            adj_matrix = nx.to_numpy_array(graph).astype(float)
            eigenvals = np.real(np.linalg.eigvals(adj_matrix))
            return np.sort(eigenvals)
        except (np.linalg.LinAlgError, ValueError, nx.NetworkXError):
            return np.array([self.config.FALLBACK_REAL])
    
    def pad_to_power_of_2(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Pad to power of two."""
        n = len(data)
        if n <= 0:
            return np.array([self.config.FALLBACK_REAL, self.config.FALLBACK_REAL]), 2
        
        next_power_of_2 = 2 ** int(np.ceil(np.log2(n))) if n > 1 else 2
        padded_data = np.zeros(next_power_of_2, dtype=data.dtype)
        padded_data[:n] = data
        return padded_data, next_power_of_2
    
    def normalize_to_unit_circle(self, fft_data: np.ndarray) -> np.ndarray:
        """Normalize FFT data."""
        normalized = np.zeros_like(fft_data, dtype=complex)
        abs_vals = np.abs(fft_data)
        valid_mask = abs_vals > self.config.FREQUENCY_THRESHOLD
        
        normalized[valid_mask] = fft_data[valid_mask] / abs_vals[valid_mask]
        normalized[~valid_mask] = self.config.FALLBACK_COMPLEX
        
        return normalized
    
    def extract_importance_weights(self, fft_data: np.ndarray) -> np.ndarray:
        """Extract importance weights."""
        weights = np.abs(fft_data)
        max_weight = np.max(weights)
        
        if max_weight > self.config.FREQUENCY_THRESHOLD:
            return weights / max_weight
        return np.ones_like(weights, dtype=float)
    
    def get_spectral_features(self, graph: nx.Graph) -> Dict[str, Any]:
        """Basic spectral characteristics."""
        if len(graph.nodes()) == 0:
            return {
                'spectral_radius': self.config.FALLBACK_REAL,
                'algebraic_connectivity': self.config.FALLBACK_REAL,
                'spectral_gap': self.config.FALLBACK_REAL
            }
        
        try:
            adj_spectrum = self.compute_adjacency_spectrum(graph)
            lap_spectrum = self.compute_laplacian_spectrum(graph)
            
            return {
                'spectral_radius': float(np.max(np.abs(adj_spectrum))),
                'algebraic_connectivity': float(lap_spectrum[1]) if len(lap_spectrum) > 1 else self.config.FALLBACK_REAL,
                'spectral_gap': float(lap_spectrum[1]) if len(lap_spectrum) > 1 else self.config.FALLBACK_REAL
            }
        except Exception:
            return {
                'spectral_radius': self.config.FALLBACK_REAL,
                'algebraic_connectivity': self.config.FALLBACK_REAL,
                'spectral_gap': self.config.FALLBACK_REAL
            }
    
    def get_component_method(self, component: str, method: str, params: dict = None):
        """Call method from another component via interconnect."""
        return route_request(component, method, params or {})

# Register component in interconnect
register_component('spectralcore', SpectralCore)