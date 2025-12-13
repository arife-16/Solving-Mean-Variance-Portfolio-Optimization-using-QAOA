"""
Centralized configuration for QAOA Spectral Core
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Add interconnect path
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import interconnect

class SpectralConfig:
    """Unified configuration layer for all spectral operations."""
    
    def __init__(self):
        # Register component in interconnect
        interconnect.register_component('spectralconfig', SpectralConfig, override=True)
    
    # Spectral constants
    FREQUENCY_THRESHOLD = 1e-12
    SCALE_MIN = 0.5 * np.pi
    SCALE_MAX = 1.95 * np.pi
    
    # Scaling
    ENTROPY_WEIGHT = 0.7
    TOPOLOGY_WEIGHT = 0.3
    SCALE_BASE = 0.6
    SCALE_RANGE = 1.3
    
    # Topological weights
    DENSITY_WEIGHT = 0.25
    DEGREE_WEIGHT = 0.25
    CLUSTERING_WEIGHT = 0.20
    VARIANCE_WEIGHT = 0.15
    DIAMETER_WEIGHT = 0.15
    
    # Caching
    CACHE_MAX_SIZE = 1000
    
    # Fallback values
    FALLBACK_COMPLEX = 1 + 0j
    FALLBACK_REAL = 0.0
    FALLBACK_SCALE = np.pi
    
    @classmethod
    def from_external(cls, config: Dict[str, Any]) -> 'SpectralConfig':
        """Create configuration from external source."""
        instance = cls()
        for key, value in config.items():
            if hasattr(instance, key.upper()):
                setattr(instance, key.upper(), value)
        return instance
    
    def validate(self) -> bool:
        """Configuration validation."""
        return (
            self.FREQUENCY_THRESHOLD > 0 and
            0 < self.ENTROPY_WEIGHT <= 1 and
            0 < self.TOPOLOGY_WEIGHT <= 1 and
            abs(self.ENTROPY_WEIGHT + self.TOPOLOGY_WEIGHT - 1.0) < 1e-6
        )
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for specific component via interconnect."""
        return interconnect.route_request(component_name, 'get_config', {
            'method_params': {'config_type': 'spectral'}
        })
    
    def sync_with_components(self) -> None:
        """Synchronize configuration with other components."""
        config_data = {
            'frequency_threshold': self.FREQUENCY_THRESHOLD,
            'scale_range': (self.SCALE_MIN, self.SCALE_MAX),
            'weights': {
                'entropy': self.ENTROPY_WEIGHT,
                'topology': self.TOPOLOGY_WEIGHT
            }
        }
        interconnect.broadcast('config_updated', config_data)