import numpy as np
from typing import Optional, Tuple
import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import register_component


class MVN:
    """Multivalued neuron (MVN) with complex weights and feature importance support."""
    def __init__(self, n_inputs: int, k: int, is_continuous: bool = False):
        self.n_inputs = n_inputs
        self.k = k
        self.is_continuous = is_continuous
        # Initialize complex weights with small random values
        self.weights = (np.random.randn(n_inputs + 1) * 0.01 + 
                       1j * np.random.randn(n_inputs + 1) * 0.01)
        # Feature importance weights (initialized to ones)
        self.importance_weights = np.ones(n_inputs + 1, dtype=np.float64)

    def set_importance_weights(self, importance: Optional[np.ndarray]) -> None:
        """Set feature importance weights."""
        if importance is not None:
            # Bias remains with importance 1.0
            self.importance_weights[0] = 1.0
            self.importance_weights[1:] = importance[:self.n_inputs]
        else:
            self.importance_weights.fill(1.0)

    def activation(self, z: np.complex128) -> np.complex128:
        """MVN activation: discrete or continuous."""
        if self.is_continuous:
            return z / np.abs(z) if np.abs(z) > 1e-12 else 1+0j
        
        if np.abs(z) < 1e-12:
            return 1+0j
            
        arg_z = np.angle(z)
        # Normalize angle to range [0, 2π)
        if arg_z < 0:
            arg_z += 2 * np.pi
        
        j = int(np.floor(self.k * arg_z / (2 * np.pi))) % self.k
        return np.exp(1j * 2 * np.pi * j / self.k)

    def forward(self, x: np.ndarray) -> Tuple[np.complex128, np.complex128]:
        """Forward pass through neuron with feature importance."""
        x_aug = np.append(1, x)
        # Apply importance weights to neuron connections
        weighted_weights = self.weights * self.importance_weights
        z = np.dot(weighted_weights, x_aug)
        activation = self.activation(z)
        return activation, z

    def forward_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch forward pass with feature importance."""
        M = X.shape[0]
        X_aug = np.hstack([np.ones((M, 1), dtype=np.complex128), X])
        # Apply importance weights
        weighted_weights = self.weights * self.importance_weights
        Z = np.dot(X_aug, weighted_weights)
        
        activations = np.zeros(M, dtype=np.complex128)
        for i in range(M):
            activations[i] = self.activation(Z[i])
            
        return activations, Z


# Register component in interconnect
register_component('mvn', MVN)