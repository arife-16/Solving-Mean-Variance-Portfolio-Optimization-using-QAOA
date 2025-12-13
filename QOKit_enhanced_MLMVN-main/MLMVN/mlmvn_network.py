import numpy as np
from typing import Optional, List, Tuple
import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import route_request, register_component


class MLMVN:
    """MLMVN-LLS-SM for QAOA parameters prediction with integrated feature importance."""
    def __init__(self, layer_sizes: List[int], k: int, theta: float):
        register_component('mlmvn', MLMVN)
        self.layer_sizes = layer_sizes
        self.k = k
        self.theta = theta
        self.layers = []
        
        # Creating layers with MVN neurons
        for i in range(len(layer_sizes) - 1):
            # Hidden layers use continuous activation, output layer uses discrete
            is_continuous = (i < len(layer_sizes) - 2)
            layer = []
            for _ in range(layer_sizes[i + 1]):
                mvn_result = route_request('mvn', 'create_instance', {
                    'method_params': {
                        'input_size': layer_sizes[i], 
                        'k': k, 
                        'is_continuous': is_continuous
                    }
                })
                layer.append(mvn_result['data'])
            self.layers.append(layer)

    def set_feature_importance(self, importance_weights: Optional[np.ndarray]) -> None:
        """Setting feature importance weights for the first layer."""
        if importance_weights is not None and len(self.layers) > 0:
            for neuron in self.layers[0]:
                route_request('mvn', 'set_importance_weights', {
                    'instance_id': f'neuron_{id(neuron)}',
                    'method_params': {'importance_weights': importance_weights}
                })

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        current_input = x.copy()
        
        for layer in self.layers:
            layer_outputs = np.zeros(len(layer), dtype=np.complex128)
            for j, neuron in enumerate(layer):
                forward_result = route_request('mvn', 'forward', {
                    'instance_id': f'neuron_{id(neuron)}',
                    'method_params': {'x': current_input}
                })
                layer_outputs[j] = forward_result['data'][0]
            current_input = layer_outputs
            
        return current_input

    def forward_batch(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Batch forward pass with intermediate values preservation."""
        M = X.shape[0]
        all_outputs = [X]  # Input data
        all_z_values = []   # Weighted sums before activation
        
        current_input = X
        
        for layer_idx, layer in enumerate(self.layers):
            layer_outputs = np.zeros((M, len(layer)), dtype=np.complex128)
            layer_z_values = np.zeros((M, len(layer)), dtype=np.complex128)
            
            for neuron_idx, neuron in enumerate(layer):
                batch_result = route_request('mvn', 'forward_batch', {
                    'instance_id': f'neuron_{id(neuron)}',
                    'method_params': {'X': current_input}
                })
                layer_outputs[:, neuron_idx] = batch_result['data'][0]
                layer_z_values[:, neuron_idx] = batch_result['data'][1]
            
            all_outputs.append(layer_outputs)
            all_z_values.append(layer_z_values)
            current_input = layer_outputs
            
        return all_outputs, all_z_values

    def compute_errors(self, y_true: np.ndarray, y_pred: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Error computation with soft boundaries consideration for batch."""
        M, n_outputs = y_true.shape
        errors = np.zeros_like(y_true, dtype=np.complex128)
        
        for i in range(M):
            for j in range(n_outputs):
                if not np.isclose(y_true[i, j], y_pred[i, j], atol=1e-10):
                    # Standard error for incorrect prediction
                    if np.abs(z[i, j]) > 1e-12:
                        errors[i, j] = y_true[i, j] - z[i, j] / np.abs(z[i, j])
                    else:
                        errors[i, j] = y_true[i, j]
                else:
                    # Soft boundaries for correct prediction
                    y_angle = np.angle(y_true[i, j])
                    if y_angle < 0:
                        y_angle += 2 * np.pi
                    
                    sector_idx = int(np.floor(self.k * y_angle / (2 * np.pi))) % self.k
                    bisector_angle = 2 * np.pi * (sector_idx + 0.5) / self.k
                    bisector = np.exp(1j * bisector_angle)
                    
                    z_angle = np.angle(z[i, j])
                    if z_angle < 0:
                        z_angle += 2 * np.pi
                    
                    angle_diff = np.abs(z_angle - bisector_angle)
                    # Considering angle periodicity
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    
                    if angle_diff >= self.theta:
                        if np.abs(z[i, j]) > 1e-12:
                            errors[i, j] = bisector - z[i, j] / np.abs(z[i, j])
                        else:
                            errors[i, j] = bisector
                            
        return errors

    def backpropagate(self, errors: np.ndarray, all_outputs: List[np.ndarray]) -> List[np.ndarray]:
        """Backpropagation of errors through all layers."""
        layer_errors = [errors]  # Output layer errors
        
        # Backward propagation from output layer to input
        for r in range(len(self.layers) - 1, 0, -1):
            current_layer_errors = layer_errors[0]  # Current layer errors
            prev_layer_size = self.layer_sizes[r]
            prev_layer_errors = np.zeros((current_layer_errors.shape[0], prev_layer_size), 
                                       dtype=np.complex128)
            
            # For each neuron in the previous layer
            for j in range(prev_layer_size):
                # Sum weighted errors from all neurons in current layer
                for i in range(len(self.layers[r])):
                    # Get weight and importance weight
                    weight_result = route_request('mvn', 'get_weight', {
                        'instance_id': f'neuron_{id(self.layers[r][i])}',
                        'method_params': {'index': j + 1}
                    })
                    importance_result = route_request('mvn', 'get_importance_weight', {
                        'instance_id': f'neuron_{id(self.layers[r][i])}',
                        'method_params': {'index': j + 1}
                    })
                    weight = weight_result['data'] * importance_result['data']
                    
                    if np.abs(weight) > 1e-12:
                        weight_inv = np.conj(weight) / (np.abs(weight) ** 2)
                        prev_layer_errors[:, j] += current_layer_errors[:, i] * weight_inv
                
                # Normalization by number of inputs
                prev_layer_errors[:, j] /= (prev_layer_size + 1)
            
            layer_errors.insert(0, prev_layer_errors)
        
        return layer_errors

    def lls_adjust_weights(self, X: np.ndarray, errors: np.ndarray, 
                          layer_idx: int, neuron_idx: int) -> None:
        """Weight adjustment using least squares method with importance consideration."""
        M = X.shape[0]
        neuron = self.layers[layer_idx][neuron_idx]
        
        # Input data extension with bias
        if X.ndim == 1:
            X_aug = np.hstack([np.ones(1, dtype=np.complex128), X.reshape(1, -1)])
        else:
            X_aug = np.hstack([np.ones((M, 1), dtype=np.complex128), X])
        
        # Applying importance weights to input data for LLS
        importance_weights_result = route_request('mvn', 'get_all_importance_weights', {
            'instance_id': f'neuron_{id(neuron)}'
        })
        X_weighted = X_aug * importance_weights_result['data']
        
        # Errors for this neuron
        if errors.ndim == 1:
            delta = errors.reshape(-1, 1)
        else:
            delta = errors[:, neuron_idx].reshape(-1, 1)
        
        try:
            # Matrix pseudoinverse for solving linear system
            X_pinv = np.linalg.pinv(X_weighted)
            delta_w = np.dot(X_pinv, delta).flatten()
            
            # Weight update with importance consideration
            route_request('mvn', 'update_weights', {
                'instance_id': f'neuron_{id(neuron)}',
                'method_params': {'delta_w': delta_w}
            })
            
        except np.linalg.LinAlgError:
            # In case of singular matrix, use regularization
            regularization = 1e-6
            XtX = np.dot(X_weighted.T.conj(), X_weighted)
            XtX += regularization * np.eye(XtX.shape[0])
            
            try:
                XtX_inv = np.linalg.inv(XtX)
                delta_w = np.dot(XtX_inv, np.dot(X_weighted.T.conj(), delta)).flatten()
                route_request('mvn', 'update_weights', {
                    'instance_id': f'neuron_{id(neuron)}',
                    'method_params': {'delta_w': delta_w}
                })
            except np.linalg.LinAlgError:
                # If this doesn't help, use gradient update
                learning_rate = 0.01
                gradient = np.dot(X_weighted.T.conj(), delta).flatten() / M
                route_request('mvn', 'update_weights', {
                    'instance_id': f'neuron_{id(neuron)}',
                    'method_params': {'delta_w': learning_rate * gradient}
                })

    def train(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000, 
              early_stopping_patience: int = 100) -> dict:
        """Network training using LLS-SM."""
        M = X.shape[0]
        training_history = {
            'rmse': [],
            'angular_rmse': [],
            'iterations': 0
        }
        
        best_rmse = float('inf')
        patience_counter = 0
        
        for iteration in range(max_iterations):
            # Forward pass through network
            all_outputs, all_z_values = self.forward_batch(X)
            
            # Getting predictions (last layer outputs)
            y_pred = all_outputs[-1]
            z_final = all_z_values[-1]
            
            # Error computation
            errors = self.compute_errors(y, y_pred, z_final)
            
            # Quality metrics
            rmse = np.sqrt(np.mean(np.abs(errors) ** 2))
            
            # Angular errors for output layer
            angular_errors = np.zeros_like(z_final, dtype=np.float64)
            for i in range(M):
                for j in range(self.layer_sizes[-1]):
                    y_angle = np.angle(y[i, j])
                    if y_angle < 0:
                        y_angle += 2 * np.pi
                    
                    sector_idx = int(np.floor(self.k * y_angle / (2 * np.pi))) % self.k
                    bisector_angle = 2 * np.pi * (sector_idx + 0.5) / self.k
                    
                    z_angle = np.angle(z_final[i, j])
                    if z_angle < 0:
                        z_angle += 2 * np.pi
                    
                    angle_diff = np.abs(z_angle - bisector_angle)
                    angular_errors[i, j] = min(angle_diff, 2 * np.pi - angle_diff)
            
            angular_rmse = np.sqrt(np.mean(angular_errors ** 2))
            
            # Training history preservation
            training_history['rmse'].append(rmse)
            training_history['angular_rmse'].append(angular_rmse)
            
            # Stopping criteria check
            if rmse <= self.theta and angular_rmse <= self.theta:
                training_history['iterations'] = iteration + 1
                break
            
            # Early stopping
            if rmse < best_rmse:
                best_rmse = rmse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    training_history['iterations'] = iteration + 1
                    break
            
            # Backpropagation of errors
            all_layer_errors = self.backpropagate(errors, all_outputs)
            
            # Weight update for each layer
            for r in range(len(self.layers)):
                layer_inputs = all_outputs[r]  # Inputs for layer r
                layer_errors = all_layer_errors[r + 1]  # Errors for layer r
                
                for j in range(self.layer_sizes[r + 1]):
                    self.lls_adjust_weights(layer_inputs, layer_errors, r, j)
        
        training_history['iterations'] = max_iterations if 'iterations' not in training_history or training_history['iterations'] == 0 else training_history['iterations']
        return training_history