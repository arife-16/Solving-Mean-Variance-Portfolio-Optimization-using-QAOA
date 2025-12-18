"""Quantum validation suite - P1 contribution"""
import numpy as np
import copy

class QuantumValidator:
    """Tests for quantum operations"""
    
    def verify_hamming_weight_preservation(self, unitary_matrix, n_qubits, budget):
        """Test if mixer preserves K-hot subspace"""
        print(f"Testing Hamming Weight {budget} Preservation:")
        
        valid_indices = []
        for z in range(1 << n_qubits):
            if bin(z).count('1') == budget:
                valid_indices.append(z)
        
        print(f"  Valid states: {len(valid_indices)}")
        
        threshold = 1e-10
        preserves = True
        
        for col_idx in valid_indices:
            column = unitary_matrix[:, col_idx]
            significant = np.abs(column) > threshold
            nonzero_rows = np.where(significant)[0]
            
            for row_idx in nonzero_rows:
                if row_idx not in valid_indices:
                    print(f"  ✗ FAIL: Leaked to invalid state")
                    preserves = False
        
        if preserves:
            print("  ✓ PASS: Subspace preserved")
        return preserves
    
    def verify_hermiticity(self, matrix):
        """Test H = H†"""
        is_hermitian = np.allclose(matrix, matrix.conj().T)
        print(f"Hermiticity: {'✓ PASS' if is_hermitian else '✗ FAIL'}")
        return is_hermitian
    
    def verify_unitary(self, matrix):
        """Test U·U† = I"""
        product = matrix @ matrix.conj().T
        identity = np.eye(matrix.shape[0])
        is_unitary = np.allclose(product, identity)
        print(f"Unitarity: {'✓ PASS' if is_unitary else '✗ FAIL'}")
        return is_unitary
    
    def check_gradient_consistency(self, energy_function, gradient_function, test_point, epsilon=1e-5):
        """Compare analytical vs numerical gradient"""
        print("Testing Gradient Consistency...")
        
        analytical = np.array(gradient_function(test_point))
        numerical = np.zeros_like(analytical)
        
        for i in range(len(test_point)):
            params_plus = copy.deepcopy(test_point)
            params_plus[i] += epsilon
            e_plus = energy_function(params_plus)
            
            params_minus = copy.deepcopy(test_point)
            params_minus[i] -= epsilon
            e_minus = energy_function(params_minus)
            
            numerical[i] = (e_plus - e_minus) / (2 * epsilon)
        
        diff = np.linalg.norm(analytical - numerical)
        is_consistent = diff < 1e-3
        
        print(f"  Difference: {diff:.6f} → {'✓ PASS' if is_consistent else '✗ FAIL'}")
        return is_consistent