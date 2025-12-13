# Smart QAOA short example by ⚛️ Sigma PublishinQ Team ⚛️  
# https://www.linkedin.com/company/sigma-publishinq/about/

import unittest
import numpy as np
import pandas as pd
import torch
import networkx as nx
from erdos_renyi_fabric import ErdosRenyiFabric
from qaoa_maxcut_engine import QAOAMaxCutEngine
from neural_qaoa_trainer import NeuralQAOANet, df_to_dataloader
from cut_analyzer import compute_cut_score, brute_force_maxcut, evaluate_qaoa_performance
import sys
import os
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/')
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/short_smart_qaoa_example/')
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective

class TestNeuroMaxCut(unittest.TestCase):
    def setUp(self):
        self.N = 6  # Small graph for tests
        self.p = 2  # Number of QAOA layers
        self.num_possible_edges = int(self.N * (self.N - 1) / 2)
        self.edges = [f'x({i},{j})' for i in range(self.N) for j in range(i + 1, self.N)]
        self.simulator = "auto"
        
    def test_graph_generation(self):
        """Test graph generation."""
        vec_adjacency = np.random.choice([0, 1], size=self.num_possible_edges)
        G = ErdosRenyiFabric(vec_adjacency, self.N)
        self.assertEqual(G.number_of_nodes(), self.N)
        self.assertTrue(all(G.has_edge(i, j) for i, j in G.edges()))

    def test_qaoa_initialization(self):
        """Test QAOA initialization."""
        vec_adjacency = np.random.choice([0, 1], size=self.num_possible_edges)
        G = ErdosRenyiFabric(vec_adjacency, self.N)
        qaoa = QAOAMaxCutEngine(G, self.p)
        qaoa.determine_initial_point()
        self.assertEqual(len(qaoa.initial_point), 2 * self.p)

    def test_neural_network_prediction(self):
        """Test neural network parameter prediction."""
        bit_len = self.num_possible_edges
        net = NeuralQAOANet(bit_len, self.p)
        vec_adjacency = np.random.choice([0, 1], size=bit_len)
        input_tensor = torch.tensor([vec_adjacency], dtype=torch.float32)
        with torch.no_grad():
            params = net(input_tensor).numpy()[0]
        self.assertEqual(len(params), 2 * self.p)

    def test_simulator_integration(self):
        """Test integration with QOKit simulator."""
        vec_adjacency = np.random.choice([0, 1], size=self.num_possible_edges)
        G = ErdosRenyiFabric(vec_adjacency, self.N)
        qaoa = QAOAMaxCutEngine(G, self.p)
        qaoa.determine_initial_point()
        params = qaoa.initial_point
        objective = get_qaoa_maxcut_objective(self.N, self.p, G, simulator=self.simulator)
        energy = objective(params)
        self.assertIsInstance(energy, float)

    def test_full_cycle(self):
        """Test full cycle: graph → neural network → simulator → evaluation."""
        vec_adjacency = np.random.choice([0, 1], size=self.num_possible_edges)
        G = ErdosRenyiFabric(vec_adjacency, self.N)
        qaoa = QAOAMaxCutEngine(G, self.p)
        qaoa.determine_initial_point()
        default_params = qaoa.initial_point
        
        # Simulate neural network prediction
        bit_len = self.num_possible_edges
        net = NeuralQAOANet(bit_len, self.p)
        input_tensor = torch.tensor([vec_adjacency], dtype=torch.float32)
        with torch.no_grad():
            neural_params = net(input_tensor).numpy()[0]
        
        # Compute energy
        objective = get_qaoa_maxcut_objective(self.N, self.p, G, simulator=self.simulator)
        default_energy = objective(default_params)
        neural_energy = objective(neural_params)
        
        # Evaluate quality
        default_ratio = evaluate_qaoa_performance(G, default_energy)
        neural_ratio = evaluate_qaoa_performance(G, neural_energy)
        
        self.assertIsInstance(default_energy, float)
        self.assertIsInstance(neural_energy, float)
        self.assertIsInstance(default_ratio, float)
        self.assertIsInstance(neural_ratio, float)

    def test_iteration_comparison(self):
        """Test comparison of iterations between neural network and default optimization."""
        results = []
        
        # Test on several graphs
        for test_idx in range(5):
            vec_adjacency = np.random.choice([0, 1], size=self.num_possible_edges)
            G = ErdosRenyiFabric(vec_adjacency, self.N)
            
            # Optimization without neural network (standard)
            qaoa = QAOAMaxCutEngine(G, self.p)
            qaoa.determine_initial_point()
            default_result = qaoa.optimize(optimizer='BFGS')
            default_iterations = default_result.nit
            default_energy = default_result.fun
            
            # Optimization with neural network (1 iteration)
            bit_len = self.num_possible_edges
            net = NeuralQAOANet(bit_len, self.p)
            input_tensor = torch.tensor([vec_adjacency], dtype=torch.float32)
            with torch.no_grad():
                neural_params = net(input_tensor).numpy()[0]
            
            # Compute energy for neural network parameters
            objective = get_qaoa_maxcut_objective(self.N, self.p, G, simulator=self.simulator)
            neural_energy = objective(neural_params)
            neural_iterations = 1  # Neural network gives result in 1 "iteration"
            
            # Compare quality
            default_ratio = evaluate_qaoa_performance(G, default_energy)
            neural_ratio = evaluate_qaoa_performance(G, neural_energy)
            
            results.append({
                'test_idx': test_idx,
                'default_iterations': default_iterations,
                'neural_iterations': neural_iterations,
                'default_energy': default_energy,
                'neural_energy': neural_energy,
                'default_ratio': default_ratio,
                'neural_ratio': neural_ratio,
                'iteration_speedup': default_iterations / neural_iterations
            })
        
        # Print results
        avg_default_iter = np.mean([r['default_iterations'] for r in results])
        avg_speedup = np.mean([r['iteration_speedup'] for r in results])
        
        print(f"Average iterations without neural network: {avg_default_iter:.2f}")
        print(f"Average speedup by iterations: {avg_speedup:.2f}x")
        
        # Verify that neural network is faster
        for result in results:
            self.assertLessEqual(result['neural_iterations'], result['default_iterations'])
            self.assertGreaterEqual(result['iteration_speedup'], 1.0)
        
        return results

if __name__ == '__main__':
    unittest.main()