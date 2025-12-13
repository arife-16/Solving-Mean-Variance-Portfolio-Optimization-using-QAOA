import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class IterationAnalyzer:
    """Class for analyzing the number of iterations."""
    
    def __init__(self, train_data_path: str, test_results_path: str):
        self.train_data = pd.read_csv(train_data_path)
        self.test_results = pd.read_csv(test_results_path)
    
    def analyze_training_iterations(self) -> Dict:
        """Analyze iterations on training data."""
        if 'iterations' not in self.train_data.columns:
            raise ValueError("Column 'iterations' not found in training data")
        
        stats = {
            'mean_iterations': self.train_data['iterations'].mean(),
            'median_iterations': self.train_data['iterations'].median(),
            'std_iterations': self.train_data['iterations'].std(),
            'min_iterations': self.train_data['iterations'].min(),
            'max_iterations': self.train_data['iterations'].max()
        }
        
        return stats
    
    def compare_methods(self) -> Dict:
        """Compare methods by number of iterations."""
        train_stats = self.analyze_training_iterations()
        
        # Neural network always uses 1 "iteration" (forward pass)
        neural_iterations = 1
        
        comparison = {
            'default_method': train_stats,
            'neural_method': {
                'mean_iterations': neural_iterations,
                'median_iterations': neural_iterations,
                'std_iterations': 0,
                'min_iterations': neural_iterations,
                'max_iterations': neural_iterations
            },
            'speedup_factor': train_stats['mean_iterations'] / neural_iterations
        }
        
        return comparison
    
    def plot_iteration_distribution(self):
        """Plot iteration distribution."""
        plt.figure(figsize=(12, 5))
        
        # Histogram of iterations during training
        plt.subplot(1, 2, 1)
        plt.hist(self.train_data['iterations'], bins=30, alpha=0.7, color='blue')
        plt.xlabel('Number of iterations')
        plt.ylabel('Frequency')
        plt.title('Iteration Distribution\n(Standard Optimization)')
        plt.axvline(self.train_data['iterations'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.train_data["iterations"].mean():.1f}')
        plt.legend()
        
        # Method comparison
        plt.subplot(1, 2, 2)
        methods = ['Standard\nOptimization', 'Neural Network']
        iterations = [self.train_data['iterations'].mean(), 1]
        colors = ['blue', 'green']
        
        bars = plt.bar(methods, iterations, color=colors, alpha=0.7)
        plt.ylabel('Average number of iterations')
        plt.title('Method Comparison')
        
        # Add values on bars
        for bar, value in zip(bars, iterations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """Generate iteration comparison report."""
        comparison = self.compare_methods()
        
        report = f"""
# Iteration Comparison Report

## Standard Optimization (Training Data):
- Mean iterations: {comparison['default_method']['mean_iterations']:.2f}
- Median: {comparison['default_method']['median_iterations']:.2f}
- Standard deviation: {comparison['default_method']['std_iterations']:.2f}
- Minimum: {comparison['default_method']['min_iterations']}
- Maximum: {comparison['default_method']['max_iterations']}

## Neural Network Method:
- Number of iterations: {comparison['neural_method']['mean_iterations']}
- Deterministic result in 1 forward pass

## Speedup:
- Speedup factor: {comparison['speedup_factor']:.2f}x
- Iteration savings: {comparison['default_method']['mean_iterations'] - 1:.2f}

## Conclusions:
The neural network approach provides significant speedup in terms of iteration count,
but requires pre-training on a large number of graphs.
"""
        
        return report