
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_large_scale_results():
    # Load data
    df = pd.read_csv('results/large_scale_results.csv')
    
    # Create plots directory if not exists
    os.makedirs('results/plots', exist_ok=True)
    
    # Calculate subspace size for each row
    from scipy.special import comb
    df['subspace_size'] = df.apply(lambda row: comb(row['N'], row['K']), axis=1)
    df['label'] = df.apply(lambda row: f"N={int(row['N'])}, K={int(row['K'])}", axis=1)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Runtime vs Subspace Size (Log Scale)
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df, x='subspace_size', y='duration_sec', hue='label', s=100, style='label')
    
    # Add theoretical line O(M) or O(M log M)
    # Just a visual guide
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Runtime vs Subspace Size (Simulated QAOA)', fontsize=14)
    plt.xlabel('Subspace Size (Number of Basis States)', fontsize=12)
    plt.ylabel('Duration (seconds)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/plots/large_scale_runtime.png', dpi=300)
    plt.close()
    
    # Plot 2: Approximation Ratio vs N/K
    # Approx Ratio = best_energy / optimal_energy (Note: energies are negative usually for minimization, 
    # but here they might be mixed. Better to use (Best - Min) / |Min| or similar if signs match.
    # Actually, QAOA aims to MINIMIZE energy. 
    # So Gap = Best - Optimal (should be >= 0).
    # Let's plot Gap.
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='label', y='energy_gap', hue='seed')
    plt.title('Energy Gap (Best - Optimal) for Large Scale Instances', fontsize=14)
    plt.xlabel('Instance Configuration', fontsize=12)
    plt.ylabel('Energy Gap (Lower is Better)', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/plots/large_scale_gap.png', dpi=300)
    plt.close()
    
    # Plot 3: Overlap
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='label', y='overlap', hue='seed')
    plt.title('Probability of Finding Optimal Solution (Overlap)', fontsize=14)
    plt.xlabel('Instance Configuration', fontsize=12)
    plt.ylabel('Overlap Probability', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/plots/large_scale_overlap.png', dpi=300)
    plt.close()

    print("Plots generated in results/plots/")

if __name__ == "__main__":
    plot_large_scale_results()
