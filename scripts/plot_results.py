
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def main():
    csv_path = 'results/comprehensive_results.csv'
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. Scaling Performance: Energy Gap vs N
    # Filter for Experiment 1 (shots=0, p_max_layers=2)
    subset_scaling = df[(df['shots'] == 0) & (df['p_max_layers'] == 2) & (df['mixer'] == 'xy')]
    
    if not subset_scaling.empty:
        # Group by mode and N, calculate mean and std
        grouped = subset_scaling.groupby(['mode', 'N']).agg({
            'energy_gap': ['mean', 'std'],
            'duration_sec': ['mean', 'std']
        }).reset_index()
        
        plt.figure(figsize=(10, 6))
        for mode in grouped['mode'].unique():
            data = grouped[grouped['mode'] == mode]
            plt.errorbar(
                data['N'], 
                data['energy_gap']['mean'], 
                yerr=data['energy_gap']['std'], 
                marker='o', 
                label=mode, 
                capsize=5
            )
        
        plt.title('Scalability: Energy Gap vs Problem Size (N) - Multi-Seed Average')
        plt.ylabel('Energy Gap (lower is better)')
        plt.xlabel('Number of Assets (N)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/plots/scaling_energy_gap.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        for mode in grouped['mode'].unique():
            data = grouped[grouped['mode'] == mode]
            plt.errorbar(
                data['N'], 
                data['duration_sec']['mean'], 
                yerr=data['duration_sec']['std'],
                marker='o', 
                label=mode,
                capsize=5
            )
            
        plt.title('Scalability: Runtime vs Problem Size (N) - Multi-Seed Average')
        plt.ylabel('Time (s)')
        plt.xlabel('Number of Assets (N)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/plots/scaling_runtime.png')
        plt.close()

    # 2. Warm Start Impact
    subset_warm = df[(df['N'] == 20) & (df['p_max_layers'] == 1) & (df['mixer'] == 'xy') & (df['mode'] == 'standard')]
    if not subset_warm.empty:
        grouped_warm = subset_warm.groupby('warm_start')['energy_gap'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(8, 6))
        # Bar chart with error bars
        plt.bar(
            [str(w) for w in grouped_warm['warm_start']], 
            grouped_warm['mean'],
            yerr=grouped_warm['std'],
            capsize=10
        )
        
        plt.title('Warm Start Impact (N=20, p=1) - Multi-Seed Average')
        plt.ylabel('Energy Gap')
        plt.xlabel('Warm Start (0=Cold, 1=Warm)')
        plt.grid(axis='y')
        plt.savefig('results/plots/warm_start_impact.png')
        plt.close()

    # 3. Mixer Comparison
    subset_mixer = df[(df['N'] == 16) & (df['p_max_layers'] == 1) & (df['warm_start'] == 0) & (df['mode'] == 'standard')]
    if not subset_mixer.empty:
        grouped_mixer = subset_mixer.groupby('mixer')['energy_gap'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(8, 6))
        plt.bar(
            grouped_mixer['mixer'], 
            grouped_mixer['mean'],
            yerr=grouped_mixer['std'],
            capsize=10
        )
        
        plt.title('Mixer Comparison (N=16, p=1) - Multi-Seed Average')
        plt.ylabel('Energy Gap')
        plt.xlabel('Mixer Type')
        plt.grid(axis='y')
        plt.savefig('results/plots/mixer_comparison.png')
        plt.close()

    # 4. Noise Robustness
    subset_noise = df[(df['shots'] == 1024)]
    if not subset_noise.empty:
        grouped_noise = subset_noise.groupby('noise_p')['overlap'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            grouped_noise['noise_p'], 
            grouped_noise['mean'], 
            yerr=grouped_noise['std'], 
            marker='o',
            capsize=5
        )
        
        plt.title('Noise Robustness: Overlap vs Noise Probability - Multi-Seed Average')
        plt.ylabel('Overlap with Optimal Solution')
        plt.xlabel('Noise Probability (Bitflip)')
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.savefig('results/plots/noise_robustness.png')
        plt.close()

    print("Plots generated in results/plots/")

if __name__ == '__main__':
    main()
