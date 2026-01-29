
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
    # Show Standard (p=2) vs ADAPT (p=3), both xy, shots=0
    subset_scaling_std = df[(df['shots'] == 0) & (df['p_max_layers'] == 2) & (df['mixer'] == 'xy') & (df['mode'] == 'standard')]
    subset_scaling_adapt = df[(df['shots'] == 0) & (df['p_max_layers'] == 3) & (df['mixer'] == 'xy') & (df['mode'] == 'adapt')]
    subset_scaling = pd.concat([subset_scaling_std, subset_scaling_adapt], ignore_index=True)
    
    if not subset_scaling.empty:
        # Group by mode and N, calculate mean and std
        grouped = subset_scaling.groupby(['mode', 'N'], as_index=False).agg({
            'energy_gap': ['mean', 'std'],
            'duration_sec': ['mean', 'std']
        })
        
        plt.figure(figsize=(10, 6))
        for mode in grouped['mode'].unique():
            data = grouped[grouped[('mode', '')] == mode] if isinstance(grouped.columns, pd.MultiIndex) else grouped[grouped['mode'] == mode]
            plt.errorbar(
                data[('N', '')] if isinstance(grouped.columns, pd.MultiIndex) else data['N'],
                data[('energy_gap', 'mean')] if isinstance(grouped.columns, pd.MultiIndex) else data['energy_gap'],
                yerr=data[('energy_gap', 'std')] if isinstance(grouped.columns, pd.MultiIndex) else None,
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
            data = grouped[grouped[('mode', '')] == mode] if isinstance(grouped.columns, pd.MultiIndex) else grouped[grouped['mode'] == mode]
            plt.errorbar(
                data[('N', '')] if isinstance(grouped.columns, pd.MultiIndex) else data['N'],
                data[('duration_sec', 'mean')] if isinstance(grouped.columns, pd.MultiIndex) else data['duration_sec'],
                yerr=data[('duration_sec', 'std')] if isinstance(grouped.columns, pd.MultiIndex) else None,
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

    # 2. Warm Start Impact (Standard, N=20, p=1)
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

    # 2b. Warm vs Cold for ADAPT (p=3), across N
    subset_adapt_wc = df[(df['mode'] == 'adapt') & (df['p_max_layers'] == 3) & (df['mixer'] == 'xy')]
    if not subset_adapt_wc.empty:
        grouped_adapt_wc = subset_adapt_wc.groupby(['N', 'warm_start'])['energy_gap'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(10, 6))
        for warm in sorted(grouped_adapt_wc['warm_start'].unique()):
            data = grouped_adapt_wc[grouped_adapt_wc['warm_start'] == warm]
            plt.errorbar(
                data['N'],
                data['mean'],
                yerr=data['std'],
                marker='o',
                label=f"adapt warm_start={warm}",
                capsize=5
            )
        plt.title('ADAPT-QAOA: Warm vs Cold (p=3, xy)')
        plt.ylabel('Energy Gap')
        plt.xlabel('Number of Assets (N)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/plots/adapt_warm_vs_cold.png')
        plt.close()

    # 5. Overlay: Standard vs ADAPT (cold/warm)
    std_overlay = df[(df['mode'] == 'standard') & (df['mixer'] == 'xy') & (df['shots'] == 0) & (df['p_max_layers'] == 2)]
    adapt_cold_overlay = df[(df['mode'] == 'adapt') & (df['mixer'] == 'xy') & (df['shots'] == 0) & (df['p_max_layers'] == 3) & (df['warm_start'] == 0)]
    adapt_warm_overlay = df[(df['mode'] == 'adapt') & (df['mixer'] == 'xy') & (df['shots'] == 0) & (df['p_max_layers'] == 3) & (df['warm_start'] == 1)]
    if not std_overlay.empty and not adapt_cold_overlay.empty and not adapt_warm_overlay.empty:
        g_std = std_overlay.groupby('N')['energy_gap'].agg(['mean', 'std']).reset_index()
        g_cold = adapt_cold_overlay.groupby('N')['energy_gap'].agg(['mean', 'std']).reset_index()
        g_warm = adapt_warm_overlay.groupby('N')['energy_gap'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(10, 6))
        plt.errorbar(g_std['N'], g_std['mean'], yerr=g_std['std'], marker='o', label='standard (p=2)', capsize=5)
        plt.errorbar(g_cold['N'], g_cold['mean'], yerr=g_cold['std'], marker='o', label='adapt cold (p=3)', capsize=5)
        plt.errorbar(g_warm['N'], g_warm['mean'], yerr=g_warm['std'], marker='o', label='adapt warm (p=3)', capsize=5)
        plt.title('Standard vs ADAPT (Cold/Warm) - Energy Gap vs N')
        plt.ylabel('Energy Gap')
        plt.xlabel('Number of Assets (N)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/plots/standard_vs_adapt_overlay.png')
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
