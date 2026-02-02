"""
Enhanced plotting script for comprehensive results
Handles all experiments including formulation comparison and CVaR objective
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

def create_plots(csv_path):
    """Generate all plots from comprehensive results"""
    
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    os.makedirs('results/plots_fixed', exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    print(f"Loaded {len(df)} results")
    print(f"Columns: {df.columns.tolist()}")
    
    # ========================================================================
    # PLOT 1: Scaling Performance (FIXED ADAPT)
    # ========================================================================
    print("\n[1/8] Generating scaling plots...")
    
    subset_scaling = df[
        (df['shots'] == 0) & 
        (df['formulation'] == 'mvo') &
        (df['mixer'] == 'xy') &
        (df['warm_start'] == 0)
    ]
    
    if not subset_scaling.empty:
        # Energy gap vs N
        grouped = subset_scaling.groupby(['mode', 'N']).agg({
            'energy_gap': ['mean', 'std'],
            'duration_sec': ['mean', 'std'],
            'layers_used': ['mean', 'std']
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Energy gap
        for mode in grouped['mode'].unique():
            data = grouped[grouped['mode'] == mode]
            ax1.errorbar(
                data['N'], 
                data['energy_gap']['mean'], 
                yerr=data['energy_gap']['std'], 
                marker='o', 
                label=mode.capitalize(), 
                capsize=5,
                linewidth=2,
                markersize=8
            )
        
        ax1.set_title('Scalability: Energy Gap vs Problem Size (FIXED ADAPT)', fontsize=14)
        ax1.set_ylabel('Energy Gap (lower is better)', fontsize=12)
        ax1.set_xlabel('Number of Assets (N)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Runtime
        for mode in grouped['mode'].unique():
            data = grouped[grouped['mode'] == mode]
            ax2.errorbar(
                data['N'], 
                data['duration_sec']['mean'], 
                yerr=data['duration_sec']['std'],
                marker='o', 
                label=mode.capitalize(),
                capsize=5,
                linewidth=2,
                markersize=8
            )
        
        ax2.set_title('Runtime vs Problem Size', fontsize=14)
        ax2.set_ylabel('Time (s)', fontsize=12)
        ax2.set_xlabel('Number of Assets (N)', fontsize=12)
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots_fixed/1_scaling_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ADAPT layers used
        adapt_data = grouped[grouped['mode'] == 'adapt']
        if not adapt_data.empty:
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                adapt_data['N'],
                adapt_data['layers_used']['mean'],
                yerr=adapt_data['layers_used']['std'],
                marker='o',
                capsize=5,
                linewidth=2,
                markersize=8,
                color='darkblue'
            )
            plt.title('ADAPT-QAOA: Layers Used vs Problem Size', fontsize=14)
            plt.ylabel('Average Layers Used', fontsize=12)
            plt.xlabel('Number of Assets (N)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig('results/plots_fixed/1b_adapt_layers.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # ========================================================================
    # PLOT 2: Warm-Start Impact (Multiple N values)
    # ========================================================================
    print("[2/8] Generating warm-start comparison...")
    
    subset_warm = df[
        (df['mixer'] == 'xy') & 
        (df['mode'] == 'standard') &
        (df['p_max_layers'] == 1)
    ]
    
    if not subset_warm.empty:
        grouped_warm = subset_warm.groupby(['N', 'warm_start'])['energy_gap'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(12, 6))
        width = 0.35
        N_values = sorted(grouped_warm['N'].unique())
        x = np.arange(len(N_values))
        
        cold_data = grouped_warm[grouped_warm['warm_start'] == 0].sort_values('N')
        warm_data = grouped_warm[grouped_warm['warm_start'] == 1].sort_values('N')
        
        plt.bar(x - width/2, cold_data['mean'], width, 
                yerr=cold_data['std'], label='Cold Start', capsize=5, alpha=0.8)
        plt.bar(x + width/2, warm_data['mean'], width,
                yerr=warm_data['std'], label='Warm Start', capsize=5, alpha=0.8)
        
        plt.xlabel('Number of Assets (N)', fontsize=12)
        plt.ylabel('Energy Gap', fontsize=12)
        plt.title('Warm-Start Impact Across Problem Sizes', fontsize=14)
        plt.xticks(x, N_values)
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots_fixed/2_warmstart_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 3: Formulation Comparison (NEW!)
    # ========================================================================
    print("[3/8] Generating formulation comparison...")
    
    subset_form = df[
        (df['N'] == 12) &
        (df['mixer'] == 'xy') &
        (df['warm_start'] == 0) &
        (df['shots'] == 0)
    ]
    
    if not subset_form.empty and len(subset_form['formulation'].unique()) > 1:
        grouped_form = subset_form.groupby(['formulation', 'mode'])['energy_gap'].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        formulations = grouped_form['formulation'].unique()
        modes = grouped_form['mode'].unique()
        x = np.arange(len(formulations))
        width = 0.35
        
        for i, mode in enumerate(modes):
            data = grouped_form[grouped_form['mode'] == mode].set_index('formulation')
            data = data.reindex(formulations)  # Ensure order
            
            ax.bar(x + i*width, data['mean'], width,
                   yerr=data['std'], label=mode.capitalize(),
                   capsize=5, alpha=0.8)
        
        ax.set_xlabel('Formulation', fontsize=12)
        ax.set_ylabel('Energy Gap', fontsize=12)
        ax.set_title('Formulation Comparison (N=12)', fontsize=14)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([f.upper() for f in formulations])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots_fixed/3_formulation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 4: CVaR Objective vs Expectation (NEW!)
    # ========================================================================
    print("[4/8] Generating CVaR objective comparison...")
    
    subset_obj = df[
        (df['N'] == 12) &
        (df['mode'] == 'standard') &
        (df['formulation'] == 'mvo')
    ]
    
    if not subset_obj.empty and 'objective' in df.columns:
        if len(subset_obj['objective'].unique()) > 1:
            grouped_obj = subset_obj.groupby(['objective', 'noise_p'])['energy_gap'].agg(['mean', 'std']).reset_index()
            
            plt.figure(figsize=(10, 6))
            
            for obj in grouped_obj['objective'].unique():
                data = grouped_obj[grouped_obj['objective'] == obj].sort_values('noise_p')
                plt.errorbar(
                    data['noise_p'],
                    data['mean'],
                    yerr=data['std'],
                    marker='o',
                    label=obj.capitalize(),
                    capsize=5,
                    linewidth=2,
                    markersize=8
                )
            
            plt.xlabel('Noise Probability (Depolarizing)', fontsize=12)
            plt.ylabel('Energy Gap', fontsize=12)
            plt.title('CVaR Objective vs Expectation Under Noise', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/plots_fixed/4_cvar_objective.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # ========================================================================
    # PLOT 5: Mixer Comparison
    # ========================================================================
    print("[5/8] Generating mixer comparison...")
    
    subset_mixer = df[
        (df['N'] == 16) & 
        (df['p_max_layers'] == 1) & 
        (df['warm_start'] == 0) & 
        (df['mode'] == 'standard')
    ]
    
    if not subset_mixer.empty:
        grouped_mixer = subset_mixer.groupby('mixer')['energy_gap'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(
            grouped_mixer['mixer'], 
            grouped_mixer['mean'],
            yerr=grouped_mixer['std'],
            capsize=10,
            alpha=0.8,
            color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(grouped_mixer)]
        )
        
        plt.title('Mixer Comparison (N=16, p=1)', fontsize=14)
        plt.ylabel('Energy Gap', fontsize=12)
        plt.xlabel('Mixer Type', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots_fixed/5_mixer_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 6: Noise Robustness (FIXED)
    # ========================================================================
    print("[6/8] Generating noise robustness plot...")
    
    subset_noise = df[(df['shots'] == 1024)]
    
    if not subset_noise.empty:
        grouped_noise = subset_noise.groupby('noise_p')['overlap'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            grouped_noise['noise_p'], 
            grouped_noise['mean'], 
            yerr=grouped_noise['std'], 
            marker='o',
            capsize=5,
            linewidth=2,
            markersize=8,
            color='darkred'
        )
        
        plt.title('Noise Robustness: Overlap vs Noise (Depolarizing)', fontsize=14)
        plt.ylabel('Overlap with Optimal Solution', fontsize=12)
        plt.xlabel('Noise Probability', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots_fixed/6_noise_robustness.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 7: ADAPT Efficiency (Gates vs Standard)
    # ========================================================================
    print("[7/8] Generating ADAPT efficiency plot...")
    
    subset_eff = df[
        (df['N'].isin([12, 16, 20])) &
        (df['formulation'] == 'mvo') &
        (df['shots'] == 0)
    ]
    
    if not subset_eff.empty:
        # Compare gate counts
        grouped_eff = subset_eff.groupby(['mode', 'N']).agg({
            'gate_two': ['mean', 'std'],
            'energy_gap': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gate counts
        for mode in grouped_eff['mode'].unique():
            data = grouped_eff[grouped_eff['mode'] == mode]
            ax1.errorbar(
                data['N'],
                data['gate_two']['mean'],
                yerr=data['gate_two']['std'],
                marker='o',
                label=mode.capitalize(),
                capsize=5,
                linewidth=2,
                markersize=8
            )
        
        ax1.set_title('Circuit Depth: 2-Qubit Gates', fontsize=14)
        ax1.set_ylabel('Number of CNOT Gates', fontsize=12)
        ax1.set_xlabel('Number of Assets (N)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Efficiency: Energy gap vs gates
        for mode in grouped_eff['mode'].unique():
            data = grouped_eff[grouped_eff['mode'] == mode]
            ax2.scatter(
                data['gate_two']['mean'],
                data['energy_gap']['mean'],
                s=data['N']*20,  # Size by N
                label=mode.capitalize(),
                alpha=0.7
            )
        
        ax2.set_title('Efficiency: Solution Quality vs Circuit Depth', fontsize=14)
        ax2.set_xlabel('Number of CNOT Gates', fontsize=12)
        ax2.set_ylabel('Energy Gap', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots_fixed/7_adapt_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 8: Summary Dashboard
    # ========================================================================
    print("[8/8] Generating summary dashboard...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Scaling
    ax1 = fig.add_subplot(gs[0, 0])
    if not subset_scaling.empty:
        for mode in grouped['mode'].unique():
            data = grouped[grouped['mode'] == mode]
            ax1.plot(data['N'], data['energy_gap']['mean'], marker='o', label=mode.capitalize())
        ax1.set_title('Scaling', fontweight='bold')
        ax1.set_xlabel('N')
        ax1.set_ylabel('Energy Gap')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Panel 2: Warm-Start
    ax2 = fig.add_subplot(gs[0, 1])
    if not subset_warm.empty:
        for ws in [0, 1]:
            data = grouped_warm[grouped_warm['warm_start'] == ws]
            label = 'Warm' if ws == 1 else 'Cold'
            ax2.plot(data['N'], data['mean'], marker='o', label=label)
        ax2.set_title('Warm-Start', fontweight='bold')
        ax2.set_xlabel('N')
        ax2.set_ylabel('Energy Gap')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: Formulation
    ax3 = fig.add_subplot(gs[1, 0])
    if not subset_form.empty and len(subset_form['formulation'].unique()) > 1:
        form_summary = grouped_form.groupby('formulation')['mean'].mean()
        ax3.bar(range(len(form_summary)), form_summary.values)
        ax3.set_xticks(range(len(form_summary)))
        ax3.set_xticklabels([f.upper() for f in form_summary.index])
        ax3.set_title('Formulations', fontweight='bold')
        ax3.set_ylabel('Avg Energy Gap')
        ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Mixer
    ax4 = fig.add_subplot(gs[1, 1])
    if not subset_mixer.empty:
        ax4.bar(range(len(grouped_mixer)), grouped_mixer['mean'].values)
        ax4.set_xticks(range(len(grouped_mixer)))
        ax4.set_xticklabels(grouped_mixer['mixer'].values)
        ax4.set_title('Mixers', fontweight='bold')
        ax4.set_ylabel('Energy Gap')
        ax4.grid(axis='y', alpha=0.3)
    
    # Panel 5: Noise
    ax5 = fig.add_subplot(gs[2, 0])
    if not subset_noise.empty:
        ax5.plot(grouped_noise['noise_p'], grouped_noise['mean'], marker='o', color='darkred')
        ax5.set_title('Noise Robustness', fontweight='bold')
        ax5.set_xlabel('Noise Probability')
        ax5.set_ylabel('Overlap')
        ax5.grid(True, alpha=0.3)
    
    # Panel 6: Key Stats
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    stats_text = f"""
    SUMMARY STATISTICS
    
    Total Experiments: {len(df)}
    Problem Sizes: {sorted(df['N'].unique())}
    
    Modes Tested: {df['mode'].unique().tolist()}
    Formulations: {df['formulation'].unique().tolist()}
    
    Best Energy Gap: {df['energy_gap'].min():.6f}
    Best Overlap: {df['overlap'].max():.4f}
    
    ADAPT Avg Layers: {df[df['mode']=='adapt']['layers_used'].mean():.1f}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('QAMP Portfolio QAOA: Comprehensive Results', fontsize=16, fontweight='bold')
    plt.savefig('results/plots_fixed/8_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print("All plots generated successfully!")
    print(f"Location: results/plots_fixed/")
    print(f"{'='*70}")

if __name__ == '__main__':
    csv_path = 'results/comprehensive_results_fixed.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    create_plots(csv_path)