import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    in_csv = 'results/large_scale_results.csv'
    if not os.path.exists(in_csv) or os.path.getsize(in_csv) == 0:
        print(f"{in_csv} not found or empty")
        return
    os.makedirs('results/plots', exist_ok=True)
    df = pd.read_csv(in_csv)
    # Group by N, aggregate across seeds/configs
    grp = df.groupby('N').agg({
        'energy_gap': ['mean', 'std', 'count'],
        'duration_sec': ['mean', 'std'],
        'gate_two': ['mean', 'std']
    }).reset_index()
    # Flatten columns
    grp.columns = ['N',
                   'gap_mean','gap_std','count',
                   'dur_mean','dur_std',
                   'cnot_mean','cnot_std']
    # Save summary CSV
    out_csv = 'results/large_scale_summary.csv'
    grp.to_csv(out_csv, index=False)
    # Plot gap and runtime
    plt.figure(figsize=(10,6))
    plt.errorbar(grp['N'], grp['gap_mean'], yerr=grp['gap_std'], marker='o', capsize=5)
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Energy Gap (mean ± std)')
    plt.grid(True)
    out1 = 'results/plots/large_scale_summary_gap.png'
    plt.savefig(out1, dpi=200)
    plt.close()
    plt.figure(figsize=(10,6))
    plt.errorbar(grp['N'], grp['dur_mean'], yerr=grp['dur_std'], marker='o', capsize=5)
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Runtime (s) (mean ± std)')
    plt.grid(True)
    out2 = 'results/plots/large_scale_summary_runtime.png'
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved {out_csv}, {out1}, {out2}")

if __name__ == '__main__':
    main()
