import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs('results/plots', exist_ok=True)
    df = pd.read_csv('results/comprehensive_results.csv')
    def req_shots(o):
        o = max(min(float(o), 0.999999), 1e-12)
        return -4.605/np.log(1.0 - o)
    df['required_shots_99'] = df['overlap'].apply(req_shots)
    std = df[(df['mode'] == 'standard') & (df['mixer'] == 'xy')]
    pivot = std.pivot_table(index='N', columns='p_max_layers', values='required_shots_99', aggfunc='mean')
    plt.figure(figsize=(8,6))
    im = plt.imshow(pivot.values, aspect='auto', cmap='coolwarm', origin='lower')
    plt.colorbar(im, label='Required Shots (99% Confidence)')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('Circuit Depth (p)')
    plt.ylabel('Problem Size (N)')
    out = 'results/plots/viability_heatmap.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
