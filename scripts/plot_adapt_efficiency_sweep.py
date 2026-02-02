import os, sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs('results/plots', exist_ok=True)
    df = pd.read_csv('results/adapt_efficiency_sweep.csv')
    # Filter N=16, mixer xy, warm_start=1
    df = df[(df['N'] == 16) & (df['mixer'] == 'xy') & (df['warm_start'] == 1)]
    # Standard
    std = df[df['mode'] == 'standard']
    # Adapt
    ada = df[df['mode'] == 'adapt']
    plt.figure(figsize=(8,5))
    # Standard: plot gate_two vs energy_gap for each p
    plt.plot(std['gate_two'], std['energy_gap'], marker='o', linestyle='-', label='Standard QAOA')
    # Adapt: plot gate_two vs energy_gap points
    plt.scatter(ada['gate_two'], ada['energy_gap'], marker='^', label='ADAPT-QAOA (Gradient)')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Two-Qubit Gate Count')
    plt.ylabel('Energy Gap (E - Eopt)')
    plt.legend()
    out = 'results/plots/adapt_efficiency_sweep.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
