import os
import pandas as pd
import matplotlib.pyplot as plt

def load(path):
    return pd.read_csv(path)

def main():
    os.makedirs('results/plots', exist_ok=True)
    dfs = []
    for fn in ['results/adapt_efficiency_N12.csv','results/adapt_efficiency_N16.csv','results/adapt_efficiency_N20.csv']:
        if os.path.exists(fn):
            dfs.append(load(fn))
    if not dfs:
        print("No input CSVs found")
        return
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df['mixer'] == 'xy') & (df['warm_start'] == 1)]
    Ns = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(5*len(Ns), 5), sharey=True)
    if len(Ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, Ns):
        sub = df[df['N'] == n]
        for mode in ['standard','adapt']:
            cur = sub[sub['mode'] == mode]
            # Aggregate by gate_two to smooth relation
            agg = cur.groupby('gate_two')['energy_gap'].mean().reset_index().sort_values('gate_two')
            ax.plot(agg['gate_two'], agg['energy_gap'], marker='o' if mode=='standard' else '^', label=mode)
        ax.set_title(f"N={n}")
        ax.set_xlabel('Two-Qubit Gate Count')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.grid(True)
    axes[0].set_ylabel('Energy Gap (E - Eopt)')
    axes[-1].legend(loc='best')
    out = 'results/plots/adapt_efficiency_comprehensive.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
