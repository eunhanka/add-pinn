"""Plot sensor sensitivity: L2% vs sensor count for B2, B5, B6.

v5: Legend outside plot (top, 3-col), Dense label near bottom.

Usage: python plot_sensor_sensitivity_v5.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, '..', 'tables')
OUT_DIR = os.path.join(SCRIPT_DIR, 'output_final')

METHOD_STYLE = {
    'B3': ('PINN (B2)',      '#e53935', 'o'),
    'B8': ('XPINN (B5)',     '#1e88e5', 's'),
    'B7': ('ADD-PINN (B6)',  '#43a047', 'D'),
}

I24_DATASETS = ['20221121', '20221122', '20221123', '20221129', '20221202']
SENSOR_COUNTS = [3, 4, 5, 6, 7]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(TABLES_DIR, 'per_seed_all.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['L2'])
    df = df[df['Dataset'].isin(I24_DATASETS)]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Regime shading
    ax.axvspan(2.8, 5.5, alpha=0.06, color='#e53935', zorder=0)
    ax.axvspan(5.5, 7.2, alpha=0.06, color='#1e88e5', zorder=0)
    # Sparse label at top, Dense label near bottom
    ax.text(4.15, 0.97, 'Sparse regime', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, color='#b71c1c', alpha=0.7,
            fontstyle='italic')
    ax.text(6.35, 0.04, 'Dense regime', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8, color='#0d47a1', alpha=0.7,
            fontstyle='italic')

    for code, (label, color, marker) in METHOD_STYLE.items():
        sub = df[df['Method'] == code]
        means = []
        seed_stds = []
        for n_s in SENSOR_COUNTS:
            vals_all = sub[sub['Sensors'] == n_s]
            means.append(vals_all['L2'].mean() * 100)
            per_ds_stds = []
            for ds in I24_DATASETS:
                ds_vals = vals_all[vals_all['Dataset'] == ds]['L2'].values * 100
                if len(ds_vals) > 1:
                    per_ds_stds.append(ds_vals.std())
            seed_stds.append(np.mean(per_ds_stds) if per_ds_stds else 0)

        means = np.array(means)
        seed_stds = np.array(seed_stds)

        ax.errorbar(SENSOR_COUNTS, means, yerr=seed_stds,
                    color=color, marker=marker, markersize=10,
                    linewidth=1.5, capsize=2, capthick=1.0,
                    elinewidth=1.0, label=label, zorder=3)

    ax.set_xlabel('Number of sensors')
    ax.set_ylabel('$L_2$ relative error (%)')
    ax.set_xticks(SENSOR_COUNTS)
    ax.legend(fontsize=9, framealpha=0.9,
              loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.grid(True, alpha=0.3, linewidth=0.5, color='gray')
    ax.set_xlim(2.8, 7.2)

    fig.tight_layout()

    base = 'sensor_sensitivity'
    fig.savefig(os.path.join(OUT_DIR, f'{base}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'{base}.png'))
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


if __name__ == '__main__':
    main()
