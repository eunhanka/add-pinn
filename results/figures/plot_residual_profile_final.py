"""Plot R(x) spatial residual profile with peaks and split positions.

v6: Only show the two major peaks (x~0.70 and x~0.93), remove minor ones.

Usage: python plot_residual_profile_v6.py --sensors 5
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'output_final')

SUB_COLORS = ['#e8f0fe', '#fff3e0', '#e8f5e9', '#fce4ec',
              '#f3e5f5', '#e0f7fa']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensors', type=int, default=3)
    args = parser.parse_args()
    n_s = args.sensors

    os.makedirs(OUT_DIR, exist_ok=True)

    rp_path = os.path.join(DATA_DIR, f'residual_profile_{n_s}s.npy')
    if not os.path.exists(rp_path):
        print(f"No residual profile for {n_s} sensors. Skipping.")
        return

    rp_data = np.load(rp_path)
    x_vals = rp_data[:, 0]
    R_x = rp_data[:, 1]
    R_smooth = rp_data[:, 2]

    si_path = os.path.join(DATA_DIR, f'split_info_{n_s}s.json')
    with open(si_path) as f:
        split_info = json.load(f)

    split_positions = split_info['split_positions']
    subdomain_bounds = split_info['subdomain_bounds']

    sensor_x = np.load(os.path.join(DATA_DIR, f'sensor_x_{n_s}s.npy'))

    r_max = R_smooth.max()

    # Find all local maxima, then keep only the top 2
    all_peaks = []
    margin = max(1, int(len(R_smooth) * 0.05))
    for i in range(margin, len(R_smooth) - margin):
        if (R_smooth[i] > R_smooth[i - 1] and R_smooth[i] > R_smooth[i + 1]
                and R_smooth[i] > 0.20 * r_max):
            all_peaks.append((float(x_vals[i]), float(R_smooth[i])))

    # Keep only the 2 tallest peaks
    all_peaks.sort(key=lambda p: p[1], reverse=True)
    major_peaks = all_peaks[:2]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Shade subdomain regions
    for i, sd in enumerate(subdomain_bounds):
        color = SUB_COLORS[i % len(SUB_COLORS)]
        ax.axvspan(sd['x_min'], sd['x_max'], alpha=0.6,
                   color=color, zorder=0)

    # Plot R(x) raw and smoothed
    ax.plot(x_vals, R_x, color='#666666', linewidth=0.5,
            alpha=0.5, label='$R(x)$ raw')
    ax.plot(x_vals, R_smooth, color='#1565c0', linewidth=1.5,
            label='$R(x)$ smoothed')

    # Major peaks only (red stars)
    for px, pv in major_peaks:
        ax.plot(px, pv, marker='*', color='#d32f2f', markersize=15,
                markeredgecolor='#b71c1c', markeredgewidth=0.5, zorder=5)
    if major_peaks:
        ax.plot([], [], marker='*', color='#d32f2f', markersize=15,
                markeredgecolor='#b71c1c', linestyle='None',
                label='Residual peaks')

    # Split positions (blue vertical lines + valley marker)
    for sp in split_positions:
        ax.axvline(sp, color='#1565c0', linestyle='-',
                   linewidth=1.2, alpha=0.8, zorder=4)
        sp_idx = np.argmin(np.abs(x_vals - sp))
        r_at_split = R_smooth[sp_idx]
        ax.plot(sp, r_at_split, 'o', color='#1565c0', markersize=8,
                markeredgecolor='#0d47a1', markeredgewidth=1.0, zorder=6)

    if split_positions:
        ax.plot([], [], color='#1565c0', linestyle='-',
                linewidth=1.2, label='Split positions')
        ax.plot([], [], 'o', color='#1565c0', markersize=8,
                markeredgecolor='#0d47a1', linestyle='None',
                label='Valley (split point)')

    # Sensor positions (green dashed)
    for sx in sensor_x:
        ax.axvline(sx, color='#2e7d32', linestyle='--',
                   linewidth=0.8, alpha=0.6, zorder=3)
    if len(sensor_x) > 0:
        ax.plot([], [], color='#2e7d32', linestyle='--',
                linewidth=0.8, label='Sensor locations')

    # --- Mechanism annotations for the 2 major peaks ---
    # Sort by x position for consistent placement
    major_peaks_sorted = sorted(major_peaks, key=lambda p: p[0])

    if len(major_peaks_sorted) >= 1:
        # Left peak (x~0.70): annotate to the right
        pk_x, pk_v = major_peaks_sorted[0]
        ax.annotate(
            'High $R(x)$\nshock-prone region',
            xy=(pk_x, pk_v),
            xytext=(pk_x + 0.10, pk_v + 0.12 * r_max),
            fontsize=8, ha='left', color='#b71c1c',
            arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f8',
                      ec='#d32f2f', alpha=0.9),
            zorder=7)

    if len(major_peaks_sorted) >= 2:
        # Right peak (x~0.93): annotate to the left
        pk_x, pk_v = major_peaks_sorted[1]
        ax.annotate(
            'High $R(x)$\nshock-prone region',
            xy=(pk_x, pk_v),
            xytext=(pk_x - 0.15, pk_v + 0.10 * r_max),
            fontsize=8, ha='center', color='#b71c1c',
            arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f8',
                      ec='#d32f2f', alpha=0.9),
            zorder=7)

    # Valley annotation
    if split_positions:
        sp = split_positions[0]
        sp_idx = np.argmin(np.abs(x_vals - sp))
        r_at_split = R_smooth[sp_idx]
        ax.annotate(
            'Low $R(x)$ valley\n$\\rightarrow$ split position',
            xy=(sp, r_at_split),
            xytext=(sp - 0.18, -0.15 * r_max),
            fontsize=8, ha='center', color='#0d47a1',
            arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.2,
                            connectionstyle='arc3,rad=0.15'),
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0f4ff',
                      ec='#1565c0', alpha=0.9),
            zorder=7)

    ax.set_xlabel('Normalized spatial coordinate $\\hat{x}$')
    ax.set_ylabel('$R(x)$: time-averaged\nPDE residual$^2$')
    ax.set_xlim(x_vals[0], x_vals[-1])
    ax.set_ylim(bottom=-0.20 * r_max, top=r_max * 1.30)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9, ncol=2)

    fig.tight_layout()

    base = f'residual_profile_{n_s}s'
    fig.savefig(os.path.join(OUT_DIR, f'{base}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'{base}.png'))
    plt.close(fig)
    print(f"Saved {base}.pdf/.png "
          f"({len(subdomain_bounds)} subdomains, "
          f"{len(split_positions)} splits, "
          f"{len(major_peaks)} peaks shown)")


if __name__ == '__main__':
    main()
