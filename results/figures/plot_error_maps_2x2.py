"""Plot error maps in 2x2 layout with improvement panel.

Usage: python plot_error_maps_2x2.py --sensors 3
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
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

ZOOM_T = (0.8, 1.6)
ZOOM_X = (0.6, 1.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensors', type=int, default=3)
    args = parser.parse_args()
    n_s = args.sensors

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, 'axis_info.json')) as f:
        info = json.load(f)

    gt = np.load(os.path.join(DATA_DIR, 'gt_20221121.npy'))
    sensor_x = np.load(os.path.join(DATA_DIR, f'sensor_x_{n_s}s.npy'))

    n_t, n_x = gt.shape
    t_hours = np.linspace(0, info['t_range_hours'], n_t)
    x_miles = np.linspace(0, info['x_range_miles'], n_x)
    sensor_miles = sensor_x * info['x_range_miles']
    extent = [t_hours[0], t_hours[-1], x_miles[0], x_miles[-1]]

    # Zoom masks
    ti = np.where((t_hours >= ZOOM_T[0]) & (t_hours <= ZOOM_T[1]))[0]
    xi = np.where((x_miles >= ZOOM_X[0]) & (x_miles <= ZOOM_X[1]))[0]

    errors = {
        'B3': np.load(os.path.join(DATA_DIR, f'error_B3_{n_s}s.npy')),
        'B8': np.load(os.path.join(DATA_DIR, f'error_B8_{n_s}s.npy')),
        'B7': np.load(os.path.join(DATA_DIR, f'error_B7_{n_s}s.npy')),
    }
    improvement = np.abs(errors['B3']) - np.abs(errors['B7'])
    delta_mae_zoom = float(improvement[np.ix_(ti, xi)].mean())

    vlim_err = 15
    vlim_imp = 10
    rect_w = ZOOM_T[1] - ZOOM_T[0]
    rect_h = ZOOM_X[1] - ZOOM_X[0]

    # --- Layout: 2x2 main + colorbar row ---
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.05],
                  hspace=0.25, wspace=0.12)

    # Error panels: (a) top-left, (b) top-right, (c) bottom-left
    err_panels = [
        (gs[0, 0], 'B3', '(a) Vanilla PINN (B2)'),
        (gs[0, 1], 'B8', '(b) XPINN (B5)'),
        (gs[1, 0], 'B7', '(c) ADD-PINN (B6)'),
    ]

    im_err = None
    err_axes = []
    for pos, code, label in err_panels:
        ax = fig.add_subplot(pos)
        err_axes.append(ax)
        err = errors[code]
        pred = err + gt
        l2 = float(np.linalg.norm(pred - gt) / np.linalg.norm(gt))
        mae = float(np.abs(err).mean())

        im_err = ax.imshow(
            err.T, cmap='RdBu_r', vmin=-vlim_err, vmax=vlim_err,
            aspect='auto', origin='lower', extent=extent,
            interpolation='nearest', rasterized=True)

        for sm in sensor_miles:
            ax.axhline(sm, color='gray', linestyle='-',
                       linewidth=0.5, alpha=0.4)

        rect = Rectangle((ZOOM_T[0], ZOOM_X[0]), rect_w, rect_h,
                          linewidth=2.0, edgecolor='black',
                          facecolor='none', linestyle='-', zorder=8)
        ax.add_patch(rect)

        ax.set_title(f'{label}\n$L_2$ = {l2*100:.2f}%,  MAE = {mae:.2f} mph',
                     fontsize=10)
        ax.set_xlim(t_hours[0], t_hours[-1])
        ax.set_ylim(x_miles[0], x_miles[-1])

    # Improvement panel: (d) bottom-right
    ax_imp = fig.add_subplot(gs[1, 1])
    im_imp = ax_imp.imshow(
        improvement.T, cmap='RdBu', vmin=-vlim_imp, vmax=vlim_imp,
        aspect='auto', origin='lower', extent=extent,
        interpolation='nearest', rasterized=True)

    for sm in sensor_miles:
        ax_imp.axhline(sm, color='gray', linestyle='-',
                       linewidth=0.5, alpha=0.4)

    rect = Rectangle((ZOOM_T[0], ZOOM_X[0]), rect_w, rect_h,
                      linewidth=2.0, edgecolor='black',
                      facecolor='none', linestyle='-', zorder=8)
    ax_imp.add_patch(rect)

    ax_imp.annotate(
        f'$\\Delta$MAE = {delta_mae_zoom:.2f} mph',
        xy=(ZOOM_T[1], ZOOM_X[1]),
        xytext=(ZOOM_T[1] + 0.25, ZOOM_X[1] + 0.4),
        fontsize=9, ha='left', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec='black', alpha=0.9),
        zorder=9)

    ax_imp.set_title(
        '(d) $|e_{\\mathrm{B2}}| - |e_{\\mathrm{B6}}|$\n'
        'Blue = ADD-PINN better',
        fontsize=10)
    ax_imp.set_xlim(t_hours[0], t_hours[-1])
    ax_imp.set_ylim(x_miles[0], x_miles[-1])

    # Axis labels
    err_axes[0].set_ylabel('Space (miles)')     # top-left
    err_axes[2].set_ylabel('Space (miles)')     # bottom-left
    err_axes[2].set_xlabel('Time (hours)')      # bottom-left
    ax_imp.set_xlabel('Time (hours)')           # bottom-right
    err_axes[0].set_xticklabels([])             # top-left
    err_axes[1].set_xticklabels([])             # top-right
    err_axes[1].set_yticklabels([])             # top-right
    ax_imp.set_yticklabels([])                  # bottom-right

    # Colorbars in bottom row
    # Left: error colorbar spanning left half
    cbar_ax1 = fig.add_subplot(gs[2, 0])
    cbar1 = fig.colorbar(im_err, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Error (mph)')

    # Right: improvement colorbar spanning right half
    cbar_ax2 = fig.add_subplot(gs[2, 1])
    cbar2 = fig.colorbar(im_imp, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Improvement (mph)')

    base = f'error_maps_{n_s}s_2x2'
    fig.savefig(os.path.join(OUT_DIR, f'{base}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'{base}.png'))
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


if __name__ == '__main__':
    main()
