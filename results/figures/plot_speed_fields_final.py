"""Plot speed field heatmaps with zoom strip.

Final: 3-row layout with zoom strip showing congestion core.

Usage: python plot_speed_fields_final.py --sensors 3
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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

PANELS = [
    ('GT',  'Ground Truth'),
    ('B3',  'Vanilla PINN (B2)'),
    ('B8',  'XPINN (B5)'),
    ('B7',  'ADD-PINN (B6)'),
]
PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)']

ZOOM_T = (0.8, 1.6)
ZOOM_X = (0.6, 1.8)


def load_data(n_sensors):
    with open(os.path.join(DATA_DIR, 'axis_info.json')) as f:
        info = json.load(f)
    gt = np.load(os.path.join(DATA_DIR, 'gt_20221121.npy'))
    sensor_x = np.load(os.path.join(DATA_DIR, f'sensor_x_{n_sensors}s.npy'))
    preds = {}
    for code, _ in PANELS:
        if code == 'GT':
            continue
        preds[code] = np.load(
            os.path.join(DATA_DIR, f'pred_{code}_{n_sensors}s.npy'))
    return info, gt, preds, sensor_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensors', type=int, default=3)
    args = parser.parse_args()
    n_s = args.sensors

    os.makedirs(OUT_DIR, exist_ok=True)
    info, gt, preds, sensor_x = load_data(n_s)

    n_t, n_x = gt.shape
    t_hours = np.linspace(0, info['t_range_hours'], n_t)
    x_miles = np.linspace(0, info['x_range_miles'], n_x)
    sensor_miles = sensor_x * info['x_range_miles']

    vmin, vmax = 0, 65
    extent_full = [t_hours[0], t_hours[-1], x_miles[0], x_miles[-1]]
    extent_zoom = [ZOOM_T[0], ZOOM_T[1], ZOOM_X[0], ZOOM_X[1]]

    # Zoom region masks
    t_mask = (t_hours >= ZOOM_T[0]) & (t_hours <= ZOOM_T[1])
    x_mask = (x_miles >= ZOOM_X[0]) & (x_miles <= ZOOM_X[1])
    ti = np.where(t_mask)[0]
    xi = np.where(x_mask)[0]

    # Build fields dict
    fields = {'GT': gt}
    fields.update(preds)

    # Compute L2 for each method
    l2_vals = {}
    for code in ['B3', 'B8', 'B7']:
        l2_vals[code] = float(
            np.linalg.norm(gt - fields[code]) / np.linalg.norm(gt))

    # Compute zoom MAE for each
    zoom_mae = {}
    for code in ['GT', 'B3', 'B8', 'B7']:
        if code == 'GT':
            zoom_mae[code] = 0.0
        else:
            err_zoom = fields[code][np.ix_(ti, xi)] - gt[np.ix_(ti, xi)]
            zoom_mae[code] = float(np.abs(err_zoom).mean())

    # --- Layout: manual GridSpec, no constrained / tight layout ---
    fig = plt.figure(figsize=(14, 13))
    outer = GridSpec(
        3, 1, figure=fig,
        height_ratios=[4.5, 2.2, 0.18],
        hspace=0.32,
    )
    top_gs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0], hspace=0.25, wspace=0.15,
    )
    bot_gs = GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], wspace=0.22,
    )
    cbar_gs = GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[2],
    )

    # --- Top 2x2: full-field panels ---
    ax_positions = [top_gs[0, 0], top_gs[0, 1], top_gs[1, 0], top_gs[1, 1]]
    main_axes = []
    im = None
    for idx, (pos, (code, label), plbl) in enumerate(
            zip(ax_positions, PANELS, PANEL_LABELS)):
        ax = fig.add_subplot(pos)
        main_axes.append(ax)

        field = fields[code]
        if code == 'GT':
            title = f'{plbl} {label}'
        else:
            title = f'{plbl} {label}, $L_2$ = {l2_vals[code]*100:.2f}%'

        im = ax.imshow(
            field.T, cmap='RdYlGn', vmin=vmin, vmax=vmax,
            aspect='auto', origin='lower', extent=extent_full,
            interpolation='nearest', rasterized=True)

        if code != 'GT':
            ax.contour(
                t_hours, x_miles, field.T,
                levels=[45], colors='black', linewidths=1.5,
                linestyles='solid', alpha=0.7)

        for sm in sensor_miles:
            ax.axhline(sm, color='white', linestyle='-',
                       linewidth=0.8, alpha=0.7)

        rect_w = ZOOM_T[1] - ZOOM_T[0]
        rect_h = ZOOM_X[1] - ZOOM_X[0]
        rect = Rectangle((ZOOM_T[0], ZOOM_X[0]), rect_w, rect_h,
                         linewidth=2.0, edgecolor='black',
                         facecolor='none', linestyle='-', zorder=8)
        ax.add_patch(rect)

        if code == 'GT':
            ax.annotate(
                'Zoom region',
                xy=(ZOOM_T[1], ZOOM_X[1]),
                xytext=(ZOOM_T[1] + 0.15, ZOOM_X[1] + 0.3),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='black', alpha=0.8),
                zorder=9)

        ax.set_title(title, fontsize=12)
        ax.set_xlim(t_hours[0], t_hours[-1])
        ax.set_ylim(x_miles[0], x_miles[-1])

    # Only bottom row of the 2x2 shows x-labels; only left column shows y.
    main_axes[0].tick_params(labelbottom=False)            # (a)
    main_axes[1].tick_params(labelbottom=False, labelleft=False)  # (b)
    main_axes[3].tick_params(labelleft=False)              # (d)
    main_axes[2].set_xlabel('Time (hours)')                # (c)
    main_axes[3].set_xlabel('Time (hours)')                # (d)
    main_axes[0].set_ylabel('Space (miles)')               # (a)
    main_axes[2].set_ylabel('Space (miles)')               # (c)

    # --- Bottom 1x4: zoom strip ---
    for i, ((code, label), plbl) in enumerate(zip(PANELS, PANEL_LABELS)):
        ax_z = fig.add_subplot(bot_gs[0, i])
        field = fields[code]
        field_zoom = field[np.ix_(ti, xi)]
        t_zoom = t_hours[ti]
        x_zoom = x_miles[xi]

        ax_z.imshow(
            field_zoom.T, cmap='RdYlGn', vmin=vmin, vmax=vmax,
            aspect='auto', origin='lower', extent=extent_zoom,
            interpolation='nearest', rasterized=True)

        if code != 'GT':
            ax_z.contour(
                t_zoom, x_zoom, field_zoom.T,
                levels=[45], colors='black', linewidths=1.0,
                linestyles='solid', alpha=0.6)

        if code == 'GT':
            ax_z.set_title(f'{plbl} {label}', fontsize=12, pad=6)
        else:
            ax_z.set_title(
                f'{plbl} {label}\nMAE = {zoom_mae[code]:.2f} mph',
                fontsize=12, pad=6)

        ax_z.set_xlim(ZOOM_T[0], ZOOM_T[1])
        ax_z.set_ylim(ZOOM_X[0], ZOOM_X[1])
        ax_z.set_xlabel('Time (h)', fontsize=10, labelpad=4)
        ax_z.tick_params(labelsize=10)
        if i == 0:
            ax_z.set_ylabel('Space (mi)', fontsize=10)
        else:
            ax_z.tick_params(labelleft=False)

    # --- Colorbar row ---
    cax = fig.add_subplot(cbar_gs[0, 0])
    fig.colorbar(im, cax=cax, orientation='horizontal', label='Speed (mph)')

    base = f'speed_fields_{n_s}s'
    fig.savefig(os.path.join(OUT_DIR, f'{base}.pdf'))
    fig.savefig(os.path.join(OUT_DIR, f'{base}.png'))
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


if __name__ == '__main__':
    main()
