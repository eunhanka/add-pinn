"""Motivation figure for the ADD-PINN paper Introduction.

Illustrates why sparse fixed-sensor traffic state estimation is hard:
the I-24 MOTION ground truth shows sharp spatial shocks and high-frequency
temporal variation, while only a handful of fixed sensors are available.

Standalone script: reads the raw CSV directly, does not touch the npy caches.

Usage:
    python results/figures/plot_motivation.py --dataset 20221121
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke


FT_PER_S_TO_MPH = 3600.0 / 5280.0

# Project constants from experiments/run_all.py DATASETS entry for I-24:
#   (csv_path, x_range_ft, t_range_s) = (..., 21120.0, 14400.0)
# 21120 ft / 5280 = 4.0 miles; 14400 s / 3600 = 4.0 hours.
I24_X_RANGE_FT = 21120.0
I24_T_RANGE_S = 14400.0


def configure_rc():
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'serif',
        'text.usetex': False,
        'figure.dpi': 300,
    })


def resolve_physical_ranges(project_root: Path):
    """Read corridor length and total time from experiments/run_all.py if
    available, otherwise fall back to the documented defaults.
    """
    run_all_path = project_root / 'experiments' / 'run_all.py'
    x_range_ft = I24_X_RANGE_FT
    t_range_s = I24_T_RANGE_S
    source = 'default'
    try:
        text = run_all_path.read_text(encoding='utf-8')
        # Grab the first I-24 tuple; format is like:
        #   '20221121': ('i24/20221121.csv', 21120.0, 14400.0),
        for line in text.splitlines():
            line_stripped = line.strip()
            if "'i24/" in line_stripped and line_stripped.startswith("'2022"):
                # Parse the tuple values.
                after_csv = line_stripped.split("',", 1)[1]
                parts = [p.strip().rstrip(',').rstrip(')')
                         for p in after_csv.split(',')]
                x_range_ft = float(parts[0])
                t_range_s = float(parts[1])
                source = str(run_all_path)
                break
    except OSError:
        pass
    length_miles = x_range_ft / 5280.0
    total_time_h = t_range_s / 3600.0
    return length_miles, total_time_h, source


def load_ground_truth(csv_path: Path):
    """Load CSV and pivot to a (n_t, n_x) speed grid in mph."""
    df = pd.read_csv(csv_path, header=0)
    df = df[['t', 'x', 'speed']]
    df['speed'] = df['speed'] * FT_PER_S_TO_MPH
    grid = df.pivot_table(index='t', columns='x', values='speed').values
    grid = np.asarray(grid, dtype=np.float64)
    # Fill any stray NaNs with column (spatial) means so cross-sections are
    # finite without altering the shape.
    if np.isnan(grid).any():
        col_means = np.nanmean(grid, axis=0)
        idx = np.where(np.isnan(grid))
        grid[idx] = col_means[idx[1]]
    return grid


def select_cross_sections(v, n_x, n_t):
    """Pick (x_0, t_0) to maximize var_x[x_0] * grad_t[t_0] subject to
    excluding the first/last 10% margins and keeping only the top 20%
    of each criterion.
    """
    var_x = v.var(axis=0)
    grad_t = np.sum(np.abs(np.diff(v, axis=1)), axis=1)

    x_lo, x_hi = int(0.10 * n_x), int(0.90 * n_x)
    t_lo, t_hi = int(0.10 * n_t), int(0.90 * n_t)

    x_interior = np.arange(x_lo, x_hi)
    t_interior = np.arange(t_lo, t_hi)

    var_interior = var_x[x_interior]
    grad_interior = grad_t[t_interior]

    var_thresh = np.quantile(var_interior, 0.80)
    grad_thresh = np.quantile(grad_interior, 0.80)

    x_candidates = x_interior[var_interior >= var_thresh]
    t_candidates = t_interior[grad_interior >= grad_thresh]

    # Product is separable, so the maximizer is the argmax of each factor
    # within the candidate sets.
    x_0 = int(x_candidates[np.argmax(var_x[x_candidates])])
    t_0 = int(t_candidates[np.argmax(grad_t[t_candidates])])
    return x_0, t_0, var_x, grad_t


def build_figure(v, length_miles, total_time_h, x_0, t_0):
    n_t, n_x = v.shape
    t_h = np.linspace(0.0, total_time_h, n_t)
    x_mi = np.linspace(0.0, length_miles, n_x)
    x_0_mi = x_mi[x_0]
    t_0_h = t_h[t_0]

    fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08,
                                    wspace=0.10, hspace=0.12)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    # LEFT: heatmap. We want time on x-axis and space on y-axis, so
    # transpose to (n_x, n_t).
    im = ax_heat.imshow(
        v.T,
        extent=[0.0, total_time_h, 0.0, length_miles],
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='RdYlGn',
        vmin=0,
        vmax=75,
    )
    ax_heat.set_xlabel('Time (h)')
    ax_heat.set_ylabel('Space (mi)')
    ax_heat.set_title(
        'I-24 MOTION ground truth speed field',
        fontsize=15, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax_heat, pad=0.02)
    cbar.set_label('Speed v(x,t) [mph]')

    # Blue dashed horizontal strip around x_0 (full time, thin in space).
    band_space = 0.02 * length_miles
    blue_y0 = max(0.0, x_0_mi - band_space / 2.0)
    blue_h = band_space
    blue_rect = Rectangle(
        (0.0, blue_y0), total_time_h, blue_h,
        edgecolor='blue', facecolor='none',
        linewidth=2.5, linestyle='--',
    )
    blue_rect.set_path_effects(
        [withStroke(linewidth=4.0, foreground='white')])
    ax_heat.add_patch(blue_rect)

    # Green dashed vertical strip around t_0 (full space, thin in time).
    band_time = 0.02 * total_time_h
    green_x0 = max(0.0, t_0_h - band_time / 2.0)
    green_w = band_time
    green_rect = Rectangle(
        (green_x0, 0.0), green_w, length_miles,
        edgecolor='green', facecolor='none',
        linewidth=2.5, linestyle='--',
    )
    green_rect.set_path_effects(
        [withStroke(linewidth=4.0, foreground='white')])
    ax_heat.add_patch(green_rect)

    # RIGHT TOP: v(x_0, t) time series.
    ax_top.plot(t_h, v[:, x_0], color='#1f4e79', linewidth=1.2,
                label=f'x = {x_0_mi:.2f} mi')
    ax_top.set_xlim(0.0, total_time_h)
    ax_top.set_xlabel('Time (h)')
    ax_top.set_ylabel('v(x,t) [mph]')
    ax_top.set_title(
        f'Ground truth at x = {x_0_mi:.2f} mi\n'
        '(high-frequency temporal variation)'
    )

    # RIGHT BOTTOM: v(x, t_0) spatial profile.
    ax_bot.plot(x_mi, v[t_0, :], color='#2d5016', linewidth=1.2,
                label=f't = {t_0_h:.2f} h')
    ax_bot.set_xlim(0.0, length_miles)
    ax_bot.set_xlabel('Location (mi)')
    ax_bot.set_ylabel('v(x,t) [mph]')
    ax_bot.set_title(
        f'Ground truth at t = {t_0_h:.2f} h\n'
        '(sharp spatial gradient, shock)'
    )

    return fig


def main():
    configure_rc()

    parser = argparse.ArgumentParser(
        description='Generate ADD-PINN Introduction motivation figure.')
    parser.add_argument('--dataset', default='20221121',
                        help='I-24 dataset identifier (default: 20221121).')
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    csv_path = project_root / 'data' / 'i24' / f'{args.dataset}.csv'
    out_dir = project_root / 'results' / 'figures' / 'output_final'
    out_dir.mkdir(parents=True, exist_ok=True)

    length_miles, total_time_h, source = resolve_physical_ranges(project_root)
    print(f'Physical ranges resolved from: {source}')
    print(f'  Corridor length: {length_miles:.4f} mi')
    print(f'  Total time:      {total_time_h:.4f} h')

    if not csv_path.exists():
        print(f'ERROR: dataset CSV not found at {csv_path}', file=sys.stderr)
        sys.exit(1)

    v = load_ground_truth(csv_path)
    n_t, n_x = v.shape
    print(f'Loaded grid shape: {v.shape} (rows=time, cols=space)')

    x_0, t_0, var_x, grad_t = select_cross_sections(v, n_x=n_x, n_t=n_t)
    x_0_mi = (x_0 / max(n_x - 1, 1)) * length_miles
    t_0_h = (t_0 / max(n_t - 1, 1)) * total_time_h

    print(f'Selected cross sections:')
    print(f'  x_0 grid index = {x_0}  (x_0 = {x_0_mi:.4f} mi)')
    print(f'  t_0 grid index = {t_0}  (t_0 = {t_0_h:.4f} h)')
    print(f'  var_x[x_0]  = {var_x[x_0]:.4f}')
    print(f'  grad_t[t_0] = {grad_t[t_0]:.4f}')

    fig = build_figure(v, length_miles, total_time_h, x_0, t_0)

    pdf_path = out_dir / 'motivation_figure.pdf'
    png_path = out_dir / 'motivation_figure.png'
    json_path = out_dir / 'motivation_figure_coords.json'

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    speed_min = float(np.min(v))
    speed_max = float(np.max(v))
    speed_mean = float(np.mean(v))
    speed_p95 = float(np.percentile(v, 95))
    congested_fraction = float(np.mean(v < 30.0))

    payload = {
        'dataset': args.dataset,
        'x_0_grid_idx': int(x_0),
        'x_0_mi': float(x_0_mi),
        't_0_grid_idx': int(t_0),
        't_0_h': float(t_0_h),
        'corridor_length_mi': float(length_miles),
        'total_time_h': float(total_time_h),
        'speed_mph': {
            'min': speed_min,
            'max': speed_max,
            'mean': speed_mean,
            'p95': speed_p95,
        },
        'congested_fraction': congested_fraction,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print()
    print('Verification summary')
    print(f'  Corridor length: {length_miles:.4f} mi, total time: {total_time_h:.4f} h')
    print(f'  (x_0, t_0) = ({x_0_mi:.4f} mi, {t_0_h:.4f} h)')
    print(f'  var_x[x_0]  = {var_x[x_0]:.6f}')
    print(f'  grad_t[t_0] = {grad_t[t_0]:.6f}')
    print(f'  speed mph: min={speed_min:.3f}, max={speed_max:.3f}, '
          f'mean={speed_mean:.3f}, p95={speed_p95:.3f}')
    print(f'  congested fraction (v < 30 mph): {100.0 * congested_fraction:.2f}%')

    print()
    print('Outputs:')
    for p in (pdf_path, png_path, json_path):
        p_abs = p.resolve()
        size = p_abs.stat().st_size if p_abs.exists() else -1
        print(f'  {p_abs}  ({size} bytes)')


if __name__ == '__main__':
    main()
