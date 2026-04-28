"""Standalone single-panel speed-field figures for 20221121, n_s=7.

Generates two clean single-panel heatmaps (Ground Truth and ADD-PINN)
matching the look of one panel in speed_fields_3s.png. No zoom region,
no contour overlay, no sensor lines, no L2/MAE labels, no panel letter.

Usage:
    python results/figures/plot_single_panel_7s.py
"""

import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'output_final')

T_RANGE_HOURS = 4.0
X_RANGE_MILES = 4.0
VMIN, VMAX = 0, 65
FT_PER_S_TO_MPH = 0.681818


def resolve_pred_path():
    primary = os.path.join(DATA_DIR, 'pred_B7_7s.npy')
    if os.path.isfile(primary):
        return primary
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, '*B7*7s*.npy')))
    candidates = [c for c in candidates if 'error' not in os.path.basename(c).lower()]
    if not candidates:
        raise FileNotFoundError(
            f"No B7 prediction at n_s=7 found under {DATA_DIR}")
    return candidates[0]


def load_field(path):
    arr = np.load(path)
    # The repo convention stores speeds in mph already; if the magnitude
    # looks like ft/s (max < ~100 but mean << 30), convert.
    if np.nanmax(np.abs(arr)) > 200:
        arr = arr * FT_PER_S_TO_MPH
    return arr


def plot_single(field, title, out_basename):
    n_t, n_x = field.shape
    extent = [0.0, T_RANGE_HOURS, 0.0, X_RANGE_MILES]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        field.T, cmap='RdYlGn', vmin=VMIN, vmax=VMAX,
        aspect='auto', origin='lower', extent=extent,
        interpolation='nearest', rasterized=True)
    ax.set_title(title)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Space (miles)')
    ax.set_xlim(0.0, T_RANGE_HOURS)
    ax.set_ylim(0.0, X_RANGE_MILES)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('Speed (mph)')

    pdf_path = os.path.join(OUT_DIR, f'{out_basename}.pdf')
    png_path = os.path.join(OUT_DIR, f'{out_basename}.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gt_path = os.path.join(DATA_DIR, 'gt_20221121.npy')
    pred_path = resolve_pred_path()

    print(f"Resolved GT path:       {gt_path}")
    print(f"Resolved ADD-PINN path: {pred_path}")

    gt = load_field(gt_path)
    pred = load_field(pred_path)

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    print(f"GT      shape={gt.shape}    "
          f"min={gt.min():.3f}  max={gt.max():.3f}  mean={gt.mean():.3f} (mph)")
    print(f"ADD-PINN shape={pred.shape}  "
          f"min={pred.min():.3f}  max={pred.max():.3f}  mean={pred.mean():.3f} (mph)")

    l2_rel = float(np.linalg.norm(gt - pred) / np.linalg.norm(gt))
    print(f"L2 relative error (ADD-PINN vs GT): {l2_rel:.6f}  ({l2_rel*100:.3f}%)")

    outputs = []
    outputs += list(plot_single(
        gt, 'Ground Truth', 'speed_field_20221121_7s_gt'))
    outputs += list(plot_single(
        pred, 'ADD-PINN', 'speed_field_20221121_7s_addpinn'))

    print("\nGenerated files:")
    for p in outputs:
        size_kb = os.path.getsize(p) / 1024.0
        print(f"  {p}  ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
