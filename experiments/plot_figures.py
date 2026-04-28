"""Generate all paper figures from saved results. No GPU needed."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ======================================================================
# Style defaults
# ======================================================================
FONT_SIZE = 12
TITLE_SIZE = 14
CMAP_SPEED = 'RdYlGn'
CMAP_ERROR = 'hot'
DPI = 300
FORMAT = 'pdf'

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 2,
    'ytick.labelsize': FONT_SIZE - 2,
    'legend.fontsize': FONT_SIZE - 2,
    'figure.dpi': DPI,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
GT_DIR = os.path.join(RESULTS_DIR, 'ground_truth')
PRED_DIR = os.path.join(RESULTS_DIR, 'predictions')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')

os.makedirs(FIG_DIR, exist_ok=True)

DATASETS = ['20221121', '20221122', '20221123',
            '20221129', '20221202', 'ngsim']

DATASET_LABELS = {
    '20221121': 'I-24 Nov 21 (Accident)',
    '20221122': 'I-24 Nov 22 (Recurrent)',
    '20221123': 'I-24 Nov 23 (Mild)',
    '20221129': 'I-24 Nov 29 (Recurrent)',
    '20221202': 'I-24 Dec 02 (Accident)',
    'ngsim': 'NGSIM',
}

METHOD_NAMES = {
    'B1': 'Linear Interp.',
    'B2': 'Simple NN',
    'B3': 'Vanilla PINN',
    'B4': 'PINN+RAR',
    'B5': 'PINN+Viscosity',
    'B6': 'cPINN-DD',
    'B7': 'CA-STD-PINN',
}

METHOD_COLORS = {
    'B1': '#999999', 'B2': '#e377c2', 'B3': '#1f77b4',
    'B4': '#ff7f0e', 'B5': '#2ca02c', 'B6': '#d62728',
    'B7': '#9467bd',
}

METHOD_MARKERS = {
    'B1': 'v', 'B2': '<', 'B3': 'o', 'B4': 's',
    'B5': 'D', 'B6': '^', 'B7': '*',
}


def load_gt(ds_name):
    """Load ground truth and coordinates."""
    gt = np.load(os.path.join(GT_DIR, f'{ds_name}_ground_truth.npy'))
    X = np.load(os.path.join(GT_DIR, f'{ds_name}_x_coords.npy'))
    T = np.load(os.path.join(GT_DIR, f'{ds_name}_t_coords.npy'))
    return gt, X, T


def load_pred(ds_name, method, n_sensors):
    """Load prediction for seed=42."""
    path = os.path.join(
        PRED_DIR, f'{ds_name}_{method}_{n_sensors}s_seed42_pred.npy')
    if os.path.exists(path):
        return np.load(path)
    return None


def load_summary():
    """Load summary CSV."""
    path = os.path.join(TABLE_DIR, 'summary_all.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_shock_indicators():
    """Load shock indicators."""
    path = os.path.join(TABLE_DIR, 'shock_indicators.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def get_sensor_positions(ds_name, n_sensors):
    """Get normalized sensor x positions."""
    if ds_name == 'ngsim':
        n_x = 81
    else:
        n_x = 100
    indices = np.linspace(0, n_x - 1, n_sensors + 2)[1:-1].astype(int)
    x_norm = indices / (n_x - 1)
    return x_norm


# ======================================================================
# Figure 1: Ground truth heatmaps
# ======================================================================

def fig1_ground_truth():
    """All 6 datasets ground truth heatmaps (2x3 grid)."""
    shock_df = load_shock_indicators()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, ds_name in enumerate(DATASETS):
        try:
            gt, X, T = load_gt(ds_name)
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, f'{ds_name}\n(no data)',
                         ha='center', va='center',
                         transform=axes[i].transAxes)
            continue

        ax = axes[i]
        im = ax.pcolormesh(T, X, gt, cmap=CMAP_SPEED,
                           vmin=0, vmax=70, shading='auto')
        ax.set_xlabel('Time (normalized)')
        ax.set_ylabel('Space (normalized)')

        label = DATASET_LABELS.get(ds_name, ds_name)
        if shock_df is not None:
            row = shock_df[shock_df['Dataset'] == ds_name]
            if len(row) > 0:
                cong = row.iloc[0]['Congestion_Pct']
                label += f' ({cong:.0f}% cong.)'
        ax.set_title(label)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Speed (mph)')

    fig.suptitle('Ground Truth Speed Fields', fontsize=TITLE_SIZE + 2, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(os.path.join(FIG_DIR, f'fig1_ground_truth.{FORMAT}'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig1_ground_truth")


# ======================================================================
# Figure 2: Prediction comparison
# ======================================================================

def fig2_prediction_comparison():
    """3 representative datasets, 5 sensors, seed=42."""
    rep_datasets = ['20221121', '20221122', 'ngsim']
    methods_show = ['B1', 'B3', 'B6', 'B7']
    n_sensors = 5

    n_rows = len(rep_datasets)
    n_cols = 1 + len(methods_show)  # GT + methods

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))

    # Load per-seed data for L2 values
    per_seed_path = os.path.join(TABLE_DIR, 'per_seed_all.csv')
    per_seed_df = None
    if os.path.exists(per_seed_path):
        per_seed_df = pd.read_csv(per_seed_path)

    for row_i, ds_name in enumerate(rep_datasets):
        try:
            gt, X, T = load_gt(ds_name)
        except FileNotFoundError:
            continue

        sensor_x = get_sensor_positions(ds_name, n_sensors)

        # Ground truth
        ax = axes[row_i, 0]
        ax.pcolormesh(T, X, gt, cmap=CMAP_SPEED,
                      vmin=0, vmax=70, shading='auto')
        for sx in sensor_x:
            ax.axhline(y=sx, color='white', linestyle='--',
                       linewidth=0.8, alpha=0.7)
        ax.set_title(f'{DATASET_LABELS[ds_name]}\nGround Truth')
        ax.set_ylabel('Space')
        if row_i == n_rows - 1:
            ax.set_xlabel('Time')

        # Methods
        for col_i, method in enumerate(methods_show):
            ax = axes[row_i, col_i + 1]
            pred = load_pred(ds_name, method, n_sensors)
            if pred is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(METHOD_NAMES[method])
                continue

            ax.pcolormesh(T, X, pred, cmap=CMAP_SPEED,
                          vmin=0, vmax=70, shading='auto')
            for sx in sensor_x:
                ax.axhline(y=sx, color='white', linestyle='--',
                           linewidth=0.8, alpha=0.7)

            # Get L2 for title
            l2_str = ''
            if per_seed_df is not None:
                mask = ((per_seed_df['Dataset'] == ds_name) &
                        (per_seed_df['Method'] == method) &
                        (per_seed_df['Sensors'] == n_sensors) &
                        (per_seed_df['Seed'] == 42))
                match = per_seed_df[mask]
                if len(match) > 0:
                    l2_val = match.iloc[0]['L2'] * 100
                    l2_str = f' (L2={l2_val:.1f}%)'

            ax.set_title(f'{METHOD_NAMES[method]}{l2_str}')
            if row_i == n_rows - 1:
                ax.set_xlabel('Time')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig2_predictions.{FORMAT}'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig2_predictions")


# ======================================================================
# Figure 3: Error heatmaps
# ======================================================================

def fig3_error_heatmaps():
    """Error |pred - GT| for same configs as Fig 2."""
    rep_datasets = ['20221121', '20221122', 'ngsim']
    methods_show = ['B1', 'B3', 'B6', 'B7']
    n_sensors = 5

    n_rows = len(rep_datasets)
    n_cols = len(methods_show)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))

    for row_i, ds_name in enumerate(rep_datasets):
        try:
            gt, X, T = load_gt(ds_name)
        except FileNotFoundError:
            continue

        for col_i, method in enumerate(methods_show):
            ax = axes[row_i, col_i]
            pred = load_pred(ds_name, method, n_sensors)
            if pred is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            error = np.abs(pred - gt)
            im = ax.pcolormesh(T, X, error, cmap=CMAP_ERROR,
                               vmin=0, vmax=20, shading='auto')
            ax.set_title(f'{METHOD_NAMES[method]}')
            if col_i == 0:
                ax.set_ylabel(DATASET_LABELS[ds_name])
            if row_i == n_rows - 1:
                ax.set_xlabel('Time')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='|Error| (mph)')

    plt.tight_layout(rect=[0, 0, 0.92, 1.0])
    fig.savefig(os.path.join(FIG_DIR, f'fig3_errors.{FORMAT}'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig3_errors")


# ======================================================================
# Figure 4: Sensor sensitivity
# ======================================================================

def fig4_sensor_sensitivity():
    """L2 vs sensor count, one panel per dataset."""
    summary = load_summary()
    if summary is None:
        print("No summary data for fig4")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    methods = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    for i, ds_name in enumerate(DATASETS):
        ax = axes[i]
        ds_data = summary[summary['Dataset'] == ds_name]

        for method in methods:
            m_data = ds_data[ds_data['Method'] == method].sort_values('Sensors')
            if len(m_data) == 0:
                continue
            ax.errorbar(
                m_data['Sensors'], m_data['L2_mean'] * 100,
                yerr=m_data['L2_std'] * 100,
                label=METHOD_NAMES[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=6, capsize=3, linewidth=1.5)

        ax.set_title(DATASET_LABELS.get(ds_name, ds_name))
        ax.set_xlabel('Number of Sensors')
        ax.set_ylabel('L2 Error (%)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='best', fontsize=FONT_SIZE - 3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig4_sensor_sensitivity.{FORMAT}'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig4_sensor_sensitivity")


# ======================================================================
# Figure 5: Ablation bar chart
# ======================================================================

def fig5_ablation():
    """A1-A4 bar chart with error bars and significance stars."""
    ablation_path = os.path.join(TABLE_DIR, 'ablation_summary.csv')
    ttest_path = os.path.join(TABLE_DIR, 'ttest_ablation_vs_A1.csv')

    if not os.path.exists(ablation_path):
        print("No ablation data for fig5")
        return

    df = pd.read_csv(ablation_path)
    ttest_df = None
    if os.path.exists(ttest_path):
        ttest_df = pd.read_csv(ttest_path)

    variant_labels = {
        'A1': 'Full Model\n(CA-STD-PINN)',
        'A2': 'w/o Two-Stage\n(Online DD)',
        'A3': 'w/o R-H\n(Flux Cont.)',
        'A4': 'w/ Adaptive\nLoss Weights',
    }
    variants = ['A1', 'A2', 'A3', 'A4']
    colors = ['#9467bd', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(variants))
    bars = []

    for i, v in enumerate(variants):
        row = df[df['Method'] == v]
        if len(row) == 0:
            continue
        mean_val = row.iloc[0]['L2_mean'] * 100
        std_val = row.iloc[0]['L2_std'] * 100
        bar = ax.bar(x_pos[i], mean_val, yerr=std_val, color=colors[i],
                     capsize=5, edgecolor='black', linewidth=0.5)
        bars.append(bar)

        # Significance star
        if ttest_df is not None and v != 'A1':
            t_row = ttest_df[ttest_df['Method'] == v]
            if len(t_row) > 0:
                p_val = t_row.iloc[0]['p_value']
                if p_val < 0.001:
                    star = '***'
                elif p_val < 0.01:
                    star = '**'
                elif p_val < 0.05:
                    star = '*'
                else:
                    star = 'n.s.'
                ax.text(x_pos[i], mean_val + std_val + 0.3, star,
                        ha='center', va='bottom', fontsize=FONT_SIZE)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([variant_labels[v] for v in variants])
    ax.set_ylabel('L2 Error (%)')
    ax.set_title('Ablation Study (I-24 Nov 21, 5 Sensors)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig5_ablation.{FORMAT}'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig5_ablation")


# ======================================================================
# Figure 6: Pareto front
# ======================================================================

def fig6_pareto():
    """L2 vs Training Time, 5 sensors, 20221121."""
    cost_path = os.path.join(TABLE_DIR, 'computation_cost.csv')
    if not os.path.exists(cost_path):
        print("No cost data for fig6")
        return

    df = pd.read_csv(cost_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        method = row['Method']
        l2 = row['L2'] * 100
        t = row['Time']
        ax.scatter(t, l2, color=METHOD_COLORS.get(method, 'gray'),
                   marker=METHOD_MARKERS.get(method, 'o'),
                   s=150, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(METHOD_NAMES.get(method, method),
                    (t, l2), textcoords="offset points",
                    xytext=(8, 5), fontsize=FONT_SIZE - 2)

    # Identify Pareto-optimal points
    points = df[['Time', 'L2']].values
    pareto_mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                if (points[j][0] <= points[i][0] and
                        points[j][1] <= points[i][1] and
                        (points[j][0] < points[i][0] or
                         points[j][1] < points[i][1])):
                    pareto_mask[i] = False
                    break

    # Mark Pareto-optimal with star outline
    pareto_df = df[pareto_mask]
    for _, row in pareto_df.iterrows():
        method = row['Method']
        ax.scatter(row['Time'], row['L2'] * 100,
                   marker='*', s=300, facecolors='none',
                   edgecolors='gold', linewidths=2, zorder=6)

    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('L2 Error (%)')
    ax.set_title('Pareto Front: Accuracy vs. Computation Cost\n'
                 '(I-24 Nov 21, 5 Sensors, Seed=42)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig6_pareto.{FORMAT}'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig6_pareto")


# ======================================================================
# Figure 7: Multi-day robustness
# ======================================================================

def fig7_multiday():
    """Grouped bar chart: L2 for B3, B6, B7 across 5 I-24 days, 5 sensors."""
    summary = load_summary()
    if summary is None:
        print("No summary data for fig7")
        return

    methods = ['B3', 'B6', 'B7']
    i24_datasets = ['20221121', '20221122', '20221123',
                    '20221129', '20221202']
    n_sensors = 5

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(i24_datasets))
    width = 0.25

    for i, method in enumerate(methods):
        means = []
        stds = []
        for ds_name in i24_datasets:
            row = summary[(summary['Dataset'] == ds_name) &
                          (summary['Method'] == method) &
                          (summary['Sensors'] == n_sensors)]
            if len(row) > 0:
                means.append(row.iloc[0]['L2_mean'] * 100)
                stds.append(row.iloc[0]['L2_std'] * 100)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, yerr=stds,
               label=METHOD_NAMES[method],
               color=METHOD_COLORS[method],
               capsize=3, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width)
    short_labels = ['Nov 21\n(Acc.)', 'Nov 22\n(Rec.)', 'Nov 23\n(Mild)',
                    'Nov 29\n(Rec.)', 'Dec 02\n(Acc.)']
    ax.set_xticklabels(short_labels)
    ax.set_ylabel('L2 Error (%)')
    ax.set_title('Multi-Day Robustness (5 Sensors)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig7_multiday.{FORMAT}'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig7_multiday")


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    print("Generating paper figures...")
    print(f"Output directory: {FIG_DIR}")
    print(f"Format: {FORMAT}, DPI: {DPI}")
    print()

    fig1_ground_truth()
    fig2_prediction_comparison()
    fig3_error_heatmaps()
    fig4_sensor_sensitivity()
    fig5_ablation()
    fig6_pareto()
    fig7_multiday()

    print(f"\nAll figures saved to {FIG_DIR}")
