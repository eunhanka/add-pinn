"""Generate LaTeX tables from experiment results."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')

METHOD_NAMES = {
    'B1': 'Linear Interp.',
    'B2': 'Simple NN',
    'B3': 'Vanilla PINN',
    'B4': 'PINN + RAR',
    'B5': 'PINN + Viscosity',
    'B6': 'cPINN-DD',
    'B7': r'\textbf{CA-STD-PINN (Ours)}',
}

ABLATION_NAMES = {
    'A1': r'\textbf{Full Model}',
    'A2': r'$-$ Two-Stage (Online DD)',
    'A3': r'$-$ R-H (Flux Cont.)',
    'A4': r'$+$ Adaptive Weights',
}


def bold_best(values, fmt='{:.2f}', lower_better=True):
    """Return formatted strings with best value bolded."""
    arr = [v for v in values if not pd.isna(v)]
    if not arr:
        return [fmt.format(v) if not pd.isna(v) else '--' for v in values]
    best = min(arr) if lower_better else max(arr)
    result = []
    for v in values:
        if pd.isna(v):
            result.append('--')
        elif abs(v - best) < 1e-10:
            result.append(r'\textbf{' + fmt.format(v) + '}')
        else:
            result.append(fmt.format(v))
    return result


def sig_marker(p_val):
    """Significance marker from p-value."""
    if pd.isna(p_val):
        return ''
    if p_val < 0.001:
        return '$^{***}$'
    if p_val < 0.01:
        return '$^{**}$'
    if p_val < 0.05:
        return '$^{*}$'
    return ''


def table_main_comparison():
    """Table 1: Main comparison (L2 mean+-std, per dataset x sensor count)."""
    summary_path = os.path.join(TABLE_DIR, 'summary_all.csv')
    ttest_path = os.path.join(TABLE_DIR, 'ttest_vs_B7.csv')

    if not os.path.exists(summary_path):
        print("No summary data")
        return

    df = pd.read_csv(summary_path)
    ttest_df = None
    if os.path.exists(ttest_path):
        ttest_df = pd.read_csv(ttest_path)

    methods = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    datasets = ['20221121', '20221122', '20221123',
                '20221129', '20221202', 'ngsim']

    for n_sensors in [3, 5, 7]:
        print(f"\n% === Table: {n_sensors} sensors ===")
        print(r'\begin{table}[htbp]')
        print(r'\centering')
        print(r'\caption{L2 Error (\%) with ' + str(n_sensors) + ' sensors}')
        print(r'\begin{tabular}{l' + 'c' * len(datasets) + '}')
        print(r'\toprule')

        header = 'Method & ' + ' & '.join(
            [ds[:8] for ds in datasets]) + r' \\'
        print(header)
        print(r'\midrule')

        for method in methods:
            vals = []
            sigs = []
            for ds in datasets:
                row = df[(df['Dataset'] == ds) &
                         (df['Method'] == method) &
                         (df['Sensors'] == n_sensors)]
                if len(row) > 0:
                    mean = row.iloc[0]['L2_mean'] * 100
                    std = row.iloc[0]['L2_std'] * 100
                    vals.append(mean)
                    # Get significance
                    sig = ''
                    if ttest_df is not None and method != 'B7':
                        t_row = ttest_df[
                            (ttest_df['Dataset'] == ds) &
                            (ttest_df['Sensors'] == n_sensors) &
                            (ttest_df['Method'] == method)]
                        if len(t_row) > 0:
                            sig = sig_marker(t_row.iloc[0]['p_value'])
                    if std > 0:
                        sigs.append(
                            f'{mean:.2f}$\\pm${std:.2f}{sig}')
                    else:
                        sigs.append(f'{mean:.2f}{sig}')
                else:
                    vals.append(np.nan)
                    sigs.append('--')

            name = METHOD_NAMES.get(method, method)
            line = name + ' & ' + ' & '.join(sigs) + r' \\'
            print(line)

        print(r'\bottomrule')
        print(r'\end{tabular}')
        print(r'\end{table}')


def table_ablation():
    """Table 2: Ablation study."""
    summary_path = os.path.join(TABLE_DIR, 'ablation_summary.csv')
    ttest_path = os.path.join(TABLE_DIR, 'ttest_ablation_vs_A1.csv')

    if not os.path.exists(summary_path):
        print("No ablation data")
        return

    df = pd.read_csv(summary_path)
    ttest_df = None
    if os.path.exists(ttest_path):
        ttest_df = pd.read_csv(ttest_path)

    print("\n% === Table: Ablation ===")
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Ablation Study (I-24 Nov 21, 5 Sensors)}')
    print(r'\begin{tabular}{lccc}')
    print(r'\toprule')
    print(r'Variant & L2 (\%) & RMSE (mph) & Time (s) \\')
    print(r'\midrule')

    for variant in ['A1', 'A2', 'A3', 'A4']:
        row = df[df['Method'] == variant]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        l2 = f"{r['L2_mean']*100:.2f}$\\pm${r['L2_std']*100:.2f}"
        rmse = f"{r['RMSE_mean']:.2f}$\\pm${r['RMSE_std']:.2f}"
        t_val = f"{r['Time_mean']:.0f}"

        sig = ''
        if ttest_df is not None and variant != 'A1':
            t_row = ttest_df[ttest_df['Method'] == variant]
            if len(t_row) > 0:
                sig = sig_marker(t_row.iloc[0]['p_value'])

        name = ABLATION_NAMES.get(variant, variant)
        print(f'{name} & {l2}{sig} & {rmse} & {t_val} \\\\')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


def table_computation_cost():
    """Table 3: Computation cost."""
    cost_path = os.path.join(TABLE_DIR, 'computation_cost.csv')
    if not os.path.exists(cost_path):
        print("No cost data")
        return

    df = pd.read_csv(cost_path)

    print("\n% === Table: Computation Cost ===")
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Computation Cost (I-24 Nov 21, 5 Sensors, Seed=42)}')
    print(r'\begin{tabular}{lcccc}')
    print(r'\toprule')
    print(r'Method & L2 (\%) & Time (s) & Parameters & Subdomains \\')
    print(r'\midrule')

    for _, row in df.iterrows():
        method = row['Method']
        name = METHOD_NAMES.get(method, method)
        l2 = f"{row['L2']*100:.2f}"
        t_val = f"{row['Time']:.0f}" if row['Time'] > 0 else '--'
        params = f"{row['Parameters']:,}" if row['Parameters'] > 0 else '--'
        subs = str(int(row['Subdomains']))
        print(f'{name} & {l2} & {t_val} & {params} & {subs} \\\\')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


def table_shock_indicators():
    """Table 4: Shock indicators."""
    shock_path = os.path.join(TABLE_DIR, 'shock_indicators.csv')
    if not os.path.exists(shock_path):
        print("No shock indicator data")
        return

    df = pd.read_csv(shock_path)

    print("\n% === Table: Shock Indicators ===")
    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Data-Driven Shock Indicators}')
    print(r'\begin{tabular}{lcccr}')
    print(r'\toprule')
    print(r'Dataset & Spatial & Temporal & $v_f$ (mph) & Cong. (\%) \\')
    print(r'\midrule')

    for _, row in df.iterrows():
        ds = row['Dataset']
        print(f"{ds} & {row['Spatial_Indicator']:.2f} & "
              f"{row['Temporal_Indicator']:.2f} & "
              f"{row['v_f_mph']:.1f} & "
              f"{row['Congestion_Pct']:.1f} \\\\")

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    print("=" * 60)
    print("LaTeX Tables")
    print("=" * 60)

    table_main_comparison()
    table_ablation()
    table_computation_cost()
    table_shock_indicators()
