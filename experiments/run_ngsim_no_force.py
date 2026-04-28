"""NGSIM B7 with force_decomp=False: test shock indicator behavior.

1 method x 5 sensors x 10 seeds = 50 runs.
Resume-safe via CSV tracking.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import time
import gc
from scipy import stats

from src.model import AdaStdpinnLWR, device
from src.utils import (
    load_dataset, get_sensor_indices, get_sensor_data, get_domain_bounds,
    compute_metrics, predict_full_field, count_params, get_split_info,
)

# ======================================================================
# Configuration
# ======================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

CSV_PATH = os.path.join(RESULTS_DIR, 'tables', 'per_seed_ngsim_no_force.csv')
CSV_FORCED = os.path.join(RESULTS_DIR, 'tables', 'per_seed_ngsim.csv')

SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]
SENSOR_COUNTS = [3, 4, 5, 6, 7]

PARAMS = {
    'N_f': 10000,
    'batch_size': 512,
    'layers': [2, 128, 64, 64, 1],
    'layers_after_split': [2, 128, 64, 1],
    'epochs': 20000,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_completed(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    completed = set()
    for _, row in df.iterrows():
        completed.add((str(row['Dataset']), str(row['Method']),
                        int(row['Sensors']), int(row['Seed'])))
    return completed


def append_row(csv_path, row_dict):
    df = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ======================================================================
# Main
# ======================================================================

def run_all():
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

    csv_file = os.path.join(DATA_DIR, 'ngsim', 'ngsim_data.csv')
    dataset = load_dataset(csv_file, 1600.0, 900.0)
    n_x = dataset['n_x']
    norm_params = dataset['norm_params']

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"NGSIM: v_f={dataset['v_f_mph']:.2f} mph, "
          f"grid={n_x}x{dataset['n_t']}")
    print(f"Test: B7 with force_decomp=False")
    print(f"Output: {CSV_PATH}\n")

    completed = load_completed(CSV_PATH)
    total = len(SENSOR_COUNTS) * len(SEEDS)
    done = len(completed)
    print(f"Total: {total} runs, {done} completed, {total - done} remaining\n")

    for n_sensors in SENSOR_COUNTS:
        sensor_idx = get_sensor_indices(n_x, n_sensors)

        for seed in SEEDS:
            key = ('ngsim', 'B7_no_force', n_sensors, seed)
            if key in completed:
                continue

            done += 1
            print(f"[{done}/{total}] Sensors={n_sensors}, Seed={seed}",
                  end=' ... ', flush=True)

            data_points, sensor_x_norm = get_sensor_data(
                dataset, sensor_idx, device)
            domain_bounds = get_domain_bounds(dataset)

            set_seed(seed)
            start_time = time.time()

            try:
                model = AdaStdpinnLWR(
                    domain_bounds, data_points,
                    layers=PARAMS['layers'],
                    layers_after_split=PARAMS['layers_after_split'],
                    w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
                    use_rh_interface=True,
                    use_entropy=True,
                    use_adaptive_loss=False,
                    use_rar=True,
                    use_causal_weighting=True,
                    use_spatial_decomp=True,
                    use_temporal_decomp=False,
                    shock_indicator_threshold=2.0,
                    **norm_params)
                model.train_two_stage(
                    total_epochs=PARAMS['epochs'],
                    stage1_epochs=5000,
                    batch_size=PARAMS['batch_size'],
                    N_f=PARAMS['N_f'],
                    force_decomp=False,
                    adaptive_sampling_freq=2500,
                    num_new_points=2500)
                train_time = time.time() - start_time

                pred_mph = predict_full_field(
                    model, dataset['X'], dataset['T'],
                    dataset['u_min'], dataset['u_max'], device)
                n_params = count_params(model)
                n_subs, splits = get_split_info(model)
                metrics = compute_metrics(dataset['Exact_mph'], pred_mph)

                row = {
                    'Dataset': 'ngsim',
                    'Method': 'B7_no_force',
                    'Seed': seed,
                    'Sensors': n_sensors,
                    'L2': metrics['L2'],
                    'RMSE': metrics['RMSE'],
                    'MSE': metrics['MSE'],
                    'Time': train_time,
                    'Subdomains': n_subs,
                    'Splits': ';'.join(splits),
                }
                append_row(CSV_PATH, row)
                completed.add(key)

                print(f"L2={metrics['L2']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                      f"Time={train_time:.1f}s, Subs={n_subs}")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                row = {
                    'Dataset': 'ngsim',
                    'Method': 'B7_no_force',
                    'Seed': seed,
                    'Sensors': n_sensors,
                    'L2': np.nan, 'RMSE': np.nan, 'MSE': np.nan,
                    'Time': np.nan, 'Subdomains': 0, 'Splits': '',
                }
                append_row(CSV_PATH, row)
                completed.add(key)

            cleanup_gpu()

    print(f"\n{'='*70}")
    print("ALL RUNS COMPLETE")
    print(f"{'='*70}\n")


# ======================================================================
# Analysis
# ======================================================================

def print_analysis():
    if not os.path.exists(CSV_PATH):
        print("No results found.")
        return

    df = pd.read_csv(CSV_PATH).dropna(subset=['L2'])

    # 1. Decomposition statistics
    print(f"{'='*70}")
    print("DECOMPOSITION BEHAVIOR (force_decomp=False)")
    print(f"{'='*70}")
    n_total = len(df)
    n_decomp = (df['Subdomains'] > 1).sum()
    n_single = (df['Subdomains'] <= 1).sum()
    print(f"Total runs: {n_total}")
    print(f"  Decomposed (>1 subdomain): {n_decomp} ({100*n_decomp/n_total:.0f}%)")
    print(f"  Single-domain:             {n_single} ({100*n_single/n_total:.0f}%)")

    print(f"\nPer sensor count:")
    for n_s in SENSOR_COUNTS:
        sub = df[df['Sensors'] == n_s]
        dec = (sub['Subdomains'] > 1).sum()
        print(f"  {n_s} sensors: {dec}/{len(sub)} decomposed, "
              f"mean subs={sub['Subdomains'].mean():.1f}")

    # 2. Mean L2 per sensor
    print(f"\n{'='*70}")
    print("L2 ERROR (force_decomp=False): mean +/- std")
    print(f"{'='*70}")
    print(f"{'Sensors':<10} {'L2 mean +/- std':<22} {'RMSE mean +/- std':<22} {'Time (s)':<10}")
    print("-" * 64)
    for n_s in SENSOR_COUNTS:
        sub = df[df['Sensors'] == n_s]
        print(f"{n_s:<10} {sub['L2'].mean():.4f} +/- {sub['L2'].std():.4f}   "
              f"{sub['RMSE'].mean():.4f} +/- {sub['RMSE'].std():.4f}   "
              f"{sub['Time'].mean():>8.1f}")

    # 3. Comparison with force_decomp=True
    if not os.path.exists(CSV_FORCED):
        print("\nNo force_decomp=True results found for comparison.")
        return

    df_forced = pd.read_csv(CSV_FORCED).dropna(subset=['L2'])
    df_forced = df_forced[df_forced['Method'] == 'B7']

    print(f"\n{'='*70}")
    print("COMPARISON: force_decomp=False vs force_decomp=True (B7)")
    print(f"{'='*70}")
    print(f"{'Sensors':<10} {'No-Force L2':<14} {'Forced L2':<14} "
          f"{'Diff':<10} {'p-value':<12} {'Winner':<12}")
    print("-" * 72)

    no_force_wins = 0
    forced_wins = 0
    ties = 0

    for n_s in SENSOR_COUNTS:
        nf = df[df['Sensors'] == n_s]['L2'].values
        fc = df_forced[df_forced['Sensors'] == n_s]['L2'].values

        if len(nf) < 2 or len(fc) < 2:
            continue

        nf_mean = nf.mean()
        fc_mean = fc.mean()
        diff = nf_mean - fc_mean
        t_stat, p_val = stats.ttest_ind(nf, fc, equal_var=False)

        if p_val < 0.05:
            if nf_mean < fc_mean:
                winner = "No-Force"
                no_force_wins += 1
            else:
                winner = "Forced"
                forced_wins += 1
        else:
            winner = "tie"
            ties += 1

        print(f"{n_s:<10} {nf_mean:<14.4f} {fc_mean:<14.4f} "
              f"{diff:<+10.4f} {p_val:<12.4e} {winner:<12}")

    print(f"\nScore: No-Force wins={no_force_wins}, "
          f"Forced wins={forced_wins}, ties={ties}")

    # Overall
    nf_all = df['L2'].mean()
    fc_all = df_forced['L2'].mean()
    print(f"\nOverall mean L2: No-Force={nf_all:.4f}, Forced={fc_all:.4f}")

    # Time comparison
    nf_time = df['Time'].mean()
    fc_time = df_forced['Time'].mean()
    print(f"Overall mean Time: No-Force={nf_time:.1f}s, Forced={fc_time:.1f}s "
          f"({100*(fc_time-nf_time)/fc_time:+.1f}% savings)")


if __name__ == '__main__':
    run_all()
    print_analysis()
