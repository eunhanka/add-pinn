"""NGSIM Quick Test: B2-B5, B7, B8 on NGSIM dataset.

6 methods x 5 sensor counts x 10 seeds = 300 runs.
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

from src.model import (
    SimpleNN, VanillaPinnLWR, RARPinnLWR, AdaStdpinnLWR, device
)
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

CSV_PATH = os.path.join(RESULTS_DIR, 'tables', 'per_seed_ngsim.csv')

SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]
SENSOR_COUNTS = [3, 4, 5, 6, 7]
METHODS = ['B2', 'B3', 'B4', 'B5', 'B7', 'B8']

NGSIM_CSV = 'ngsim/ngsim_data.csv'
X_RANGE = 1600.0
T_RANGE = 900.0

PARAMS = {
    'N_f': 10000,
    'batch_size': 512,
    'layers': [2, 128, 64, 64, 1],
    'layers_after_split': [2, 128, 64, 1],
    'epochs': 20000,
}

METHOD_NAMES = {
    'B2': 'Simple NN',
    'B3': 'Vanilla PINN',
    'B4': 'PINN + RAR',
    'B5': 'PINN + Viscosity',
    'B7': 'CA-STD-PINN (Ours)',
    'B8': 'XPINN',
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
        completed.add((
            str(row['Dataset']), str(row['Method']),
            int(row['Sensors']), int(row['Seed'])))
    return completed


def append_row(csv_path, row_dict):
    df = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ======================================================================
# Run one method
# ======================================================================

def run_method(method, dataset, data_points, sensor_x_norm, norm_params, seed):
    """Returns (pred_mph, train_time, n_params, n_subdomains, splits_str)."""
    domain_bounds = get_domain_bounds(dataset)
    layers = PARAMS['layers']
    layers_after = PARAMS['layers_after_split']
    epochs = PARAMS['epochs']
    N_f = PARAMS['N_f']
    batch_size = PARAMS['batch_size']

    set_seed(seed)
    start_time = time.time()

    if method == 'B2':
        model = SimpleNN(data_points, layers=layers)
        model.train(epochs=epochs, batch_size=batch_size)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        return pred_mph, train_time, n_params, 1, ''

    elif method == 'B3':
        model = VanillaPinnLWR(
            domain_bounds, data_points, layers=layers,
            w_data_init=0.85, w_pde_init=0.05, **norm_params)
        model.train(epochs=epochs, batch_size=batch_size, N_f=N_f)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        return pred_mph, train_time, n_params, 1, ''

    elif method == 'B4':
        model = RARPinnLWR(
            domain_bounds, data_points, layers=layers,
            w_data_init=0.85, w_pde_init=0.05, **norm_params)
        model.train(
            epochs=epochs, batch_size=batch_size, N_f=N_f,
            adaptive_sampling_freq=2500, num_new_points=2500)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        return pred_mph, train_time, n_params, 1, ''

    elif method == 'B5':
        model = AdaStdpinnLWR(
            domain_bounds, data_points, layers=layers,
            layers_after_split=layers_after,
            w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
            use_viscous_pde=True, viscous_epsilon=0.1,
            use_causal_weighting=False,
            use_adaptive_loss=False,
            use_rar=False,
            use_spatial_decomp=False,
            use_temporal_decomp=False,
            use_rh_interface=False,
            use_entropy=False,
            shock_indicator_threshold=999.0,
            **norm_params)
        model.w_data = torch.tensor(
            0.85, requires_grad=False, device=device, dtype=torch.float32)
        model.w_pde = torch.tensor(
            0.05, requires_grad=False, device=device, dtype=torch.float32)
        model.w_int = torch.tensor(
            0.10, requires_grad=False, device=device, dtype=torch.float32)
        net_params = list(model.subdomains[0]['net'].parameters())
        model.optimizer_model = torch.optim.Adam(net_params, lr=1e-3)
        model.scheduler = torch.optim.lr_scheduler.StepLR(
            model.optimizer_model, step_size=5000, gamma=0.9)
        model.train(
            epochs=epochs, batch_size=batch_size, N_f=N_f,
            adaptive_sampling_freq=999999,
            domain_decomp_freq=999999,
            num_new_points=0,
            residual_threshold=999.0)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        return pred_mph, train_time, n_params, 1, ''

    elif method == 'B7':
        model = AdaStdpinnLWR(
            domain_bounds, data_points, layers=layers,
            layers_after_split=layers_after,
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
            total_epochs=epochs, stage1_epochs=5000,
            batch_size=batch_size, N_f=N_f,
            force_decomp=True,
            adaptive_sampling_freq=2500,
            num_new_points=2500)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        n_subs, splits = get_split_info(model)
        return pred_mph, train_time, n_params, n_subs, ';'.join(splits)

    elif method == 'B8':
        model = AdaStdpinnLWR(
            domain_bounds, data_points, layers=layers,
            layers_after_split=layers_after,
            w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
            use_rh_interface=False,
            use_entropy=False,
            use_adaptive_loss=False,
            use_rar=False,
            use_causal_weighting=False,
            use_spatial_decomp=True,
            use_temporal_decomp=True,
            **norm_params)
        model.train_xpinn(
            epochs=epochs, batch_size=batch_size,
            N_f=N_f, n_spatial=2, n_temporal=2)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        n_subs, splits = get_split_info(model)
        return pred_mph, train_time, n_params, n_subs, ';'.join(splits)

    else:
        raise ValueError(f"Unknown method: {method}")


# ======================================================================
# Main experiment loop
# ======================================================================

def run_all():
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'predictions'), exist_ok=True)

    # Load dataset once
    csv_file = os.path.join(DATA_DIR, NGSIM_CSV)
    dataset = load_dataset(csv_file, X_RANGE, T_RANGE)
    n_x = dataset['n_x']

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"NGSIM: v_f={dataset['v_f_mph']:.2f} mph, "
          f"grid={n_x}x{dataset['n_t']}, "
          f"speed=[{dataset['u_min']:.1f}, {dataset['u_max']:.1f}] mph")
    print(f"Methods: {METHODS}")
    print(f"Sensors: {SENSOR_COUNTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {PARAMS['epochs']}")
    print(f"Output: {CSV_PATH}")

    completed = load_completed(CSV_PATH)
    total_runs = len(METHODS) * len(SENSOR_COUNTS) * len(SEEDS)
    done = len(completed)
    print(f"\nTotal: {total_runs} runs, {done} already completed, "
          f"{total_runs - done} remaining\n")

    pred_dir = os.path.join(RESULTS_DIR, 'predictions')

    for n_sensors in SENSOR_COUNTS:
        sensor_idx = get_sensor_indices(n_x, n_sensors)

        for method in METHODS:
            for seed in SEEDS:
                key = ('ngsim', method, n_sensors, seed)
                if key in completed:
                    continue

                done += 1
                print(f"[{done}/{total_runs}] "
                      f"Method={method}, Sensors={n_sensors}, Seed={seed}",
                      end=' ... ', flush=True)

                data_points, sensor_x_norm = get_sensor_data(
                    dataset, sensor_idx, device)

                try:
                    pred_mph, train_time, n_params, n_subs, splits = \
                        run_method(
                            method, dataset, data_points,
                            sensor_x_norm, dataset['norm_params'], seed)

                    metrics = compute_metrics(dataset['Exact_mph'], pred_mph)

                    row = {
                        'Dataset': 'ngsim',
                        'Method': method,
                        'Seed': seed,
                        'Sensors': n_sensors,
                        'L2': metrics['L2'],
                        'RMSE': metrics['RMSE'],
                        'MSE': metrics['MSE'],
                        'Time': train_time,
                        'Subdomains': n_subs,
                        'Splits': splits,
                    }
                    append_row(CSV_PATH, row)
                    completed.add(key)

                    print(f"L2={metrics['L2']:.4f}, "
                          f"RMSE={metrics['RMSE']:.4f}, "
                          f"Time={train_time:.1f}s, Subs={n_subs}")

                    # Save prediction for seed=42
                    if seed == 42:
                        np.save(
                            os.path.join(
                                pred_dir,
                                f'ngsim_{method}_{n_sensors}s'
                                f'_seed42_pred.npy'),
                            pred_mph)

                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    row = {
                        'Dataset': 'ngsim',
                        'Method': method,
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
# Comparison tables
# ======================================================================

def print_comparison():
    """Print comparison tables from results CSV."""
    if not os.path.exists(CSV_PATH):
        print("No results file found.")
        return

    df = pd.read_csv(CSV_PATH).dropna(subset=['L2'])

    # --- Per-sensor summary: mean +- std for each method ---
    print(f"\n{'='*90}")
    print("NGSIM RESULTS: L2 Error (mean +/- std) across 10 seeds")
    print(f"{'='*90}")

    for n_sensors in SENSOR_COUNTS:
        sub = df[df['Sensors'] == n_sensors]
        if sub.empty:
            continue

        print(f"\n--- {n_sensors} Sensors ---")
        print(f"{'Method':<22} {'L2 (mean +/- std)':<22} "
              f"{'RMSE (mean +/- std)':<22} {'Time (s)':<10}")
        print("-" * 76)

        best_l2 = None
        best_method = None
        rows_data = []

        for method in METHODS:
            grp = sub[sub['Method'] == method]
            if grp.empty:
                continue
            l2_mean = grp['L2'].mean()
            l2_std = grp['L2'].std()
            rmse_mean = grp['RMSE'].mean()
            rmse_std = grp['RMSE'].std()
            time_mean = grp['Time'].mean()

            rows_data.append((method, l2_mean, l2_std,
                              rmse_mean, rmse_std, time_mean))

            if best_l2 is None or l2_mean < best_l2:
                best_l2 = l2_mean
                best_method = method

        for (method, l2_m, l2_s, rmse_m, rmse_s, t_m) in rows_data:
            marker = " *" if method == best_method else ""
            name = METHOD_NAMES.get(method, method)
            print(f"{name:<22} {l2_m:.4f} +/- {l2_s:.4f}{marker:<4} "
                  f"{rmse_m:.4f} +/- {rmse_s:.4f}   {t_m:>8.1f}")

    # --- Head-to-head: B7 vs B3, B7 vs B8 ---
    print(f"\n{'='*90}")
    print("HEAD-TO-HEAD COMPARISON (wins/losses across sensor configs)")
    print(f"{'='*90}")

    for opponent in ['B3', 'B8']:
        print(f"\n--- B7 vs {opponent} ---")
        print(f"{'Sensors':<10} {'B7 mean':<12} {f'{opponent} mean':<12} "
              f"{'Winner':<10} {'p-value':<12}")
        print("-" * 56)

        b7_wins = 0
        opp_wins = 0
        ties = 0

        for n_sensors in SENSOR_COUNTS:
            sub = df[df['Sensors'] == n_sensors]
            b7_vals = sub[sub['Method'] == 'B7']['L2'].values
            opp_vals = sub[sub['Method'] == opponent]['L2'].values

            if len(b7_vals) < 2 or len(opp_vals) < 2:
                continue

            b7_mean = b7_vals.mean()
            opp_mean = opp_vals.mean()
            t_stat, p_val = stats.ttest_ind(
                b7_vals, opp_vals, equal_var=False)

            if p_val < 0.05:
                if b7_mean < opp_mean:
                    winner = "B7"
                    b7_wins += 1
                else:
                    winner = opponent
                    opp_wins += 1
            else:
                winner = "tie"
                ties += 1

            print(f"{n_sensors:<10} {b7_mean:<12.4f} {opp_mean:<12.4f} "
                  f"{winner:<10} {p_val:<12.4e}")

        print(f"\nTotal: B7 wins={b7_wins}, {opponent} wins={opp_wins}, "
              f"ties={ties}")

    # --- Overall ranking ---
    print(f"\n{'='*90}")
    print("OVERALL RANKING (mean L2 across all sensors)")
    print(f"{'='*90}")

    overall = []
    for method in METHODS:
        grp = df[df['Method'] == method]
        if grp.empty:
            continue
        overall.append({
            'Method': METHOD_NAMES.get(method, method),
            'L2_mean': grp['L2'].mean(),
            'L2_std': grp['L2'].std(),
            'RMSE_mean': grp['RMSE'].mean(),
            'Time_mean': grp['Time'].mean(),
        })

    overall.sort(key=lambda x: x['L2_mean'])
    print(f"\n{'Rank':<6} {'Method':<22} {'L2 mean':<12} {'L2 std':<12} "
          f"{'RMSE mean':<12} {'Time (s)':<10}")
    print("-" * 74)
    for i, r in enumerate(overall, 1):
        print(f"{i:<6} {r['Method']:<22} {r['L2_mean']:<12.4f} "
              f"{r['L2_std']:<12.4f} {r['RMSE_mean']:<12.4f} "
              f"{r['Time_mean']:<10.1f}")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    run_all()
    print_comparison()
