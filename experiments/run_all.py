"""CA-STD-PINNs: Main experiment script.

Part 1: 7 methods x 5 sensor counts x 10 seeds x 6 datasets
Part 2: Ablation study (20221121, 5 sensors, 10 seeds)
Part 3: Computation cost (20221121, 5 sensors, seed=42)
Part 4: Shock indicator values (all datasets)

Resume-safe: saves incremental CSV after every run.
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
    compute_metrics, predict_full_field, linear_interpolation,
    count_params, get_split_info,
)

# ======================================================================
# Configuration
# ======================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]
SENSOR_COUNTS = [3, 4, 5, 6, 7]

# Dataset configs: name -> (csv_path, x_range_ft, t_range_s)
DATASETS = {
    '20221121': ('i24/20221121.csv', 21120.0, 14400.0),
    '20221122': ('i24/20221122.csv', 21120.0, 14400.0),
    '20221123': ('i24/20221123.csv', 21120.0, 14400.0),
    '20221129': ('i24/20221129.csv', 21120.0, 14400.0),
    '20221202': ('i24/20221202.csv', 21120.0, 14400.0),
    'ngsim':    ('ngsim/ngsim_data.csv', 1600.0, 900.0),
}

I24_PARAMS = {
    'N_f': 50000, 'batch_size': 4096,
    'layers': [2, 256, 128, 128, 128, 1],
    'layers_after_split': [2, 256, 128, 128, 1],
    'epochs': 20000,
}

NGSIM_PARAMS = {
    'N_f': 10000, 'batch_size': 512,
    'layers': [2, 128, 64, 64, 1],
    'layers_after_split': [2, 128, 64, 1],
    'epochs': 20000,
}

# Priority order
DATASET_ORDER = [
    '20221121', '20221122', '20221202',
    '20221129', '20221123', 'ngsim'
]

METHODS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
METHOD_NAMES = {
    'B1': 'Linear Interp.',
    'B2': 'Simple NN',
    'B3': 'Vanilla PINN',
    'B4': 'PINN + RAR',
    'B5': 'PINN + Viscosity',
    'B6': 'cPINN',
    'B7': 'CA-STD-PINN (Ours)',
    'B8': 'XPINN',
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_params(dataset_name):
    """Get hyperparameters for a dataset."""
    if dataset_name == 'ngsim':
        return NGSIM_PARAMS
    return I24_PARAMS


def cleanup_gpu():
    """Free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ======================================================================
# Model creation and training
# ======================================================================

def run_method(method, dataset, data_points, sensor_x_norm, params,
               norm_params, seed):
    """Create, train, and predict for one method.

    Returns (pred_mph, train_time, n_params, n_subdomains, splits_str).
    For B1, pred_mph is computed directly (no model).
    """
    domain_bounds = get_domain_bounds(dataset)
    layers = params['layers']
    layers_after = params['layers_after_split']
    epochs = params['epochs']
    N_f = params['N_f']
    batch_size = params['batch_size']

    set_seed(seed)
    start_time = time.time()

    # --- B1: Linear Interpolation ---
    if method == 'B1':
        sensor_idx = []
        x_axis = dataset['x_axis']
        for sx in sensor_x_norm:
            sensor_idx.append(np.argmin(np.abs(x_axis - sx)))
        pred_mph = linear_interpolation(dataset, np.array(sensor_idx))
        return pred_mph, 0.0, 0, 0, ''

    # --- B2: Simple NN ---
    elif method == 'B2':
        model = SimpleNN(data_points, layers=layers)
        model.train(epochs=epochs, batch_size=batch_size)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        return pred_mph, train_time, n_params, 1, ''

    # --- B3: Vanilla PINN ---
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

    # --- B4: PINN + RAR ---
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

    # --- B5: PINN + Artificial Viscosity ---
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
        # Fix weights to non-trainable
        model.w_data = torch.tensor(
            0.85, requires_grad=False, device=device, dtype=torch.float32)
        model.w_pde = torch.tensor(
            0.05, requires_grad=False, device=device, dtype=torch.float32)
        model.w_int = torch.tensor(
            0.10, requires_grad=False, device=device, dtype=torch.float32)
        # Rebuild optimizer (exclude weight tensors)
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

    # --- B6: cPINN (online, equal-spaced, flux continuity) ---
    elif method == 'B6':
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
            use_temporal_decomp=False,
            **norm_params)
        model.train_cpinn(
            epochs=epochs, batch_size=batch_size,
            N_f=N_f, n_subdomains=3)
        train_time = time.time() - start_time
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        n_params = count_params(model)
        n_subs, splits = get_split_info(model)
        return pred_mph, train_time, n_params, n_subs, ';'.join(splits)

    # --- B7: Our Method (Two-Stage DD) ---
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

    # --- B8: XPINN (online, space-time grid, residual continuity) ---
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
# Ablation variants (Part 2)
# ======================================================================

def run_ablation(variant, dataset, data_points, sensor_x_norm, params,
                 norm_params, seed):
    """Run one ablation variant. Returns (pred_mph, train_time, n_params, n_subs)."""
    domain_bounds = get_domain_bounds(dataset)
    layers = params['layers']
    layers_after = params['layers_after_split']
    epochs = params['epochs']
    N_f = params['N_f']
    batch_size = params['batch_size']

    set_seed(seed)
    start_time = time.time()

    if variant == 'A1':
        # Full model (same as B7)
        model = AdaStdpinnLWR(
            domain_bounds, data_points, layers=layers,
            layers_after_split=layers_after,
            w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
            use_rh_interface=True, use_entropy=True,
            use_adaptive_loss=False, use_rar=True,
            use_causal_weighting=True, use_spatial_decomp=True,
            shock_indicator_threshold=2.0, **norm_params)
        model.train_two_stage(
            total_epochs=epochs, stage1_epochs=5000,
            batch_size=batch_size, N_f=N_f,
            force_decomp=True,
            adaptive_sampling_freq=2500, num_new_points=2500)

    elif variant == 'A2':
        # B7 minus Two-Stage = Online DD with stability fixes
        model = AdaStdpinnLWR(
            domain_bounds, data_points, layers=layers,
            layers_after_split=layers_after,
            w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
            use_rh_interface=True, use_entropy=True,
            use_adaptive_loss=False, use_rar=True,
            use_causal_weighting=True, use_spatial_decomp=True,
            shock_indicator_threshold=0.0, **norm_params)
        # Fix weights to non-trainable
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
            adaptive_sampling_freq=2500,
            domain_decomp_freq=5000,
            num_new_points=2500,
            residual_threshold=1e-3)

    elif variant == 'A3':
        # B7 minus R-H = flux continuity at interfaces
        model = AdaStdpinnLWR(
            domain_bounds, data_points, layers=layers,
            layers_after_split=layers_after,
            w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
            use_rh_interface=False, use_entropy=False,
            use_adaptive_loss=False, use_rar=True,
            use_causal_weighting=True, use_spatial_decomp=True,
            shock_indicator_threshold=2.0, **norm_params)
        model.train_two_stage(
            total_epochs=epochs, stage1_epochs=5000,
            batch_size=batch_size, N_f=N_f,
            force_decomp=True,
            adaptive_sampling_freq=2500, num_new_points=2500)

    elif variant == 'A4':
        # B7 plus Adaptive Loss = adaptive weights instead of fixed
        model = AdaStdpinnLWR(
            domain_bounds, data_points, layers=layers,
            layers_after_split=layers_after,
            w_data_init=10.0, w_pde_init=1.0, w_int_init=1.0,
            use_rh_interface=True, use_entropy=True,
            use_adaptive_loss=True, use_rar=True,
            use_causal_weighting=True, use_spatial_decomp=True,
            shock_indicator_threshold=2.0, **norm_params)
        model.train_two_stage(
            total_epochs=epochs, stage1_epochs=5000,
            batch_size=batch_size, N_f=N_f,
            force_decomp=True,
            adaptive_sampling_freq=2500, num_new_points=2500)

    else:
        raise ValueError(f"Unknown ablation variant: {variant}")

    train_time = time.time() - start_time
    pred_mph = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    n_params = count_params(model)
    n_subs, _ = get_split_info(model)
    return pred_mph, train_time, n_params, n_subs


# ======================================================================
# Resume helpers
# ======================================================================

def load_completed(csv_path):
    """Load set of completed (Dataset, Method, Sensors, Seed) tuples."""
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
    """Append a single result row to CSV (create if needed)."""
    df = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ======================================================================
# Part 0: Ground truth and sensor configs
# ======================================================================

def save_ground_truth():
    """Save ground truth arrays and sensor configs."""
    gt_dir = os.path.join(RESULTS_DIR, 'ground_truth')
    sc_dir = os.path.join(RESULTS_DIR, 'sensor_configs')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(sc_dir, exist_ok=True)

    for ds_name in DATASET_ORDER:
        csv_rel, x_range, t_range = DATASETS[ds_name]
        csv_path = os.path.join(DATA_DIR, csv_rel)
        dataset = load_dataset(csv_path, x_range, t_range)

        np.save(os.path.join(gt_dir, f'{ds_name}_ground_truth.npy'),
                dataset['Exact_mph'])
        np.save(os.path.join(gt_dir, f'{ds_name}_x_coords.npy'),
                dataset['X'])
        np.save(os.path.join(gt_dir, f'{ds_name}_t_coords.npy'),
                dataset['T'])

        print(f"[GT] {ds_name}: shape={dataset['Exact_mph'].shape}, "
              f"v_f={dataset['v_f_mph']:.2f} mph, "
              f"speed=[{dataset['u_min']:.1f}, {dataset['u_max']:.1f}] mph")

    # Sensor configs
    for n_s in SENSOR_COUNTS:
        # I-24
        idx_i24 = get_sensor_indices(100, n_s)
        np.save(os.path.join(sc_dir, f'sensors_i24_{n_s}.npy'), idx_i24)
        # NGSIM
        idx_ngsim = get_sensor_indices(81, n_s)
        np.save(os.path.join(sc_dir, f'sensors_ngsim_{n_s}.npy'), idx_ngsim)
        print(f"[SC] {n_s} sensors: I-24={idx_i24}, NGSIM={idx_ngsim}")


# ======================================================================
# Part 1: Main comparison
# ======================================================================

def run_part1():
    """Main comparison: 7 methods x 5 sensors x 10 seeds x 6 datasets."""
    csv_path = os.path.join(RESULTS_DIR, 'tables', 'per_seed_all.csv')
    pred_dir = os.path.join(RESULTS_DIR, 'predictions')
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    completed = load_completed(csv_path)
    total_runs = (len(DATASET_ORDER) * len(METHODS)
                  * len(SENSOR_COUNTS) * len(SEEDS))
    run_idx = len(completed)

    for ds_name in DATASET_ORDER:
        csv_rel, x_range, t_range = DATASETS[ds_name]
        csv_file = os.path.join(DATA_DIR, csv_rel)
        dataset = load_dataset(csv_file, x_range, t_range)
        params = get_params(ds_name)
        n_x = dataset['n_x']

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} | v_f={dataset['v_f_mph']:.2f} mph | "
              f"grid={dataset['n_x']}x{dataset['n_t']}")
        print(f"{'='*70}")

        for n_sensors in SENSOR_COUNTS:
            sensor_idx = get_sensor_indices(n_x, n_sensors)

            for method in METHODS:
                seeds_to_run = [SEEDS[0]] if method == 'B1' else SEEDS

                for seed in seeds_to_run:
                    key = (ds_name, method, n_sensors, seed)
                    if key in completed:
                        continue

                    run_idx += 1
                    print(f"\nRun {run_idx}/{total_runs}: "
                          f"Dataset={ds_name}, Method={method}, "
                          f"Sensors={n_sensors}, Seed={seed}")

                    data_points, sensor_x_norm = get_sensor_data(
                        dataset, sensor_idx, device)

                    try:
                        pred_mph, train_time, n_params, n_subs, splits = \
                            run_method(
                                method, dataset, data_points,
                                sensor_x_norm, params,
                                dataset['norm_params'], seed)

                        metrics = compute_metrics(
                            dataset['Exact_mph'], pred_mph)

                        row = {
                            'Dataset': ds_name,
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
                        append_row(csv_path, row)
                        completed.add(key)

                        print(f"  -> L2={metrics['L2']:.4f}, "
                              f"RMSE={metrics['RMSE']:.4f}, "
                              f"Time={train_time:.1f}s, "
                              f"Subs={n_subs}")

                        # Save prediction for seed=42
                        if seed == 42:
                            np.save(
                                os.path.join(
                                    pred_dir,
                                    f'{ds_name}_{method}_{n_sensors}s'
                                    f'_seed42_pred.npy'),
                                pred_mph)

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        row = {
                            'Dataset': ds_name,
                            'Method': method,
                            'Seed': seed,
                            'Sensors': n_sensors,
                            'L2': np.nan, 'RMSE': np.nan, 'MSE': np.nan,
                            'Time': np.nan, 'Subdomains': 0, 'Splits': '',
                        }
                        append_row(csv_path, row)
                        completed.add(key)

                    cleanup_gpu()

    # Generate summary
    generate_summary()


def generate_summary():
    """Compute mean+-std and Welch's t-test tables."""
    csv_path = os.path.join(RESULTS_DIR, 'tables', 'per_seed_all.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['L2'])

    # For B1 (1 run per config), replicate the result for summary
    summary_rows = []
    for (ds, method, sensors), grp in df.groupby(
            ['Dataset', 'Method', 'Sensors']):
        summary_rows.append({
            'Dataset': ds, 'Method': method, 'Sensors': sensors,
            'L2_mean': grp['L2'].mean(),
            'L2_std': grp['L2'].std() if len(grp) > 1 else 0.0,
            'RMSE_mean': grp['RMSE'].mean(),
            'RMSE_std': grp['RMSE'].std() if len(grp) > 1 else 0.0,
            'MSE_mean': grp['MSE'].mean(),
            'MSE_std': grp['MSE'].std() if len(grp) > 1 else 0.0,
            'Time_mean': grp['Time'].mean(),
            'n_runs': len(grp),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, 'tables', 'summary_all.csv'), index=False)
    print("\nSaved summary_all.csv")

    # Welch's t-test: each method vs B7
    ttest_rows = []
    for (ds, sensors), grp in df.groupby(['Dataset', 'Sensors']):
        b7 = grp[grp['Method'] == 'B7']['L2'].values
        if len(b7) < 2:
            continue
        for method in METHODS:
            if method == 'B7':
                continue
            other = grp[grp['Method'] == method]['L2'].values
            if len(other) < 2:
                ttest_rows.append({
                    'Dataset': ds, 'Sensors': sensors,
                    'Method': method, 'p_value': np.nan,
                    't_stat': np.nan,
                })
                continue
            t_stat, p_val = stats.ttest_ind(other, b7, equal_var=False)
            ttest_rows.append({
                'Dataset': ds, 'Sensors': sensors,
                'Method': method, 'p_value': p_val,
                't_stat': t_stat,
            })

    if ttest_rows:
        ttest_df = pd.DataFrame(ttest_rows)
        ttest_df.to_csv(
            os.path.join(RESULTS_DIR, 'tables', 'ttest_vs_B7.csv'),
            index=False)
        print("Saved ttest_vs_B7.csv")


# ======================================================================
# Part 2: Ablation study
# ======================================================================

def run_part2():
    """Ablation: A1-A4 on 20221121, 5 sensors, 10 seeds."""
    csv_path = os.path.join(RESULTS_DIR, 'tables', 'ablation_per_seed.csv')
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

    completed = load_completed(csv_path)

    ds_name = '20221121'
    n_sensors = 5
    csv_rel, x_range, t_range = DATASETS[ds_name]
    csv_file = os.path.join(DATA_DIR, csv_rel)
    dataset = load_dataset(csv_file, x_range, t_range)
    params = get_params(ds_name)
    sensor_idx = get_sensor_indices(dataset['n_x'], n_sensors)

    variants = ['A1', 'A2', 'A3', 'A4']
    total = len(variants) * len(SEEDS)
    run_i = 0

    for variant in variants:
        for seed in SEEDS:
            key = (ds_name, variant, n_sensors, seed)
            if key in completed:
                run_i += 1
                continue

            run_i += 1
            print(f"\n[Ablation] Run {run_i}/{total}: "
                  f"{variant}, Seed={seed}")

            data_points, sensor_x_norm = get_sensor_data(
                dataset, sensor_idx, device)

            try:
                pred_mph, train_time, n_params, n_subs = run_ablation(
                    variant, dataset, data_points, sensor_x_norm,
                    params, dataset['norm_params'], seed)

                metrics = compute_metrics(dataset['Exact_mph'], pred_mph)

                row = {
                    'Dataset': ds_name, 'Method': variant,
                    'Seed': seed, 'Sensors': n_sensors,
                    'L2': metrics['L2'], 'RMSE': metrics['RMSE'],
                    'MSE': metrics['MSE'], 'Time': train_time,
                    'Subdomains': n_subs, 'Splits': '',
                }
                append_row(csv_path, row)
                completed.add(key)

                print(f"  -> L2={metrics['L2']:.4f}, "
                      f"Time={train_time:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                row = {
                    'Dataset': ds_name, 'Method': variant,
                    'Seed': seed, 'Sensors': n_sensors,
                    'L2': np.nan, 'RMSE': np.nan, 'MSE': np.nan,
                    'Time': np.nan, 'Subdomains': 0, 'Splits': '',
                }
                append_row(csv_path, row)
                completed.add(key)

            cleanup_gpu()

    # Summary + t-test
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).dropna(subset=['L2'])
        summary_rows = []
        for variant, grp in df.groupby('Method'):
            summary_rows.append({
                'Method': variant,
                'L2_mean': grp['L2'].mean(),
                'L2_std': grp['L2'].std(),
                'RMSE_mean': grp['RMSE'].mean(),
                'RMSE_std': grp['RMSE'].std(),
                'Time_mean': grp['Time'].mean(),
                'n_runs': len(grp),
            })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(RESULTS_DIR, 'tables', 'ablation_summary.csv'),
            index=False)

        # t-test vs A1
        a1_vals = df[df['Method'] == 'A1']['L2'].values
        ttest_rows = []
        for variant in ['A2', 'A3', 'A4']:
            other = df[df['Method'] == variant]['L2'].values
            if len(other) >= 2 and len(a1_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(
                    other, a1_vals, equal_var=False)
            else:
                t_stat, p_val = np.nan, np.nan
            ttest_rows.append({
                'Method': variant, 'p_value': p_val, 't_stat': t_stat,
            })
        pd.DataFrame(ttest_rows).to_csv(
            os.path.join(RESULTS_DIR, 'tables', 'ttest_ablation_vs_A1.csv'),
            index=False)
        print("\nSaved ablation tables.")


# ======================================================================
# Part 3: Computation cost
# ======================================================================

def run_part3():
    """Computation cost: all methods, 20221121, 5 sensors, seed=42."""
    ds_name = '20221121'
    n_sensors = 5
    seed = 42

    csv_rel, x_range, t_range = DATASETS[ds_name]
    csv_file = os.path.join(DATA_DIR, csv_rel)
    dataset = load_dataset(csv_file, x_range, t_range)
    params = get_params(ds_name)
    sensor_idx = get_sensor_indices(dataset['n_x'], n_sensors)
    data_points, sensor_x_norm = get_sensor_data(
        dataset, sensor_idx, device)

    rows = []
    for method in METHODS:
        print(f"\n[Cost] Method={method}")
        set_seed(seed)

        pred_mph, train_time, n_params, n_subs, splits = run_method(
            method, dataset, data_points, sensor_x_norm,
            params, dataset['norm_params'], seed)

        metrics = compute_metrics(dataset['Exact_mph'], pred_mph)
        rows.append({
            'Method': method,
            'Name': METHOD_NAMES[method],
            'L2': metrics['L2'],
            'RMSE': metrics['RMSE'],
            'Time': train_time,
            'Parameters': n_params,
            'Subdomains': n_subs,
        })
        print(f"  -> L2={metrics['L2']:.4f}, Time={train_time:.1f}s, "
              f"Params={n_params}, Subs={n_subs}")
        cleanup_gpu()

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, 'tables', 'computation_cost.csv'),
        index=False)
    print("\nSaved computation_cost.csv")


# ======================================================================
# Part 4: Shock indicators
# ======================================================================

def run_part4():
    """Compute shock indicators for all datasets."""
    rows = []
    for ds_name in DATASET_ORDER:
        csv_rel, x_range, t_range = DATASETS[ds_name]
        csv_file = os.path.join(DATA_DIR, csv_rel)
        dataset = load_dataset(csv_file, x_range, t_range)

        # Use 5 sensors for shock indicator computation
        n_sensors = 5
        sensor_idx = get_sensor_indices(dataset['n_x'], n_sensors)
        data_points, sensor_x_norm = get_sensor_data(
            dataset, sensor_idx, device)
        domain_bounds = get_domain_bounds(dataset)
        params = get_params(ds_name)

        # Create a temporary model just to compute shock indicators
        model = AdaStdpinnLWR(
            domain_bounds, data_points,
            layers=params['layers'],
            layers_after_split=params['layers_after_split'],
            **dataset['norm_params'])
        model._compute_shock_indicators()

        spatial = getattr(model, '_spatial_shock_indicator', 0.0)
        temporal = getattr(model, '_temporal_shock_indicator', 0.0)

        # Congestion percentage: fraction of speeds below 45 mph
        speeds = dataset['Exact_mph'].flatten()
        congestion_pct = float(np.mean(speeds < 45.0) * 100)

        rows.append({
            'Dataset': ds_name,
            'Spatial_Indicator': spatial,
            'Temporal_Indicator': temporal,
            'Congestion_Pct': congestion_pct,
            'v_f_mph': dataset['v_f_mph'],
        })
        print(f"[Shock] {ds_name}: spatial={spatial:.3f}, "
              f"temporal={temporal:.3f}, "
              f"congestion={congestion_pct:.1f}%")
        cleanup_gpu()

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, 'tables', 'shock_indicators.csv'),
        index=False)
    print("\nSaved shock_indicators.csv")


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Seeds: {SEEDS}")
    print(f"Sensors: {SENSOR_COUNTS}")
    print(f"Datasets: {DATASET_ORDER}")
    print()

    # Step 0: Ground truth + sensor configs
    print("=" * 70)
    print("STEP 0: Saving ground truth arrays and sensor configs")
    print("=" * 70)
    save_ground_truth()

    # Step 1: Main comparison
    print("\n" + "=" * 70)
    print("STEP 1: Main comparison (Part 1)")
    print("=" * 70)
    run_part1()

    # Step 2: Ablation (Part 2)
    print("\n" + "=" * 70)
    print("STEP 2: Ablation study (Part 2)")
    print("=" * 70)
    run_part2()

    # Step 3: Computation cost (Part 3)
    print("\n" + "=" * 70)
    print("STEP 3: Computation cost (Part 3)")
    print("=" * 70)
    run_part3()

    # Step 4: Shock indicators (Part 4)
    print("\n" + "=" * 70)
    print("STEP 4: Shock indicators (Part 4)")
    print("=" * 70)
    run_part4()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
