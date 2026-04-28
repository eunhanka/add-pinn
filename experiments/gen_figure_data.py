"""Generate prediction data for paper figures.

Runs B3 (Vanilla PINN), B8 (XPINN), B7 (ADD-PINN) on 20221121
for all 5 sensor configs, seed=42.

Saves ground truth, predictions, error fields, and B7 residual profiles.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
import gc

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
FIG_DATA_DIR = os.path.join(RESULTS_DIR, 'figures', 'data')

SEED = 42
SENSOR_COUNTS = [3, 4, 5, 6, 7]
METHODS = ['B3', 'B8', 'B7']  # paper order: B2(PINN), B5(XPINN), B6(ADD-PINN)

I24_PARAMS = {
    'N_f': 50000, 'batch_size': 4096,
    'layers': [2, 256, 128, 128, 128, 1],
    'layers_after_split': [2, 256, 128, 128, 1],
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


def run_b3(dataset, data_points, norm_params):
    """Vanilla PINN (paper: B2)."""
    domain_bounds = get_domain_bounds(dataset)
    set_seed(SEED)
    start = time.time()
    model = VanillaPinnLWR(
        domain_bounds, data_points,
        layers=I24_PARAMS['layers'],
        w_data_init=0.85, w_pde_init=0.05, **norm_params)
    model.train(epochs=I24_PARAMS['epochs'],
                batch_size=I24_PARAMS['batch_size'],
                N_f=I24_PARAMS['N_f'])
    train_time = time.time() - start
    pred_mph = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    return pred_mph, train_time, model


def run_b8(dataset, data_points, norm_params):
    """XPINN (paper: B5)."""
    domain_bounds = get_domain_bounds(dataset)
    set_seed(SEED)
    start = time.time()
    model = AdaStdpinnLWR(
        domain_bounds, data_points,
        layers=I24_PARAMS['layers'],
        layers_after_split=I24_PARAMS['layers_after_split'],
        w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
        use_rh_interface=False, use_entropy=False,
        use_adaptive_loss=False, use_rar=False,
        use_causal_weighting=False,
        use_spatial_decomp=True, use_temporal_decomp=True,
        **norm_params)
    model.train_xpinn(
        epochs=I24_PARAMS['epochs'],
        batch_size=I24_PARAMS['batch_size'],
        N_f=I24_PARAMS['N_f'],
        n_spatial=2, n_temporal=2)
    train_time = time.time() - start
    pred_mph = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    return pred_mph, train_time, model


def run_b7(dataset, data_points, norm_params):
    """ADD-PINN / CA-STD-PINN (paper: B6)."""
    domain_bounds = get_domain_bounds(dataset)
    set_seed(SEED)
    start = time.time()
    model = AdaStdpinnLWR(
        domain_bounds, data_points,
        layers=I24_PARAMS['layers'],
        layers_after_split=I24_PARAMS['layers_after_split'],
        w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
        use_rh_interface=True, use_entropy=True,
        use_adaptive_loss=False, use_rar=True,
        use_causal_weighting=True,
        use_spatial_decomp=True, use_temporal_decomp=False,
        shock_indicator_threshold=2.0,
        **norm_params)
    model.train_two_stage(
        total_epochs=I24_PARAMS['epochs'],
        stage1_epochs=5000,
        batch_size=I24_PARAMS['batch_size'],
        N_f=I24_PARAMS['N_f'],
        force_decomp=True,
        adaptive_sampling_freq=2500,
        num_new_points=2500)
    train_time = time.time() - start
    pred_mph = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    return pred_mph, train_time, model


def extract_residual_profile(model):
    """Extract R(x) residual profile and split info from B7 model."""
    # Recompute residual profile from the first subdomain's network
    # After decomposition, subdomains have been split, so we need
    # to compute over the full domain using all subdomains
    b0 = model.subdomains[0]['bounds']
    b_last = model.subdomains[-1]['bounds']
    x_left = b0[2]
    x_right = b_last[3]
    t_left = b0[0]
    t_right = b0[1]

    n_x = 200
    n_t = 100
    x_vals = torch.linspace(x_left, x_right, n_x, device=device)
    t_vals = torch.linspace(t_left, t_right, n_t, device=device)

    x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing='ij')
    x_flat = x_grid.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)

    # Use the first subdomain's network for the coarse residual
    # (this is what was used for split detection)
    # NOTE: _pde_residual needs gradients for u_x/u_t, so no torch.no_grad()
    net = model.subdomains[0]['net']
    chunk_size = 5000
    all_res = []
    for start in range(0, x_flat.shape[0], chunk_size):
        end = min(start + chunk_size, x_flat.shape[0])
        x_c = x_flat[start:end].detach().requires_grad_(True)
        t_c = t_flat[start:end].detach().requires_grad_(True)
        res = model._pde_residual(net, x_c, t_c,
                                   create_graph=False).detach()
        all_res.append(res)

    residuals = torch.cat(all_res, dim=0)
    res_grid = residuals.reshape(n_x, n_t)
    R_x = torch.mean(torch.square(res_grid), dim=1)

    # Get split positions from subdomain bounds
    split_positions = []
    for i in range(len(model.subdomains) - 1):
        split_positions.append(float(model.subdomains[i]['bounds'][3]))

    # Find peaks in R(x) for visualization
    R_np = R_x.cpu().numpy()
    x_np = x_vals.cpu().numpy()
    kernel_size = max(3, n_x // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    R_smooth = np.convolve(R_np, kernel, mode='same')

    R_max = np.max(R_smooth)
    threshold = 0.30 * R_max
    margin_idx = max(1, int(n_x * 0.10))
    min_dist = max(1, int(n_x * 0.10))

    peaks = []
    for i in range(margin_idx, n_x - margin_idx):
        if (R_smooth[i] > threshold
                and R_smooth[i] >= R_smooth[i - 1]
                and R_smooth[i] >= R_smooth[i + 1]):
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)

    peak_positions = [float(x_np[p]) for p in peaks]
    peak_values = [float(R_np[p]) for p in peaks]

    return (x_np, R_np, R_smooth,
            peak_positions, peak_values, split_positions)


# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs(FIG_DATA_DIR, exist_ok=True)

    # Load dataset
    csv_file = os.path.join(DATA_DIR, 'i24', '20221121.csv')
    dataset = load_dataset(csv_file, 21120.0, 14400.0)
    norm_params = dataset['norm_params']
    n_x = dataset['n_x']
    n_t = dataset['n_t']

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: 20221121, grid={n_t}x{n_x}, "
          f"v_f={dataset['v_f_mph']:.2f} mph")
    print(f"Methods: {METHODS} (paper: B2, B5, B6)")
    print(f"Sensors: {SENSOR_COUNTS}, Seed: {SEED}")
    print(f"Output: {FIG_DATA_DIR}\n")

    # Save ground truth (once)
    gt_path = os.path.join(FIG_DATA_DIR, 'gt_20221121.npy')
    np.save(gt_path, dataset['Exact_mph'])
    print(f"Saved ground truth: {gt_path} shape={dataset['Exact_mph'].shape}")

    # Save axis info for plotting
    axis_info = {
        'x_axis': dataset['x_axis'].tolist(),
        't_axis': dataset['t_axis'].tolist(),
        'x_range_ft': 21120.0,
        't_range_s': 14400.0,
        'x_range_miles': 21120.0 / 5280.0,
        't_range_hours': 14400.0 / 3600.0,
        'u_min': dataset['u_min'],
        'u_max': dataset['u_max'],
        'v_f_mph': dataset['v_f_mph'],
        'n_x': n_x,
        'n_t': n_t,
    }
    with open(os.path.join(FIG_DATA_DIR, 'axis_info.json'), 'w') as f:
        json.dump(axis_info, f, indent=2)

    # Save sensor positions for each config
    for n_sensors in SENSOR_COUNTS:
        sensor_idx = get_sensor_indices(n_x, n_sensors)
        sensor_x_norm = dataset['x_axis'][sensor_idx]
        np.save(os.path.join(FIG_DATA_DIR, f'sensor_x_{n_sensors}s.npy'),
                sensor_x_norm)

    # Dispatch table
    runners = {
        'B3': run_b3,
        'B8': run_b8,
        'B7': run_b7,
    }

    results = []

    for n_sensors in SENSOR_COUNTS:
        sensor_idx = get_sensor_indices(n_x, n_sensors)
        data_points, sensor_x_norm = get_sensor_data(
            dataset, sensor_idx, device)

        for method in METHODS:
            print(f"\n{'='*60}")
            print(f"Method={method}, Sensors={n_sensors}, Seed={SEED}")
            print(f"{'='*60}")

            # Check if already exists
            pred_path = os.path.join(
                FIG_DATA_DIR, f'pred_{method}_{n_sensors}s.npy')
            if os.path.exists(pred_path):
                pred_mph = np.load(pred_path)
                metrics = compute_metrics(dataset['Exact_mph'], pred_mph)
                print(f"  SKIPPED (exists): L2={metrics['L2']:.4f}")
                results.append((method, n_sensors, metrics['L2'], 0.0))

                # For B7, check if residual data exists too
                if method == 'B7':
                    rp = os.path.join(
                        FIG_DATA_DIR, f'residual_profile_{n_sensors}s.npy')
                    if os.path.exists(rp):
                        continue
                    # Need to re-run B7 to get residual profile
                    print("  Re-running B7 for residual profile...")
                else:
                    continue

            pred_mph, train_time, model = runners[method](
                dataset, data_points, norm_params)

            metrics = compute_metrics(dataset['Exact_mph'], pred_mph)
            error_field = pred_mph - dataset['Exact_mph']

            # Save prediction and error
            np.save(pred_path, pred_mph)
            np.save(os.path.join(
                FIG_DATA_DIR, f'error_{method}_{n_sensors}s.npy'),
                error_field)

            print(f"  L2={metrics['L2']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                  f"Time={train_time:.1f}s")
            results.append((method, n_sensors, metrics['L2'], train_time))

            # B7: extract and save residual profile + split info
            if method == 'B7':
                try:
                    (x_np, R_np, R_smooth, peak_positions,
                     peak_values, split_positions) = \
                        extract_residual_profile(model)

                    np.save(os.path.join(
                        FIG_DATA_DIR,
                        f'residual_profile_{n_sensors}s.npy'),
                        np.column_stack([x_np, R_np, R_smooth]))

                    n_subs, splits_str = get_split_info(model)
                    split_info = {
                        'peak_positions': peak_positions,
                        'peak_values': peak_values,
                        'split_positions': split_positions,
                        'subdomain_bounds': [
                            {'x_min': float(sd['bounds'][2]),
                             'x_max': float(sd['bounds'][3])}
                            for sd in model.subdomains
                        ],
                        'n_subdomains': n_subs,
                        'splits_str': splits_str,
                    }
                    with open(os.path.join(
                            FIG_DATA_DIR,
                            f'split_info_{n_sensors}s.json'), 'w') as f:
                        json.dump(split_info, f, indent=2)

                    print(f"  Residual profile saved. "
                          f"Peaks={len(peak_positions)}, "
                          f"Splits={split_positions}, "
                          f"Subs={n_subs}")
                except Exception as e:
                    print(f"  WARNING: Could not extract residual "
                          f"profile: {e}")

            del model
            cleanup_gpu()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<8} {'Sensors':<10} {'L2':<10} {'Time (s)':<10}")
    print("-" * 38)
    for method, n_s, l2, t in results:
        print(f"{method:<8} {n_s:<10} {l2:<10.4f} {t:<10.1f}")


if __name__ == '__main__':
    main()
