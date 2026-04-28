"""Re-extract B7 residual profiles for all sensor counts.

Reruns B7 only to capture R(x) and split info (predictions already saved).
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
import gc

from src.model import AdaStdpinnLWR, device
from src.utils import (
    load_dataset, get_sensor_indices, get_sensor_data, get_domain_bounds,
    compute_metrics, predict_full_field, get_split_info,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIG_DATA_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', 'data')

SEED = 42
SENSOR_COUNTS = [3, 4, 5, 6, 7]

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


def extract_residual_profile(model):
    """Extract R(x) from the first subdomain's network (needs grads)."""
    b0 = model.subdomains[0]['bounds']
    b_last = model.subdomains[-1]['bounds']
    x_left, x_right = b0[2], b_last[3]
    t_left, t_right = b0[0], b0[1]

    n_x, n_t = 200, 100
    x_vals = torch.linspace(x_left, x_right, n_x, device=device)
    t_vals = torch.linspace(t_left, t_right, n_t, device=device)

    x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing='ij')
    x_flat = x_grid.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)

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

    # Split positions from subdomain bounds
    split_positions = []
    for i in range(len(model.subdomains) - 1):
        split_positions.append(float(model.subdomains[i]['bounds'][3]))

    # Peak detection on R(x)
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


def main():
    csv_file = os.path.join(DATA_DIR, 'i24', '20221121.csv')
    dataset = load_dataset(csv_file, 21120.0, 14400.0)
    norm_params = dataset['norm_params']
    n_x = dataset['n_x']

    print(f"Device: {device}")
    print(f"Re-extracting B7 residual profiles for {SENSOR_COUNTS}\n")

    for n_sensors in SENSOR_COUNTS:
        print(f"\n{'='*50}")
        print(f"Sensors={n_sensors}")
        print(f"{'='*50}")

        sensor_idx = get_sensor_indices(n_x, n_sensors)
        data_points, sensor_x_norm = get_sensor_data(
            dataset, sensor_idx, device)
        domain_bounds = get_domain_bounds(dataset)

        set_seed(SEED)
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

        # Verify prediction matches saved
        pred_mph = predict_full_field(
            model, dataset['X'], dataset['T'],
            dataset['u_min'], dataset['u_max'], device)
        metrics = compute_metrics(dataset['Exact_mph'], pred_mph)
        print(f"  L2={metrics['L2']:.4f}")

        # Extract residual profile
        try:
            (x_np, R_np, R_smooth,
             peak_positions, peak_values, split_positions) = \
                extract_residual_profile(model)

            np.save(os.path.join(
                FIG_DATA_DIR, f'residual_profile_{n_sensors}s.npy'),
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
                    FIG_DATA_DIR, f'split_info_{n_sensors}s.json'), 'w') as f:
                json.dump(split_info, f, indent=2)

            print(f"  Saved residual profile + split info. "
                  f"Peaks={len(peak_positions)}, "
                  f"Splits={split_positions}, Subs={n_subs}")
        except Exception as e:
            print(f"  ERROR extracting residual: {e}")
            import traceback
            traceback.print_exc()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\nDone.")


if __name__ == '__main__':
    main()
