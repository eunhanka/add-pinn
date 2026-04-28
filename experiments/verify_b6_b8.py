"""Quick verification: B3, B6, B7, B8 on 20221121, 5 sensors, seed=42.

Runs each method once and prints L2/RMSE/time to confirm correctness.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
import gc

from src.model import AdaStdpinnLWR, device
from src.utils import (
    load_dataset, get_sensor_indices, get_sensor_data, get_domain_bounds,
    compute_metrics, predict_full_field, count_params, get_split_info,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

DS_CONFIG = ('i24/20221121.csv', 21120.0, 14400.0)
SEED = 42
N_SENSORS = 5
EPOCHS = 500  # short run for verification

LAYERS = [2, 256, 128, 128, 128, 1]
LAYERS_AFTER = [2, 256, 128, 128, 1]
N_F = 10000
BATCH_SIZE = 2048


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    csv_file = os.path.join(DATA_DIR, DS_CONFIG[0])
    dataset = load_dataset(csv_file, DS_CONFIG[1], DS_CONFIG[2])
    domain_bounds = get_domain_bounds(dataset)
    norm_params = dataset['norm_params']
    n_x = dataset['n_x']
    sensor_idx = get_sensor_indices(n_x, N_SENSORS)
    data_points, sensor_x_norm = get_sensor_data(dataset, sensor_idx, device)

    print(f"Device: {device}")
    print(f"Dataset: 20221121 | Sensors: {N_SENSORS} | Seed: {SEED}")
    print(f"Epochs: {EPOCHS} (short verification run)")
    print()

    results = {}

    # --- B3: Vanilla PINN ---
    print("=" * 60)
    print("B3: Vanilla PINN")
    print("=" * 60)
    set_seed(SEED)
    model = AdaStdpinnLWR(
        domain_bounds, data_points, layers=LAYERS,
        w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
        use_adaptive_loss=False, use_rar=False,
        use_causal_weighting=False, use_spatial_decomp=False,
        use_temporal_decomp=False, use_rh_interface=False,
        use_entropy=False, **norm_params)
    start = time.time()
    model.train(
        epochs=EPOCHS, batch_size=BATCH_SIZE, N_f=N_F,
        adaptive_sampling_freq=999999, domain_decomp_freq=999999,
        num_new_points=0, residual_threshold=999.0)
    t_b3 = time.time() - start
    pred = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    m = compute_metrics(dataset['Exact_mph'], pred)
    n_p = count_params(model)
    n_s, sp = get_split_info(model)
    results['B3'] = (m['L2'], m['RMSE'], t_b3, n_p, n_s)
    print(f"  L2={m['L2']:.4f} RMSE={m['RMSE']:.4f} "
          f"Time={t_b3:.1f}s Params={n_p} Subs={n_s}")
    cleanup_gpu()

    # --- B6: cPINN ---
    print("\n" + "=" * 60)
    print("B6: cPINN (3 equal-spaced subdomains, flux continuity)")
    print("=" * 60)
    set_seed(SEED)
    model = AdaStdpinnLWR(
        domain_bounds, data_points, layers=LAYERS,
        layers_after_split=LAYERS_AFTER,
        w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
        use_rh_interface=False, use_entropy=False,
        use_adaptive_loss=False, use_rar=False,
        use_causal_weighting=False, use_spatial_decomp=True,
        use_temporal_decomp=False, **norm_params)
    start = time.time()
    model.train_cpinn(
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        N_f=N_F, n_subdomains=3)
    t_b6 = time.time() - start
    pred = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    m = compute_metrics(dataset['Exact_mph'], pred)
    n_p = count_params(model)
    n_s, sp = get_split_info(model)
    results['B6'] = (m['L2'], m['RMSE'], t_b6, n_p, n_s)
    print(f"  L2={m['L2']:.4f} RMSE={m['RMSE']:.4f} "
          f"Time={t_b6:.1f}s Params={n_p} Subs={n_s}")
    cleanup_gpu()

    # --- B7: CA-STD-PINN ---
    print("\n" + "=" * 60)
    print("B7: CA-STD-PINN (two-stage, residual-based decomp)")
    print("=" * 60)
    set_seed(SEED)
    model = AdaStdpinnLWR(
        domain_bounds, data_points, layers=LAYERS,
        layers_after_split=LAYERS_AFTER,
        w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
        use_rh_interface=True, use_entropy=True,
        use_adaptive_loss=False, use_rar=True,
        use_causal_weighting=True, use_spatial_decomp=True,
        use_temporal_decomp=False, shock_indicator_threshold=2.0,
        **norm_params)
    start = time.time()
    model.train_two_stage(
        total_epochs=EPOCHS, stage1_epochs=200,
        batch_size=BATCH_SIZE, N_f=N_F,
        force_decomp=True,
        adaptive_sampling_freq=250, num_new_points=500)
    t_b7 = time.time() - start
    pred = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    m = compute_metrics(dataset['Exact_mph'], pred)
    n_p = count_params(model)
    n_s, sp = get_split_info(model)
    results['B7'] = (m['L2'], m['RMSE'], t_b7, n_p, n_s)
    print(f"  L2={m['L2']:.4f} RMSE={m['RMSE']:.4f} "
          f"Time={t_b7:.1f}s Params={n_p} Subs={n_s}")
    cleanup_gpu()

    # --- B8: XPINN ---
    print("\n" + "=" * 60)
    print("B8: XPINN (2x2 grid, residual continuity)")
    print("=" * 60)
    set_seed(SEED)
    model = AdaStdpinnLWR(
        domain_bounds, data_points, layers=LAYERS,
        layers_after_split=LAYERS_AFTER,
        w_data_init=0.85, w_pde_init=0.05, w_int_init=0.10,
        use_rh_interface=False, use_entropy=False,
        use_adaptive_loss=False, use_rar=False,
        use_causal_weighting=False, use_spatial_decomp=True,
        use_temporal_decomp=True, **norm_params)
    start = time.time()
    model.train_xpinn(
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        N_f=N_F, n_spatial=2, n_temporal=2)
    t_b8 = time.time() - start
    pred = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    m = compute_metrics(dataset['Exact_mph'], pred)
    n_p = count_params(model)
    n_s, sp = get_split_info(model)
    results['B8'] = (m['L2'], m['RMSE'], t_b8, n_p, n_s)
    print(f"  L2={m['L2']:.4f} RMSE={m['RMSE']:.4f} "
          f"Time={t_b8:.1f}s Params={n_p} Subs={n_s}")
    cleanup_gpu()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY (500 epochs)")
    print("=" * 60)
    print(f"{'Method':<12} {'L2':>8} {'RMSE':>8} {'Time':>8} "
          f"{'Params':>8} {'Subs':>5}")
    print("-" * 52)
    for meth, (l2, rmse, t, np_, ns) in results.items():
        print(f"{meth:<12} {l2:8.4f} {rmse:8.4f} {t:7.1f}s "
              f"{np_:8d} {ns:5d}")


if __name__ == '__main__':
    main()
