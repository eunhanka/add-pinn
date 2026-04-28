"""PHASE 1: Domain Direction Ablation.

Determines whether B7 should use spatial-only or space-time decomposition.

Variants:
  A5_spatial          — Current B7 (spatial-only). Extracted from per_seed_all.csv.
  A6_temporal         — Temporal-only decomposition. NEW.
  A7_spatial_temporal — Spatial + Temporal decomposition. NEW.

Datasets: 20221121 (I-24 Accident), 20221122 (I-24 Normal)
Sensors: 3, 5, 7
Seeds: 10 per combo
Epochs: 20000 (stage1=5000, stage3=15000)

Resume-safe via results/tables/domain_direction_ablation.csv.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import time
import gc

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

SEEDS = [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5555, 7777]
SENSOR_COUNTS = [3, 5, 7]

DATASETS = {
    '20221121': ('i24/20221121.csv', 21120.0, 14400.0),
    '20221122': ('i24/20221122.csv', 21120.0, 14400.0),
}
DATASET_ORDER = ['20221121', '20221122']

PARAMS = {
    'N_f': 50000, 'batch_size': 4096,
    'layers': [2, 256, 128, 128, 128, 1],
    'layers_after_split': [2, 256, 128, 128, 1],
    'epochs': 20000,
    'stage1_epochs': 5000,
}

NEW_METHODS = ['A6_temporal', 'A7_spatial_temporal']


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
# Extract A5 from existing results
# ======================================================================

def extract_a5(csv_out):
    """Extract B7 rows from per_seed_all.csv, rename to A5_spatial."""
    src = os.path.join(RESULTS_DIR, 'tables', 'per_seed_all.csv')
    if not os.path.exists(src):
        print("ERROR: per_seed_all.csv not found!")
        return

    df = pd.read_csv(src)
    mask = (
        (df['Method'] == 'B7')
        & (df['Dataset'].isin(DATASET_ORDER))
        & (df['Sensors'].isin(SENSOR_COUNTS))
    )
    a5 = df[mask].copy()
    a5['Method'] = 'A5_spatial'

    print(f"[A5] Extracted {len(a5)} rows from per_seed_all.csv")

    # Check what A5 rows already exist in output
    completed = load_completed(csv_out)
    added = 0
    for _, row in a5.iterrows():
        key = (str(row['Dataset']), 'A5_spatial',
               int(row['Sensors']), int(row['Seed']))
        if key not in completed:
            append_row(csv_out, row.to_dict())
            added += 1
    print(f"[A5] Added {added} new rows (skipped {len(a5) - added} existing)")


# ======================================================================
# Custom two-stage training with temporal decomposition
# ======================================================================

def train_custom_two_stage(model, method, total_epochs, stage1_epochs,
                           batch_size, N_f,
                           adaptive_sampling_freq=2500, num_new_points=2500):
    """Two-stage training with temporal / spatial+temporal decomposition.

    Replicates train_two_stage logic but replaces Stage 2 with:
      A6_temporal:         temporal split only
      A7_spatial_temporal: spatial splits (as B7) then temporal split per subdomain
    """
    N_f_batch_size = 2048
    stage3_epochs = total_epochs - stage1_epochs
    n_data = model.x_data.shape[0]

    # Save original flags
    orig_use_rar = model.use_rar

    # ==================================================================
    # STAGE 1: Coarse PINN (identical to original train_two_stage)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"[S1] STAGE 1: Coarse PINN training ({stage1_epochs} epochs)")
    print(f"{'='*60}")

    model.use_adaptive_loss = False
    model.use_rar = False
    model.use_spatial_decomp = False
    model.use_temporal_decomp = False

    model.w_data = torch.tensor(
        0.85, requires_grad=False, device=device, dtype=torch.float32)
    model.w_pde = torch.tensor(
        0.05, requires_grad=False, device=device, dtype=torch.float32)
    model.w_int = torch.tensor(
        0.10, requires_grad=False, device=device, dtype=torch.float32)

    collocation_points = model._initialize_collocation_points(N_f)
    start_time = time.time()

    for epoch in range(stage1_epochs):
        idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                            device=device)
        data_batch = (model.x_data[idx], model.t_data[idx],
                      model.u_data[idx])

        collocation_batch_dict = {}
        n_subs = max(1, len(model.subdomains))
        effective_bs = max(512, N_f_batch_size // n_subs)
        for i, subdomain in enumerate(model.subdomains):
            x_col_full, t_col_full = collocation_points[f'subdomain_{i}']
            n_pts = x_col_full.shape[0]
            if n_pts == 0:
                continue
            bs = min(effective_bs, n_pts)
            col_idx = torch.randint(0, n_pts, (bs,), device=device)
            collocation_batch_dict[f'subdomain_{i}'] = (
                x_col_full[col_idx], t_col_full[col_idx])

        if not collocation_batch_dict:
            continue

        total_loss, losses = model._train_step(
            collocation_batch_dict, data_batch)

        if model._epochs_since_split is not None:
            model._epochs_since_split += 1

        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f'[S1] Epoch {epoch+1}/{stage1_epochs} | '
                  f'Loss: {total_loss.item():.3e} | Time: {elapsed:.2f}s')
            start_time = time.time()

    print(f"\n[S1] Stage 1 complete. Final loss: {total_loss.item():.3e}")

    # ==================================================================
    # STAGE 2: Custom decomposition
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"[S2] STAGE 2: {'Temporal' if method == 'A6_temporal' else 'Spatial+Temporal'} decomposition")
    print(f"{'='*60}")

    model._compute_shock_indicators()
    shock_detected = (
        hasattr(model, '_spatial_shock_indicator')
        and hasattr(model, '_temporal_shock_indicator')
        and (model._spatial_shock_indicator > model.shock_indicator_threshold
             or model._temporal_shock_indicator > model.shock_indicator_threshold)
    )

    coarse_sub = model.subdomains[0]
    coarse_net = coarse_sub['net']
    b = coarse_sub['bounds']

    if method == 'A6_temporal':
        # --- Temporal-only: one temporal split ---
        print("[S2] Computing temporal split position from coarse PINN residuals...")
        t_split = model._find_split_position_t(coarse_sub)
        print(f"[S2] Temporal split at t={t_split:.4f}")

        children = model._create_children_temporal(coarse_sub, t_split)
        model._initialize_children_from_parent(coarse_net, children,
                                               n_init_epochs=200)
        model.subdomains = children
        model._num_splits_done = 1
        model._rebuild_interfaces_from_subdomains()

    elif method == 'A7_spatial_temporal':
        # --- Spatial splits first (same as B7) ---
        print("[S2] Phase 1: Spatial decomposition (same as B7)...")

        # Compute residual-based spatial splits
        x_vals, R_x = model._compute_residual_profile(n_x=200, n_t=100)
        if shock_detected:
            split_positions = model._find_split_positions_from_profile(
                x_vals, R_x, n_target_subdomains=None)
        else:
            split_positions = model._find_split_positions_from_profile(
                x_vals, R_x, n_target_subdomains=2)

        if not split_positions:
            # Fallback: split at midpoint
            x_mid = (b[2] + b[3]) / 2.0
            split_positions = [x_mid]
            print(f"  [Fallback] Single spatial split at x={x_mid:.4f}")

        # Create spatial children
        edges = [b[2]] + split_positions + [b[3]]
        spatial_children = []
        for k in range(len(edges) - 1):
            child_bounds = (b[0], b[1], edges[k], edges[k + 1])
            child_net = model._create_child_net(coarse_net)
            spatial_children.append({
                'bounds': child_bounds, 'net': child_net,
                'level': 1, 'parent_id': None,
            })

        print(f"[S2] Created {len(spatial_children)} spatial subdomains "
              f"from {len(split_positions)} splits")
        model._initialize_children_from_parent(coarse_net, spatial_children,
                                               n_init_epochs=200)
        model.subdomains = spatial_children

        # --- Temporal split on each spatial subdomain ---
        print("\n[S2] Phase 2: Temporal decomposition of each spatial subdomain...")
        all_children = []
        for si, sd in enumerate(model.subdomains):
            t_split = model._find_split_position_t(sd)
            sb = sd['bounds']
            print(f"  Subdomain {si} x=[{sb[2]:.3f},{sb[3]:.3f}]: "
                  f"temporal split at t={t_split:.4f}")
            t_children = model._create_children_temporal(sd, t_split)
            # Initialize temporal children from their spatial parent
            model._initialize_children_from_parent(sd['net'], t_children,
                                                   n_init_epochs=200)
            all_children.extend(t_children)

        model.subdomains = all_children
        model._num_splits_done = len(split_positions) + len(spatial_children)
        model._rebuild_interfaces_from_subdomains()

    print(f"[S2] Final: {len(model.subdomains)} subdomains")

    # Rebuild optimizer for all subnet parameters
    all_params = []
    for sd in model.subdomains:
        all_params.extend(list(sd['net'].parameters()))
    # Include shock_speed variables from spatial interfaces
    for intf in model.interfaces:
        if 'shock_speed' in intf:
            all_params.append(intf['shock_speed'])
    model.optimizer_model = torch.optim.Adam(all_params, lr=1e-4)
    model.scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer_model, step_size=5000, gamma=0.9)
    model._post_split_warmup_remaining = 0

    # ==================================================================
    # STAGE 3: Fine-tune (identical logic to original train_two_stage)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"[S3] STAGE 3: Fine-tuning ({stage3_epochs} epochs)")
    print(f"{'='*60}")

    model.use_rar = orig_use_rar
    model.use_spatial_decomp = False
    model.use_temporal_decomp = True   # area-based collocation allocation
    model.use_adaptive_loss = False

    collocation_points = model._initialize_collocation_points(N_f)
    start_time = time.time()

    for epoch in range(stage3_epochs):
        idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                            device=device)
        data_batch = (model.x_data[idx], model.t_data[idx],
                      model.u_data[idx])

        collocation_batch_dict = {}
        n_subs = max(1, len(model.subdomains))
        effective_bs = max(512, N_f_batch_size // n_subs)
        for i, subdomain in enumerate(model.subdomains):
            x_col_full, t_col_full = collocation_points[f'subdomain_{i}']
            n_pts = x_col_full.shape[0]
            if n_pts == 0:
                continue
            bs = min(effective_bs, n_pts)
            col_idx = torch.randint(0, n_pts, (bs,), device=device)
            collocation_batch_dict[f'subdomain_{i}'] = (
                x_col_full[col_idx], t_col_full[col_idx])

        if not collocation_batch_dict:
            continue

        total_loss, losses = model._train_step(
            collocation_batch_dict, data_batch)

        if model._epochs_since_split is not None:
            model._epochs_since_split += 1

        global_epoch = stage1_epochs + epoch + 1
        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f'[S3] Epoch {epoch+1}/{stage3_epochs} '
                  f'(global {global_epoch}) | '
                  f'Loss: {total_loss.item():.3e} | Time: {elapsed:.2f}s')
            start_time = time.time()

        # RAR in stage 3
        if (model.use_rar
                and (epoch + 1) % adaptive_sampling_freq == 0
                and epoch < stage3_epochs - 1):
            collocation_points = model._adaptive_sampling_step(
                collocation_points, num_new_points=num_new_points)

    print(f"\n[S3] Stage 3 complete. Final loss: {total_loss.item():.3e}")


# ======================================================================
# Run one experiment
# ======================================================================

def run_variant(method, dataset, data_points, seed):
    """Run A6_temporal or A7_spatial_temporal."""
    domain_bounds = get_domain_bounds(dataset)
    norm_params = dataset['norm_params']

    set_seed(seed)
    start_time = time.time()

    # Create B7-identical model
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
        use_temporal_decomp=True,
        shock_indicator_threshold=2.0,
        **norm_params)

    train_custom_two_stage(
        model, method,
        total_epochs=PARAMS['epochs'],
        stage1_epochs=PARAMS['stage1_epochs'],
        batch_size=PARAMS['batch_size'],
        N_f=PARAMS['N_f'],
        adaptive_sampling_freq=2500,
        num_new_points=2500)

    train_time = time.time() - start_time
    pred_mph = predict_full_field(
        model, dataset['X'], dataset['T'],
        dataset['u_min'], dataset['u_max'], device)
    n_params = count_params(model)
    n_subs, splits = get_split_info(model)
    return pred_mph, train_time, n_params, n_subs, ';'.join(splits)


# ======================================================================
# Main
# ======================================================================

def main():
    csv_path = os.path.join(RESULTS_DIR, 'tables',
                            'domain_direction_ablation.csv')
    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Step 1: Extract A5 from existing results
    print("="*70)
    print("Extracting A5_spatial from per_seed_all.csv (B7 rows)")
    print("="*70)
    extract_a5(csv_path)
    print()

    # Step 2: Run A6 and A7
    completed = load_completed(csv_path)

    total_new = (len(NEW_METHODS) * len(DATASET_ORDER)
                 * len(SENSOR_COUNTS) * len(SEEDS))
    run_idx = 0

    print("="*70)
    print(f"Running A6_temporal and A7_spatial_temporal ({total_new} total)")
    print("="*70)

    for ds_name in DATASET_ORDER:
        csv_rel, x_range, t_range = DATASETS[ds_name]
        csv_file = os.path.join(DATA_DIR, csv_rel)
        dataset = load_dataset(csv_file, x_range, t_range)
        n_x = dataset['n_x']

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} | v_f={dataset['v_f_mph']:.2f} mph | "
              f"grid={n_x}x{dataset['n_t']}")
        print(f"{'='*70}")

        for n_sensors in SENSOR_COUNTS:
            sensor_idx = get_sensor_indices(n_x, n_sensors)

            for method in NEW_METHODS:
                for seed in SEEDS:
                    run_idx += 1
                    key = (ds_name, method, n_sensors, seed)
                    if key in completed:
                        continue

                    print(f"\nRun {run_idx}/{total_new}: "
                          f"Dataset={ds_name}, Method={method}, "
                          f"Sensors={n_sensors}, Seed={seed}")

                    data_points, sensor_x_norm = get_sensor_data(
                        dataset, sensor_idx, device)

                    try:
                        pred_mph, train_time, n_params, n_subs, splits = \
                            run_variant(method, dataset, data_points, seed)

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
                              f"Time={train_time:.1f}s, Subs={n_subs}")

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

    # Summary
    print(f"\n{'='*70}")
    print("DOMAIN DIRECTION ABLATION COMPLETE")
    print(f"{'='*70}")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nResults: {csv_path} ({len(df)} rows)")
        summary = df.groupby(['Dataset', 'Method', 'Sensors']).agg(
            L2_mean=('L2', 'mean'),
            L2_std=('L2', 'std'),
            RMSE_mean=('RMSE', 'mean'),
            Subs_mean=('Subdomains', 'mean'),
            Time_mean=('Time', 'mean'),
            N=('L2', 'count'),
        ).round(4)
        print(summary.to_string())


if __name__ == '__main__':
    main()
