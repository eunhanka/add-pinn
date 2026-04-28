"""Utility functions for CA-STD-PINNs experiments.

Data loading, metrics, sensor placement, linear interpolation baseline.
"""

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

FT_PER_S_TO_MPH = 3600.0 / 5280.0


def load_dataset(csv_path, x_range_physical, t_range_physical):
    """Load and preprocess a traffic dataset.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns [t, x, speed] (speed in ft/s, x/t as indices).
    x_range_physical : float
        Physical spatial range in feet.
    t_range_physical : float
        Physical temporal range in seconds.

    Returns
    -------
    dict with keys: data, x_axis, t_axis, X, T, Exact_norm, Exact_mph,
         norm_params, v_f_mph, u_min, u_max, x_min, x_max, t_min, t_max,
         n_x, n_t
    """
    data = pd.read_csv(csv_path, header=0)
    data = data[['t', 'x', 'speed']]
    data['speed'] = data['speed'] * FT_PER_S_TO_MPH  # ft/s -> mph
    data = data.astype(np.float32)

    x_min, x_max = data['x'].min(), data['x'].max()
    t_min, t_max = data['t'].min(), data['t'].max()
    u_min, u_max = data['speed'].min(), data['speed'].max()

    # Free-flow speed: 95th percentile
    v_f = float(np.percentile(data['speed'], 95))

    # Min-max normalization to [0, 1]
    data['x_norm'] = (data['x'] - x_min) / (x_max - x_min)
    data['t_norm'] = (data['t'] - t_min) / (t_max - t_min)
    data['speed_norm'] = (data['speed'] - u_min) / (u_max - u_min)

    x_axis = np.sort(data['x_norm'].unique())
    t_axis = np.sort(data['t_norm'].unique())
    n_x = len(x_axis)
    n_t = len(t_axis)
    X, T = np.meshgrid(x_axis, t_axis)

    Exact_norm = data.pivot_table(
        index='t_norm', columns='x_norm',
        values='speed_norm').values.astype(np.float32)

    # Fill NaN from missing (x,t) grid points via linear interpolation
    nan_count = np.isnan(Exact_norm).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} missing grid points found, "
              f"filling with interpolation")
        from scipy.ndimage import generic_filter
        # Fill NaN: use mean of neighbors, iterate until no NaN
        filled = Exact_norm.copy()
        for _ in range(10):
            if not np.isnan(filled).any():
                break
            mask = np.isnan(filled)
            # For each NaN, average its non-NaN neighbors
            for idx in zip(*np.where(mask)):
                i, j = idx
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < filled.shape[0] and
                            0 <= nj < filled.shape[1] and
                            not np.isnan(filled[ni, nj])):
                        neighbors.append(filled[ni, nj])
                if neighbors:
                    filled[i, j] = np.mean(neighbors)
        Exact_norm = filled
        remaining = np.isnan(Exact_norm).sum()
        if remaining > 0:
            print(f"  WARNING: {remaining} NaN still remain after filling")

    Exact_mph = Exact_norm * (u_max - u_min) + u_min

    # PDE coefficient parameters — use PHYSICAL ranges, not index ranges
    norm_params = {
        'v_f_physical': v_f,
        'u_min_physical': float(u_min),
        'u_max_physical': float(u_max),
        'x_range_physical': float(x_range_physical),
        't_range_physical': float(t_range_physical),
    }

    return {
        'data': data,
        'x_axis': x_axis,
        't_axis': t_axis,
        'n_x': n_x,
        'n_t': n_t,
        'X': X, 'T': T,
        'Exact_norm': Exact_norm,
        'Exact_mph': Exact_mph,
        'norm_params': norm_params,
        'v_f_mph': v_f,
        'u_min': float(u_min), 'u_max': float(u_max),
        'x_min': float(x_min), 'x_max': float(x_max),
        't_min': float(t_min), 't_max': float(t_max),
    }


def get_sensor_indices(n_x, n_sensors):
    """Uniform interior sensor placement."""
    return np.linspace(0, n_x - 1, n_sensors + 2)[1:-1].astype(int)


def get_sensor_data(dataset, sensor_indices, device):
    """Extract training data at sensor locations.

    Returns (data_points_tuple, sensor_x_norm_array).
    """
    x_axis = dataset['x_axis']
    sensor_x_norm = x_axis[sensor_indices]

    mask = dataset['data']['x_norm'].isin(sensor_x_norm)
    sensor_df = dataset['data'][mask]

    x_s = sensor_df[['x_norm']].values
    t_s = sensor_df[['t_norm']].values
    u_s = sensor_df[['speed_norm']].values

    data_points = (
        torch.tensor(x_s, dtype=torch.float32, device=device),
        torch.tensor(t_s, dtype=torch.float32, device=device),
        torch.tensor(u_s, dtype=torch.float32, device=device),
    )
    return data_points, sensor_x_norm


def get_domain_bounds(dataset):
    """Get normalized domain bounds as (lb, ub) arrays."""
    x_axis = dataset['x_axis']
    t_axis = dataset['t_axis']
    lb = np.array([x_axis.min(), t_axis.min()])
    ub = np.array([x_axis.max(), t_axis.max()])
    return (lb, ub)


def compute_metrics(Exact_mph, U_pred_mph):
    """Compute L2 relative error, RMSE, MSE."""
    mse = float(np.mean((Exact_mph - U_pred_mph) ** 2))
    rmse = float(np.sqrt(mse))
    l2 = float(np.linalg.norm(Exact_mph - U_pred_mph)
               / np.linalg.norm(Exact_mph))
    return {'L2': l2, 'RMSE': rmse, 'MSE': mse}


def predict_full_field(model, X, T, u_min, u_max, device, batch_size=8192):
    """Predict on full grid and unnormalize to mph."""
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    predictions = []
    with torch.no_grad():
        for i in range(0, X_flat.shape[0], batch_size):
            x_b = torch.tensor(
                X_flat[i:i + batch_size],
                dtype=torch.float32, device=device)
            t_b = torch.tensor(
                T_flat[i:i + batch_size],
                dtype=torch.float32, device=device)
            u_b = model.predict(x_b, t_b)
            predictions.append(u_b)
    u_pred_norm = torch.cat(predictions, dim=0)
    if u_pred_norm.ndim > 1 and u_pred_norm.shape[1] == 1:
        u_pred_norm = u_pred_norm.squeeze(-1)
    u_pred_mph = (u_pred_norm.detach().cpu().numpy().reshape(X.shape)
                  * (u_max - u_min) + u_min)
    return u_pred_mph


def linear_interpolation(dataset, sensor_indices):
    """B1: Linear interpolation baseline from sensor data.

    Returns prediction in mph on the full grid.
    """
    x_axis = dataset['x_axis']
    t_axis = dataset['t_axis']
    Exact_norm = dataset['Exact_norm']

    # Build (x_norm, t_norm) -> speed_norm from sensor locations
    points = []
    values = []
    for sx_idx in sensor_indices:
        sx = x_axis[sx_idx]
        for j, tv in enumerate(t_axis):
            points.append([sx, tv])
            values.append(Exact_norm[j, sx_idx])
    points = np.array(points, dtype=np.float64)
    values = np.array(values, dtype=np.float64)

    interp = LinearNDInterpolator(points, values)
    nearest = NearestNDInterpolator(points, values)

    X, T = dataset['X'], dataset['T']
    query = np.column_stack([X.flatten(), T.flatten()])

    pred_norm = interp(query)
    nan_mask = np.isnan(pred_norm)
    if nan_mask.any():
        pred_norm[nan_mask] = nearest(query[nan_mask])

    pred_norm = pred_norm.reshape(X.shape).astype(np.float32)
    pred_mph = pred_norm * (dataset['u_max'] - dataset['u_min']) + dataset['u_min']
    return pred_mph


def count_params(model):
    """Count trainable parameters across all subdomains."""
    total = 0
    if hasattr(model, 'subdomains') and model.subdomains:
        for sd in model.subdomains:
            total += sum(p.numel() for p in sd['net'].parameters())
    elif hasattr(model, 'net'):
        total += sum(p.numel() for p in model.net.parameters())
    return total


def get_split_info(model):
    """Get number of subdomains and split positions."""
    if not hasattr(model, 'subdomains'):
        return 1, []
    n_subs = len(model.subdomains)
    splits = []
    for sd in model.subdomains:
        b = sd['bounds']
        splits.append(f"x=[{b[2]:.3f},{b[3]:.3f}]")
    return n_subs, splits
