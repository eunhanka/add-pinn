"""
Ada-STDPINNs v2 — Session 3: R-H / Entropy Conditions & Causal Weighting
=========================================================================
PyTorch conversion of ada_stdpinn_v2_final.py

Modified from ada_stdpinn_v2_session2.py

Session 1 (preserved): PDE coefficients, 2D bounds, interface types, temporal loss
Session 2 (preserved): Anisotropic indicators, split direction, split position

Session 3 Changes (AdaStdpinnLWR only):
  [Task 8]  R-H interface loss + Lax entropy condition at spatial_shock interfaces
  [Task 9]  Causal weighting for PDE loss within each subdomain
  [Task 10] Ablation study flags for systematic component evaluation

Backward compatibility: all ablation flags default to True (full model).
"""

import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyDOE import lhs
import os

# ==============================================================================
# Device setup
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

torch.set_default_dtype(torch.float32)

# ==============================================================================
# 0. Utility Functions
# ==============================================================================

# NEW: Estimate free-flow speed from data
def estimate_free_flow_speed(speed_series, method='percentile_85'):
    """
    Estimate free-flow speed v_f from observed speed data.

    Parameters
    ----------
    speed_series : array-like
        Speed observations in physical units (e.g., mph).
    method : str
        'percentile_85' (default, traffic engineering standard),
        'percentile_95', or 'max'.

    Returns
    -------
    float : Estimated v_f in the same unit as input.
    """
    arr = np.asarray(speed_series)
    if method == 'percentile_85':
        return float(np.percentile(arr, 85))
    elif method == 'percentile_95':
        return float(np.percentile(arr, 95))
    elif method == 'max':
        return float(np.max(arr))
    else:
        raise ValueError(f"Unknown method: {method}")


# NEW: Helper to extract x/t bounds from 2D subdomain bounds
def x_bounds(bounds):
    """Extract (x_left, x_right) from 4-tuple bounds."""
    return bounds[2], bounds[3]


def t_bounds(bounds):
    """Extract (t_left, t_right) from 4-tuple bounds."""
    return bounds[0], bounds[1]


# ==============================================================================
# 1. Neural Network Architecture (unchanged)
# ==============================================================================

class FourierFeatureMapping(nn.Module):
    """ Fourier Feature Mapping layer """
    def __init__(self, input_dims, embedding_size=256, scale=10.0):
        super(FourierFeatureMapping, self).__init__()
        self.input_dims = input_dims
        self.embedding_size = embedding_size
        self.scale = scale
        B = torch.randn(input_dims, embedding_size) * self.scale
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BaseNet(nn.Module):
    """ Base neural network for all PINN models """
    def __init__(self, layers, activation='tanh'):
        super(BaseNet, self).__init__()
        self.fourier_embedding = FourierFeatureMapping(
            input_dims=layers[0], embedding_size=layers[1] // 2)
        self.hidden_layers = nn.ModuleList()
        # First hidden layer: input from Fourier embedding (2 * embedding_size)
        self.hidden_layers.append(nn.Linear(layers[1], layers[1]))
        for i in range(1, len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        else:
            self.activation = torch.tanh

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        embedded_inputs = self.fourier_embedding(inputs)
        temp = embedded_inputs
        for layer in self.hidden_layers:
            temp = self.activation(layer(temp))
        return self.output_layer(temp)


# ==============================================================================
# 2. Benchmark Model Classes
# ==============================================================================

# --- Model 1: Simple Neural Network (unchanged) ---
class SimpleNN:
    """
    Basic NN baseline — data-only loss, no physics.
    """
    def __init__(self, data_points, layers, **kwargs):
        self.x_data, self.t_data, self.u_data = data_points
        self.net = BaseNet(layers, activation='tanh').to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def _loss_fn(self, data_batch):
        x_data_b, t_data_b, u_data_b = data_batch
        u_pred = self.net(x_data_b, t_data_b)
        return torch.mean(torch.square(u_pred - u_data_b))

    def _train_step(self, data_batch):
        self.optimizer.zero_grad()
        loss = self._loss_fn(data_batch)
        loss.backward()
        self.optimizer.step()
        return loss, {'data': loss, 'pde': 0.0, 'int': 0.0}

    def train(self, epochs, batch_size, **kwargs):
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]
        start_time = time.time()

        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])
            loss, _ = self._train_step(data_batch)

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f'Epoch {epoch+1} | Loss: {loss.item():.3e} | Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(loss.item())
                history['data'].append(loss.item())
                history['pde'].append(0)
                history['int'].append(0)
                start_time = time.time()

        self.subdomains = []
        self.interfaces = []
        return history

    def predict(self, x_pred, t_pred):
        with torch.no_grad():
            return self.net(x_pred, t_pred)


# --- Base Physics-Informed Model ---
class BasePinnModel:
    """
    MODIFIED: PDE residual now uses data-derived coefficients instead of
    hardcoded NGSIM values (0.20, 46.64).
    """
    def __init__(self, domain_bounds, data_points, layers, **kwargs):
        self.domain_bounds = domain_bounds
        self.x_data, self.t_data, self.u_data = data_points
        self.layers = layers
        self.u_max = 1.0
        self.rho_jam = 1.0

        # MODIFIED: Store normalization parameters for PDE coefficient computation
        # These are set via kwargs from the main block.
        # v_f_physical: free-flow speed in physical units (mph)
        # u_min_physical / u_max_physical: speed range in physical units (mph)
        # x_range_physical: x_max - x_min in physical units (feet)
        # t_range_physical: t_max - t_min in physical units (seconds)
        self.v_f_physical = kwargs.get('v_f_physical', None)
        self.u_min_physical = kwargs.get('u_min_physical', None)
        self.u_max_physical = kwargs.get('u_max_physical', None)
        self.x_range_physical = kwargs.get('x_range_physical', None)
        self.t_range_physical = kwargs.get('t_range_physical', None)

        # MODIFIED: Compute PDE coefficients for normalized space
        self._compute_pde_coefficients()

    def _compute_pde_coefficients(self):
        """
        MODIFIED: Derive PDE residual coefficients from data normalization.

        The LWR model in speed form (physical coords, speed in ft/s):
            (v_f - 2u) * u_x - u_t = 0

        After min-max normalization (u_hat, x_hat, t_hat in [0,1]) and unit
        conversion (speed in mph, distance in feet, time in seconds):
            A * u_hat_x  -  B * u_hat * u_hat_x  -  u_hat_t = 0

        where:
            MPH_TO_FPS = 5280 / 3600
            C = MPH_TO_FPS * dt / dx
            A = (v_f_mph - 2 * u_min_mph) * C
            B = 2 * delta_u_mph * C
        """
        if self.v_f_physical is None:
            # Fallback: cannot compute coefficients, use placeholder
            print("WARNING: Physical normalization params not provided. "
                  "PDE residual coefficients set to 1.0 (placeholder).")
            self.pde_coeff_A = 1.0
            self.pde_coeff_B = 2.0
            self.pde_norm_factor = 1.0
            return

        MPH_TO_FPS = 5280.0 / 3600.0  # ~1.4667
        delta_u = self.u_max_physical - self.u_min_physical
        C = MPH_TO_FPS * self.t_range_physical / self.x_range_physical

        self.pde_coeff_A = (self.v_f_physical - 2.0 * self.u_min_physical) * C
        self.pde_coeff_B = 2.0 * delta_u * C

        # Fix 1: Normalization factor to keep PDE residual O(1)
        self.pde_norm_factor = math.sqrt(
            self.pde_coeff_A ** 2 + self.pde_coeff_B ** 2 + 1.0)

        # Fix 3: Sanity check for large coefficients
        if abs(self.pde_coeff_A) > 100 or abs(self.pde_coeff_B) > 100:
            print(f"  WARNING: PDE coefficients are very large! "
                  f"A={self.pde_coeff_A:.1f}, B={self.pde_coeff_B:.1f}")
            print(f"  -> Normalization factor = {self.pde_norm_factor:.1f} "
                  f"will be applied to keep PDE residual O(1)")

        print(f"  PDE coefficients: A={self.pde_coeff_A:.6f}, "
              f"B={self.pde_coeff_B:.6f} "
              f"(v_f={self.v_f_physical:.2f} mph, "
              f"delta_u={delta_u:.2f} mph, C={C:.6f})")
        print(f"  PDE normalization factor: {self.pde_norm_factor:.6f}")

    def _u_to_rho(self, u):
        return self.rho_jam * (1.0 - u / self.u_max)

    def _flux_function(self, rho):
        u = self.u_max * (1.0 - rho / self.rho_jam)
        return rho * u

    # Session 3: characteristic speed for entropy conditions
    def _characteristic_speed(self, rho):
        """lambda(rho) = q'(rho) = u_max * (1 - 2*rho/rho_jam)"""
        return self.u_max * (1.0 - 2.0 * rho / self.rho_jam)

    def _pde_residual(self, net, x, t, create_graph=True):
        """
        Normalized LWR residual:
            R = (pde_coeff_A * u_x - pde_coeff_B * u * u_x - u_t) / norm_factor

        The residual is divided by pde_norm_factor = sqrt(A^2 + B^2 + 1)
        to keep PDE loss O(1), comparable to data loss.

        Args:
            create_graph: True for training (backprop through residual),
                          False for evaluation-only (faster, no graph retained).
        """
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.detach().requires_grad_(True)
        u = net(x, t)
        # Compute both gradients in a single backward pass
        u_x, u_t = torch.autograd.grad(
            u.sum(), [x, t], create_graph=create_graph)
        if u_x is None or u_t is None:
            return torch.zeros_like(x)
        raw_residual = (self.pde_coeff_A * u_x
                        - self.pde_coeff_B * u * u_x - u_t)
        return raw_residual / self.pde_norm_factor


# --- Model 2: Vanilla PINN ---
class VanillaPinnLWR(BasePinnModel):
    """
    Basic PINN with fixed loss weights, no domain decomposition.
    MODIFIED: 2D subdomain bounds.
    """
    def __init__(self, domain_bounds, data_points, layers,
                 w_data_init=1.0, w_pde_init=1.0, **kwargs):
        super().__init__(domain_bounds, data_points, layers, **kwargs)
        self.net = BaseNet(layers, activation='tanh').to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        self.w_data = w_data_init
        self.w_pde = w_pde_init

        # MODIFIED: 2D bounds (t_left, t_right, x_left, x_right)
        self.subdomains = [{
            'bounds': (domain_bounds[0][1], domain_bounds[1][1],
                       domain_bounds[0][0], domain_bounds[1][0]),
            'net': self.net
        }]
        self.interfaces = []

    def _loss_fn(self, collocation_points, data_batch):
        x_data_b, t_data_b, u_data_b = data_batch
        x_col, t_col = collocation_points
        u_pred_data = self.net(x_data_b, t_data_b)
        loss_data = torch.mean(torch.square(u_pred_data - u_data_b))
        residual = self._pde_residual(self.net, x_col, t_col)
        loss_pde = torch.mean(torch.square(residual))
        total_loss = self.w_data * loss_data + self.w_pde * loss_pde
        return total_loss, {'data': loss_data, 'pde': loss_pde, 'int': 0.0}

    def _train_step(self, collocation_points, data_batch):
        self.optimizer.zero_grad()
        total_loss, individual_losses = self._loss_fn(
            collocation_points, data_batch)
        total_loss.backward()
        self.optimizer.step()
        return total_loss, individual_losses

    def train(self, epochs, batch_size, N_f, **kwargs):
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]

        lb = self.domain_bounds[0]
        ub = self.domain_bounds[1]
        points = lb + (ub - lb) * lhs(2, N_f)
        x_col = torch.tensor(points[:, 0:1], dtype=torch.float32,
                              device=device)
        t_col = torch.tensor(points[:, 1:2], dtype=torch.float32,
                              device=device)

        start_time = time.time()
        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])
            total_loss, losses = self._train_step((x_col, t_col), data_batch)

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                w_vals = f"W(d,p): {self.w_data:.2f}, {self.w_pde:.2f}"
                print(f'Epoch {epoch+1} | Loss: {total_loss.item():.3e} '
                      f'| {w_vals} | Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(0)
                start_time = time.time()
        return history

    def predict(self, x_pred, t_pred):
        with torch.no_grad():
            return self.net(x_pred, t_pred)


# --- Model 3: Adaptive Weight PINN ---
class AdaptiveWeightPinnLWR(BasePinnModel):
    """
    Adaptive loss weights only, no RAR or decomposition.
    MODIFIED: 2D subdomain bounds.
    """
    def __init__(self, domain_bounds, data_points, layers,
                 w_data_init=10.0, w_pde_init=1.0, **kwargs):
        super().__init__(domain_bounds, data_points, layers, **kwargs)
        self.net = BaseNet(layers, activation='tanh').to(device)

        self.w_data = torch.tensor(w_data_init, requires_grad=True,
                                   device=device, dtype=torch.float32)
        self.w_pde = torch.tensor(w_pde_init, requires_grad=True,
                                  device=device, dtype=torch.float32)

        self.optimizer_model = torch.optim.Adam(self.net.parameters(),
                                                lr=1e-3)
        self.scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)
        self.optimizer_weights = torch.optim.Adam([self.w_data, self.w_pde],
                                                  lr=1e-5)

        # MODIFIED: 2D bounds
        self.subdomains = [{
            'bounds': (domain_bounds[0][1], domain_bounds[1][1],
                       domain_bounds[0][0], domain_bounds[1][0]),
            'net': self.net
        }]
        self.interfaces = []

    def _loss_fn(self, collocation_points, data_batch):
        x_data_b, t_data_b, u_data_b = data_batch
        x_col, t_col = collocation_points
        u_pred_data = self.net(x_data_b, t_data_b)
        loss_data = torch.mean(torch.square(u_pred_data - u_data_b))
        residual = self._pde_residual(self.net, x_col, t_col)
        loss_pde = torch.mean(torch.square(residual))
        total_loss = self.w_data * loss_data + self.w_pde * loss_pde
        return total_loss, {'data': loss_data, 'pde': loss_pde, 'int': 0.0}

    def _train_step(self, collocation_points, data_batch):
        self.optimizer_model.zero_grad()
        self.optimizer_weights.zero_grad()
        total_loss, individual_losses = self._loss_fn(
            collocation_points, data_batch)
        # Compute weights_loss
        loss_data_w = self.w_data * individual_losses['data']
        loss_pde_w = self.w_pde * individual_losses['pde']
        weights_loss = torch.var(torch.stack([loss_data_w, loss_pde_w]),
                                 correction=0)
        # Backward for model params (retain graph for weights backward)
        total_loss.backward(retain_graph=True)
        # Backward for weight params before stepping model optimizer
        # (model step modifies params in-place, invalidating the graph)
        weights_loss.backward()
        # Now step both optimizers
        self.optimizer_model.step()
        self.scheduler_model.step()
        self.optimizer_weights.step()
        with torch.no_grad():
            self.w_data.clamp_(min=0.0)
            self.w_pde.clamp_(min=0.0)
        return total_loss, individual_losses

    def train(self, epochs, batch_size, N_f, **kwargs):
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]

        lb = self.domain_bounds[0]
        ub = self.domain_bounds[1]
        points = lb + (ub - lb) * lhs(2, N_f)
        collocation_points = (
            torch.tensor(points[:, 0:1], dtype=torch.float32, device=device),
            torch.tensor(points[:, 1:2], dtype=torch.float32, device=device))

        start_time = time.time()
        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])
            total_loss, losses = self._train_step(collocation_points,
                                                  data_batch)

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                w_vals = (f"W(d,p): {self.w_data.item():.2f}, "
                          f"{self.w_pde.item():.2f}")
                print(f'Epoch {epoch+1} | Loss: {total_loss.item():.3e} '
                      f'| {w_vals} | Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(losses['int'])
                start_time = time.time()
        return history

    def predict(self, x_pred, t_pred):
        with torch.no_grad():
            return self.net(x_pred, t_pred)


# --- Model 4: RAR-PINN ---
class RARPinnLWR(BasePinnModel):
    """
    RAR sampling only, fixed weights, no decomposition.
    MODIFIED: 2D subdomain bounds.
    """
    def __init__(self, domain_bounds, data_points, layers,
                 w_data_init=1.0, w_pde_init=1.0, **kwargs):
        super().__init__(domain_bounds, data_points, layers, **kwargs)
        self.net = BaseNet(layers, activation='tanh').to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        self.w_data = w_data_init
        self.w_pde = w_pde_init

        # MODIFIED: 2D bounds
        self.subdomains = [{
            'bounds': (domain_bounds[0][1], domain_bounds[1][1],
                       domain_bounds[0][0], domain_bounds[1][0]),
            'net': self.net
        }]
        self.interfaces = []

    def _loss_fn(self, collocation_points, data_batch):
        x_data_b, t_data_b, u_data_b = data_batch
        x_col, t_col = collocation_points
        u_pred_data = self.net(x_data_b, t_data_b)
        loss_data = torch.mean(torch.square(u_pred_data - u_data_b))
        residual = self._pde_residual(self.net, x_col, t_col)
        loss_pde = torch.mean(torch.square(residual))
        total_loss = self.w_data * loss_data + self.w_pde * loss_pde
        return total_loss, {'data': loss_data, 'pde': loss_pde, 'int': 0.0}

    def _train_step(self, collocation_points, data_batch):
        self.optimizer.zero_grad()
        total_loss, individual_losses = self._loss_fn(
            collocation_points, data_batch)
        total_loss.backward()
        self.optimizer.step()
        return total_loss, individual_losses

    def _adaptive_sampling_step(self, collocation_points,
                                num_new_points=2000):
        print("  - Performing adaptive sampling (RAR)...")
        x_col, t_col = collocation_points
        with torch.no_grad():
            x_cand = torch.rand((10000, 1), device=device) * (
                self.domain_bounds[1][0] - self.domain_bounds[0][0]
            ) + self.domain_bounds[0][0]
            t_cand = torch.rand((10000, 1), device=device) * (
                self.domain_bounds[1][1] - self.domain_bounds[0][1]
            ) + self.domain_bounds[0][1]
        x_cand_g = x_cand.detach().requires_grad_(True)
        t_cand_g = t_cand.detach().requires_grad_(True)
        residuals = self._pde_residual(
            self.net, x_cand_g, t_cand_g, create_graph=False).detach()
        _, top_indices = torch.topk(
            torch.squeeze(torch.abs(residuals)), k=num_new_points)
        new_x = x_cand[top_indices]
        new_t = t_cand[top_indices]
        return (torch.cat([x_col, new_x], dim=0),
                torch.cat([t_col, new_t], dim=0))

    def train(self, epochs, batch_size, N_f, adaptive_sampling_freq,
              num_new_points, **kwargs):
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]

        lb = self.domain_bounds[0]
        ub = self.domain_bounds[1]
        points = lb + (ub - lb) * lhs(2, N_f)
        collocation_points = (
            torch.tensor(points[:, 0:1], dtype=torch.float32, device=device),
            torch.tensor(points[:, 1:2], dtype=torch.float32, device=device))

        N_f_batch_size = 2048
        start_time = time.time()
        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])
            x_col_full, t_col_full = collocation_points
            num_points_total = x_col_full.shape[0]
            col_idx = torch.randint(0, num_points_total, (N_f_batch_size,),
                                    device=device)
            collocation_batch = (x_col_full[col_idx], t_col_full[col_idx])
            total_loss, losses = self._train_step(
                collocation_batch, data_batch)

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                w_vals = f"W(d,p): {self.w_data:.2f}, {self.w_pde:.2f}"
                print(f'Epoch {epoch+1} | Loss: {total_loss.item():.3e} '
                      f'| {w_vals} | Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(losses['int'])
                start_time = time.time()

            if ((epoch + 1) % adaptive_sampling_freq == 0
                    and epoch < epochs - 1):
                collocation_points = self._adaptive_sampling_step(
                    collocation_points, num_new_points)
        return history

    def predict(self, x_pred, t_pred):
        with torch.no_grad():
            return self.net(x_pred, t_pred)


# --- Model 5: ADA-PINN (Adaptive Loss + RAR, No Decomposition) ---
class AdaPinnLWR(BasePinnModel):
    """
    Adaptive loss weights + RAR sampling, no decomposition.
    MODIFIED: 2D subdomain bounds.
    """
    def __init__(self, domain_bounds, data_points, layers,
                 w_data_init=10.0, w_pde_init=1.0, **kwargs):
        super().__init__(domain_bounds, data_points, layers, **kwargs)
        self.net = BaseNet(layers, activation='tanh').to(device)

        self.w_data = torch.tensor(w_data_init, requires_grad=True,
                                   device=device, dtype=torch.float32)
        self.w_pde = torch.tensor(w_pde_init, requires_grad=True,
                                  device=device, dtype=torch.float32)

        self.optimizer_model = torch.optim.Adam(self.net.parameters(),
                                                lr=1e-3)
        self.scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)
        self.optimizer_weights = torch.optim.Adam([self.w_data, self.w_pde],
                                                  lr=1e-5)

        # MODIFIED: 2D bounds
        self.subdomains = [{
            'bounds': (domain_bounds[0][1], domain_bounds[1][1],
                       domain_bounds[0][0], domain_bounds[1][0]),
            'net': self.net
        }]
        self.interfaces = []

    def _loss_fn(self, collocation_points, data_batch):
        x_data_b, t_data_b, u_data_b = data_batch
        x_col, t_col = collocation_points
        u_pred_data = self.net(x_data_b, t_data_b)
        loss_data = torch.mean(torch.square(u_pred_data - u_data_b))
        residual = self._pde_residual(self.net, x_col, t_col)
        loss_pde = torch.mean(torch.square(residual))
        total_loss = self.w_data * loss_data + self.w_pde * loss_pde
        return total_loss, {'data': loss_data, 'pde': loss_pde, 'int': 0.0}

    def _train_step(self, collocation_points, data_batch):
        self.optimizer_model.zero_grad()
        self.optimizer_weights.zero_grad()
        total_loss, individual_losses = self._loss_fn(
            collocation_points, data_batch)
        # Compute weights_loss
        loss_data_w = self.w_data * individual_losses['data']
        loss_pde_w = self.w_pde * individual_losses['pde']
        weights_loss = torch.var(torch.stack([loss_data_w, loss_pde_w]),
                                 correction=0)
        # Backward for model params (retain graph for weights backward)
        total_loss.backward(retain_graph=True)
        # Backward for weight params before stepping model optimizer
        # (model step modifies params in-place, invalidating the graph)
        weights_loss.backward()
        # Now step both optimizers
        self.optimizer_model.step()
        self.scheduler_model.step()
        self.optimizer_weights.step()
        with torch.no_grad():
            self.w_data.clamp_(min=0.0)
            self.w_pde.clamp_(min=0.0)
        return total_loss, individual_losses

    def _adaptive_sampling_step(self, collocation_points,
                                num_new_points=2000):
        print("  - Performing adaptive sampling (RAR)...")
        x_col, t_col = collocation_points
        with torch.no_grad():
            x_cand = torch.rand((10000, 1), device=device) * (
                self.domain_bounds[1][0] - self.domain_bounds[0][0]
            ) + self.domain_bounds[0][0]
            t_cand = torch.rand((10000, 1), device=device) * (
                self.domain_bounds[1][1] - self.domain_bounds[0][1]
            ) + self.domain_bounds[0][1]
        x_cand_g = x_cand.detach().requires_grad_(True)
        t_cand_g = t_cand.detach().requires_grad_(True)
        residuals = self._pde_residual(
            self.net, x_cand_g, t_cand_g, create_graph=False).detach()
        _, top_indices = torch.topk(
            torch.squeeze(torch.abs(residuals)), k=num_new_points)
        new_x = x_cand[top_indices]
        new_t = t_cand[top_indices]
        return (torch.cat([x_col, new_x], dim=0),
                torch.cat([t_col, new_t], dim=0))

    def train(self, epochs, batch_size, N_f, adaptive_sampling_freq,
              num_new_points, **kwargs):
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]

        lb = self.domain_bounds[0]
        ub = self.domain_bounds[1]
        points = lb + (ub - lb) * lhs(2, N_f)
        collocation_points = (
            torch.tensor(points[:, 0:1], dtype=torch.float32, device=device),
            torch.tensor(points[:, 1:2], dtype=torch.float32, device=device))

        N_f_batch_size = 2048
        start_time = time.time()
        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])
            x_col_full, t_col_full = collocation_points
            num_points_total = x_col_full.shape[0]
            col_idx = torch.randint(0, num_points_total, (N_f_batch_size,),
                                    device=device)
            collocation_batch = (x_col_full[col_idx], t_col_full[col_idx])
            total_loss, losses = self._train_step(
                collocation_batch, data_batch)

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                w_vals = (f"W(d,p): {self.w_data.item():.2f}, "
                          f"{self.w_pde.item():.2f}")
                print(f'Epoch {epoch+1} | Loss: {total_loss.item():.3e} '
                      f'| {w_vals} | Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(losses['int'])
                start_time = time.time()

            if ((epoch + 1) % adaptive_sampling_freq == 0
                    and epoch < epochs - 1):
                collocation_points = self._adaptive_sampling_step(
                    collocation_points, num_new_points)
        return history

    def predict(self, x_pred, t_pred):
        with torch.no_grad():
            return self.net(x_pred, t_pred)


# ==============================================================================
# Model 6: STDPINN — Sensor-based decomposition, fixed weights
# ==============================================================================
class StdpinnLWR(BasePinnModel):
    """
    Sensor-location-based spatial domain decomposition, fixed weights.

    MODIFIED (Session 1):
      - 2D bounds: (t_left, t_right, x_left, x_right)
      - Interface type classification: spatial_smooth / spatial_shock / temporal
      - Temporal interface loss placeholder
      - use_temporal_decomp flag for backward compat
    """
    def __init__(self, domain_bounds, data_points, layers, sensor_locations,
                 w_data_init=10.0, w_pde_init=1.0, w_int_init=1.0,
                 layers_after_split=None,
                 use_temporal_decomp=False,      # NEW: v2 flag
                 **kwargs):
        super().__init__(domain_bounds, data_points, layers, **kwargs)

        self.w_data = w_data_init
        self.w_pde = w_pde_init
        self.w_int = w_int_init
        # NEW: temporal decomposition flag
        self.use_temporal_decomp = use_temporal_decomp

        self.subdomains = []
        self.interfaces = []

        # --- Spatial decomposition based on sensor locations ---
        print(f"STDPINN: Manual spatial decomposition with "
              f"{len(sensor_locations)} sensors.")

        subdomain_layers = layers_after_split if layers_after_split else self.layers
        if layers_after_split:
            print(f"STDPINN: Subdomain architecture: {subdomain_layers}")

        x_start, x_end = domain_bounds[0][0], domain_bounds[1][0]
        # MODIFIED: time bounds from domain
        t_start, t_end = domain_bounds[0][1], domain_bounds[1][1]

        boundary_points = sorted(list(
            set([x_start] + sensor_locations.tolist() + [x_end])))

        # Create subdomains with 2D bounds
        for i in range(len(boundary_points) - 1):
            x_left = boundary_points[i]
            x_right = boundary_points[i + 1]
            if x_left >= x_right:
                continue
            # MODIFIED: 2D bounds (t_left, t_right, x_left, x_right)
            sd_bounds = (t_start, t_end, x_left, x_right)
            net = BaseNet(subdomain_layers, activation='tanh').to(device)
            self.subdomains.append({
                'bounds': sd_bounds,
                'net': net,
                'level': 0,           # NEW: quadtree level
                'parent_id': None     # NEW: for transfer learning
            })
            print(f"  - Subdomain {len(self.subdomains)-1}: "
                  f"t=[{sd_bounds[0]:.3f},{sd_bounds[1]:.3f}], "
                  f"x=[{sd_bounds[2]:.3f},{sd_bounds[3]:.3f}]")

        # Create interfaces with type classification
        for i in range(len(self.subdomains) - 1):
            # MODIFIED: extract x position from 2D bounds
            interface_x = self.subdomains[i + 1]['bounds'][2]  # x_left of right
            shock_speed_var = torch.tensor(
                0.0, device=device, dtype=torch.float32,
                requires_grad=True)
            # MODIFIED: interface type classification
            self.interfaces.append({
                'position': interface_x,
                'left_idx': i,
                'right_idx': i + 1,
                'type': 'spatial_shock',       # NEW: type classification
                'direction': 'spatial',        # NEW: direction tag
                'shock_speed': shock_speed_var
            })

        # Build optimizer — only network params (not shock_speed vars)
        # shock_speed vars are plain tensors, updated via manual SGD
        net_params = []
        for sd in self.subdomains:
            net_params.extend(list(sd['net'].parameters()))
        self.optimizer_model = torch.optim.Adam(net_params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)

    # NEW: Temporal interface loss
    def _temporal_interface_loss(self, interface):
        """
        NEW: Compute L_cont_t = MSE(u_after(x, t_n) - u_before(x, t_n))
        at a temporal interface t = t_n.

        Causal: only forward continuity (before -> after).
        """
        t_n = interface['position']
        net_before = self.subdomains[interface['left_idx']]['net']
        net_after = self.subdomains[interface['right_idx']]['net']

        # Sample spatial points along the interface
        # Use x range from the "after" subdomain
        bounds_after = self.subdomains[interface['right_idx']]['bounds']
        xl, xr = bounds_after[2], bounds_after[3]
        x_points = torch.rand((200, 1), device=device) * (xr - xl) + xl
        t_points = torch.full((200, 1), t_n, device=device,
                              dtype=torch.float32)

        u_before = net_before(x_points, t_points)
        u_after = net_after(x_points, t_points)

        loss_cont_t = torch.mean(torch.square(u_after - u_before))
        return loss_cont_t

    def _interface_loss(self):
        """
        MODIFIED: Dispatches by interface type.
        - spatial_shock / spatial_smooth: v1-style spatial loss
        - temporal: L_cont_t (NEW)
        """
        if not self.interfaces:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for interface in self.interfaces:
            # NEW: dispatch by type
            if interface.get('direction') == 'temporal':
                total_loss = total_loss + self._temporal_interface_loss(
                    interface)
                continue

            # --- Spatial interface loss (v1 logic preserved) ---
            net_left = self.subdomains[interface['left_idx']]['net']
            net_right = self.subdomains[interface['right_idx']]['net']

            x_interface = torch.full((200, 1), interface['position'],
                                     device=device, dtype=torch.float32)
            t_interface = torch.rand((200, 1), device=device) * (
                self.domain_bounds[1][1] - self.domain_bounds[0][1]
            ) + self.domain_bounds[0][1]

            x_interface = x_interface.detach().requires_grad_(True)
            t_interface_g = t_interface.detach().requires_grad_(True)

            u_left = net_left(x_interface, t_interface_g)
            u_right = net_right(x_interface, t_interface_g)

            u_x_left = torch.autograd.grad(
                u_left.sum(), x_interface, create_graph=True,
                retain_graph=True)[0]
            u_x_right = torch.autograd.grad(
                u_right.sum(), x_interface, create_graph=True,
                retain_graph=True)[0]

            # C0 continuity
            loss_c0 = torch.mean(torch.square(u_left - u_right))
            # C1 continuity
            loss_c1 = (torch.mean(torch.square(u_x_left - u_x_right))
                        if u_x_left is not None and u_x_right is not None
                        else 0.0)
            # R-H condition
            s = interface['shock_speed']
            rho_left = self._u_to_rho(u_left)
            rho_right = self._u_to_rho(u_right)
            flux_left = self._flux_function(rho_left)
            flux_right = self._flux_function(rho_right)
            rh_residual = s * (rho_right - rho_left) - (flux_right - flux_left)
            loss_rh = torch.mean(torch.square(rh_residual))

            total_loss = total_loss + loss_c0 + loss_c1 + loss_rh
        return total_loss

    def _loss_fn(self, collocation_points_dict, data_batch):
        x_data_b, t_data_b, u_data_b = data_batch
        u_pred_data = self.predict(x_data_b, t_data_b)
        loss_data = torch.mean(torch.square(u_pred_data - u_data_b))

        loss_pde = torch.tensor(0.0, device=device, dtype=torch.float32)
        if self.subdomains:
            num_valid = 0
            for i, subdomain in enumerate(self.subdomains):
                key = f'subdomain_{i}'
                if key not in collocation_points_dict:
                    continue
                x_col, t_col = collocation_points_dict[key]
                residual = self._pde_residual(subdomain['net'], x_col, t_col)
                loss_pde = loss_pde + torch.mean(torch.square(residual))
                num_valid += 1
            if num_valid > 0:
                loss_pde = loss_pde / num_valid

        loss_int = self._interface_loss()
        total_loss = (self.w_data * loss_data
                      + self.w_pde * loss_pde
                      + self.w_int * loss_int)
        return total_loss, {'data': loss_data, 'pde': loss_pde,
                            'int': loss_int}

    def _train_step(self, collocation_points_dict, data_batch):
        # MODIFIED: separate network params from plain tensor (shock_speed)
        # to avoid issues — shock_speed vars updated via manual SGD
        self.optimizer_model.zero_grad()

        # Zero grads for shock_speed vars
        for interface in self.interfaces:
            if 'shock_speed' in interface:
                sv = interface['shock_speed']
                if sv.grad is not None:
                    sv.grad.zero_()

        total_loss, individual_losses = self._loss_fn(
            collocation_points_dict, data_batch)
        total_loss.backward()

        # Fix 2: Gradient clipping after decomposition events
        if (hasattr(self, '_epochs_since_split')
                and self._epochs_since_split is not None
                and self._epochs_since_split < 1000):
            torch.nn.utils.clip_grad_norm_(
                self.optimizer_model.param_groups[0]['params'], max_norm=1.0)

        # Apply optimizer to network params only
        self.optimizer_model.step()
        self.scheduler.step()

        # Manual SGD for shock_speed vars (plain tensor)
        with torch.no_grad():
            for interface in self.interfaces:
                if 'shock_speed' in interface:
                    sv = interface['shock_speed']
                    if sv.grad is not None:
                        sv.data -= 1e-3 * sv.grad
                        sv.grad.zero_()

        return total_loss, individual_losses

    def _initialize_collocation_points(self, N_f):
        """
        MODIFIED: Uses 2D bounds for collocation point sampling.
        """
        print(" - Initializing collocation points for subdomains...")
        collocation_points = {}
        # Total spatial width for proportional allocation
        total_x_width = self.domain_bounds[1][0] - self.domain_bounds[0][0]
        if total_x_width == 0:
            total_x_width = 1.0

        for i, subdomain in enumerate(self.subdomains):
            b = subdomain['bounds']
            # MODIFIED: extract from 2D bounds
            x_left, x_right = b[2], b[3]
            t_left, t_right = b[0], b[1]
            subdomain_width = x_right - x_left
            num_points = max(2000,
                             int(N_f * (subdomain_width / total_x_width)))
            lb_sub = np.array([x_left, t_left])
            ub_sub = np.array([x_right, t_right])
            points = lb_sub + (ub_sub - lb_sub) * lhs(2, num_points)
            x_col = torch.tensor(points[:, 0:1], dtype=torch.float32,
                                 device=device)
            t_col = torch.tensor(points[:, 1:2], dtype=torch.float32,
                                 device=device)
            collocation_points[f'subdomain_{i}'] = (x_col, t_col)
        return collocation_points

    def train(self, epochs, batch_size, N_f, **kwargs):
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]

        N_f_batch_size = 4096
        collocation_points = self._initialize_collocation_points(N_f)
        start_time = time.time()

        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])

            collocation_batch_dict = {}
            n_subs = max(1, len(self.subdomains))
            effective_bs = max(512, N_f_batch_size // n_subs)
            for i, subdomain in enumerate(self.subdomains):
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

            total_loss, losses = self._train_step(
                collocation_batch_dict, data_batch)

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                w_vals = (f"W(d,p,i): {self.w_data:.2f}, "
                          f"{self.w_pde:.2f}, {self.w_int:.2f}")
                print(f'Epoch {epoch+1} | Loss: {total_loss.item():.3e} '
                      f'| {w_vals} | Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(losses['int'].item())
                start_time = time.time()
        return history

    def predict(self, x_pred, t_pred):
        """
        MODIFIED: Uses 2D bounds for subdomain lookup.
        Memory-optimized: only evaluates each subnet on its subdomain points.
        """
        u_pred = torch.zeros_like(x_pred)
        for i, subdomain in enumerate(self.subdomains):
            b = subdomain['bounds']
            # MODIFIED: extract from 2D bounds
            x_left, x_right = b[2], b[3]
            t_left, t_right = b[0], b[1]
            is_last = (i == len(self.subdomains) - 1)

            if is_last:
                cond_x = (x_pred >= x_left) & (x_pred <= x_right)
            else:
                cond_x = (x_pred >= x_left) & (x_pred < x_right)
            cond_t = (t_pred >= t_left) & (t_pred <= t_right)
            mask = cond_x & cond_t

            if mask.any():
                mask_flat = mask.view(-1)
                x_sub = x_pred[mask_flat]
                t_sub = t_pred[mask_flat]
                u_sub_vals = subdomain['net'](x_sub, t_sub)
                u_full = torch.zeros_like(x_pred)
                u_full[mask_flat] = u_sub_vals
                u_pred = torch.where(mask, u_full, u_pred)
        return u_pred


# ==============================================================================
# Model 7: ADA-STDPINN — Full adaptive model
# ==============================================================================
class AdaStdpinnLWR(BasePinnModel):
    """
    Full model: adaptive weights + RAR + domain decomposition.

    MODIFIED (Session 1):
      - 2D bounds: (t_left, t_right, x_left, x_right)
      - Interface type classification
      - Temporal interface loss
      - use_temporal_decomp flag

    MODIFIED (Session 2):
      - Anisotropic residual computation (eta_t, eta_x, r_k)
      - Split direction decision (temporal / spatial / quadtree)
      - Characteristic-aware split position selection
    """
    def __init__(self, domain_bounds, data_points, layers,
                 w_data_init=10.0, w_pde_init=1.0, w_int_init=1.0,
                 layers_after_split=None,
                 # Session 2 params
                 use_temporal_decomp=False,
                 gamma_upper=2.0, gamma_lower=0.5,
                 delta_x_min=0.15, delta_t_min=0.2,
                 max_subdomains=8,
                 split_cooldown=2000,
                 # Session 3 — Task 8: R-H / entropy
                 delta_shock=0.1,
                 w_entropy=1.0,
                 # Session 3 — Task 9: causal weighting
                 causal_epsilon=1.0,
                 n_causal_bins=10,
                 # Session 3 — Task 10: ablation flags
                 use_spatial_decomp=True,
                 use_adaptive_loss=True,
                 use_rar=True,
                 use_rh_interface=True,
                 use_entropy=True,
                 use_causal_weighting=True,
                 # Part C: temporal interface weight scaling
                 w_int_temporal_scale=1.0,
                 # Part C: relative decomposition threshold
                 tau_relative=0.1,
                 # Part C: heterogeneity check (CV threshold)
                 heterogeneity_threshold=0.5,
                 # Part C: data-driven shock indicator
                 shock_indicator_threshold=2.0,
                 w_pde_smooth=None,
                 # Part C: viscous PINN (Huang & Agarwal 2023)
                 use_viscous_pde=False,
                 viscous_epsilon=0.1,
                 **kwargs):
        super().__init__(domain_bounds, data_points, layers, **kwargs)

        self.layers_after_split = layers_after_split
        self.use_temporal_decomp = use_temporal_decomp

        # Session 2: anisotropic split parameters
        self.gamma_upper = gamma_upper
        self.gamma_lower = gamma_lower
        self.delta_x_min = delta_x_min
        self.delta_t_min = delta_t_min
        self.max_subdomains = max_subdomains
        self.split_cooldown = split_cooldown

        # Session 3 — Task 8: R-H / entropy
        self.delta_shock = delta_shock
        self.w_entropy = w_entropy

        # Session 3 — Task 9: causal weighting
        self.causal_epsilon = causal_epsilon
        self.n_causal_bins = n_causal_bins

        # Session 3 — Task 10: ablation flags
        self.use_spatial_decomp = use_spatial_decomp
        self.use_adaptive_loss = use_adaptive_loss
        self.use_rar = use_rar
        self.use_rh_interface = use_rh_interface
        self.use_entropy = use_entropy
        self.use_causal_weighting = use_causal_weighting

        # Part C: temporal interface weight scaling
        self.w_int_temporal_scale = w_int_temporal_scale

        # Part C: relative decomposition threshold
        self.tau_relative = tau_relative
        self.baseline_residual = None

        # Part C: heterogeneity check
        self.heterogeneity_threshold = heterogeneity_threshold

        # Part C: data-driven shock indicator
        self.shock_indicator_threshold = shock_indicator_threshold
        self.w_pde_smooth = w_pde_smooth
        self._shock_check_done = False

        # Part C: viscous PINN (Huang & Agarwal 2023)
        self.use_viscous_pde = use_viscous_pde
        self.viscous_epsilon = viscous_epsilon

        # v4: Adaptive split controls
        self._last_split_epoch = -999999  # cooldown tracker
        self._num_splits_done = 0         # progressive threshold tracker
        self._epochs_since_split = None   # gradient clipping (legacy)
        self._post_split_warmup_remaining = 0  # Fix 1: LR warmup counter
        self._pre_split_lr = 1e-4  # LR to restore after warmup

        self.w_data = torch.tensor(w_data_init, requires_grad=True,
                                   device=device, dtype=torch.float32)
        self.w_pde = torch.tensor(w_pde_init, requires_grad=True,
                                  device=device, dtype=torch.float32)
        self.w_int = torch.tensor(w_int_init, requires_grad=True,
                                  device=device, dtype=torch.float32)

        # MODIFIED: 2D bounds for initial single subdomain
        initial_bounds = (domain_bounds[0][1], domain_bounds[1][1],
                          domain_bounds[0][0], domain_bounds[1][0])
        self.subdomains = [{
            'bounds': initial_bounds,
            'net': BaseNet(self.layers, activation='tanh').to(device),
            'level': 0,           # NEW: quadtree level
            'parent_id': None     # NEW: for transfer learning
        }]
        self.interfaces = []

        net_params = list(self.subdomains[0]['net'].parameters())
        self.optimizer_model = torch.optim.Adam(net_params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)
        self.optimizer_weights = torch.optim.Adam(
            [self.w_data, self.w_pde, self.w_int], lr=1e-5)

    # ------------------------------------------------------------------
    # Session 2 — Task 5: Anisotropic residual indicators
    # ------------------------------------------------------------------
    def _compute_anisotropic_indicators(self, subdomain):
        """
        Compute directional indicators eta_t, eta_x and anisotropy ratio r_k.

        eta_t = |int_{x_L}^{x_R} [rho(x,t_T) - rho(x,t_B)] dx|
        eta_x = |int_{t_B}^{t_T} [q(rho(x_R,t)) - q(rho(x_L,t))] dt|
        r_k   = (eta_t / dx) / (eta_x / dt)
        """
        b = subdomain['bounds']
        t_left, t_right = b[0], b[1]
        x_left, x_right = b[2], b[3]
        net = subdomain['net']
        delta_x = x_right - x_left
        delta_t = t_right - t_left
        N_quad = 100

        with torch.no_grad():
            # --- eta_t: density imbalance across temporal faces ---
            x_pts = torch.linspace(x_left, x_right, N_quad,
                                   device=device).reshape(-1, 1)
            t_top = torch.full([N_quad, 1], t_right, device=device,
                               dtype=torch.float32)
            t_bot = torch.full([N_quad, 1], t_left, device=device,
                               dtype=torch.float32)
            u_top = net(x_pts, t_top)
            u_bot = net(x_pts, t_bot)
            rho_top = self._u_to_rho(u_top)
            rho_bot = self._u_to_rho(u_bot)
            eta_t = torch.abs(torch.mean(rho_top - rho_bot) * delta_x)

            # --- eta_x: flux imbalance across spatial faces ---
            t_pts = torch.linspace(t_left, t_right, N_quad,
                                   device=device).reshape(-1, 1)
            x_right_pts = torch.full([N_quad, 1], x_right, device=device,
                                     dtype=torch.float32)
            x_left_pts = torch.full([N_quad, 1], x_left, device=device,
                                    dtype=torch.float32)
            u_r = net(x_right_pts, t_pts)
            u_l = net(x_left_pts, t_pts)
            q_r = self._flux_function(self._u_to_rho(u_r))
            q_l = self._flux_function(self._u_to_rho(u_l))
            eta_x = torch.abs(torch.mean(q_r - q_l) * delta_t)

            # --- anisotropy ratio ---
            eps = 1e-10
            r_k = (eta_t / (delta_x + eps)) / (eta_x / (delta_t + eps) + eps)

        return float(eta_t.item()), float(eta_x.item()), float(r_k.item())

    # ------------------------------------------------------------------
    # Session 2 — Task 7: Characteristic-aware split positions
    # ------------------------------------------------------------------
    def _find_split_position_x(self, subdomain):
        """x* = argmax_x int_{t_B}^{t_T} |R(x,t)| dt"""
        b = subdomain['bounds']
        t_left, t_right = b[0], b[1]
        x_left, x_right = b[2], b[3]
        N_x, N_t = 50, 50
        x_vals = torch.linspace(x_left, x_right, N_x, device=device)
        t_vals = torch.linspace(t_left, t_right, N_t, device=device)
        x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing='xy')
        x_flat = x_grid.reshape(-1, 1)
        t_flat = t_grid.reshape(-1, 1)
        residuals = self._pde_residual(
            subdomain['net'], x_flat, t_flat, create_graph=False).detach()
        res_grid = torch.abs(residuals).reshape(N_t, N_x)
        int_over_t = torch.mean(res_grid, dim=0)
        margin = max(int(N_x * 0.2), 1)
        if margin < N_x // 2:
            search = int_over_t[margin:N_x - margin]
            idx = torch.argmax(search).item() + margin
        else:
            idx = torch.argmax(int_over_t).item()
        x_split = float(x_vals[idx].item())
        x_margin = (x_right - x_left) * 0.2
        return float(np.clip(x_split, x_left + x_margin, x_right - x_margin))

    def _find_split_position_t(self, subdomain):
        """t* = argmax_t int_{x_L}^{x_R} |R(x,t)| dx"""
        b = subdomain['bounds']
        t_left, t_right = b[0], b[1]
        x_left, x_right = b[2], b[3]
        N_x, N_t = 50, 50
        x_vals = torch.linspace(x_left, x_right, N_x, device=device)
        t_vals = torch.linspace(t_left, t_right, N_t, device=device)
        x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing='xy')
        x_flat = x_grid.reshape(-1, 1)
        t_flat = t_grid.reshape(-1, 1)
        residuals = self._pde_residual(
            subdomain['net'], x_flat, t_flat, create_graph=False).detach()
        res_grid = torch.abs(residuals).reshape(N_t, N_x)
        int_over_x = torch.mean(res_grid, dim=1)
        margin = max(int(N_t * 0.2), 1)
        if margin < N_t // 2:
            search = int_over_x[margin:N_t - margin]
            idx = torch.argmax(search).item() + margin
        else:
            idx = torch.argmax(int_over_x).item()
        t_split = float(t_vals[idx].item())
        t_margin = (t_right - t_left) * 0.2
        return float(np.clip(t_split, t_left + t_margin, t_right - t_margin))

    # ------------------------------------------------------------------
    # Session 2 — Task 6: Child creation helpers
    # ------------------------------------------------------------------
    def _create_child_net(self, parent_net):
        """Create a child network with weight transfer from parent."""
        new_layers = self.layers_after_split if self.layers_after_split else self.layers
        net = BaseNet(new_layers, activation='tanh').to(device)
        if self.layers_after_split and self.layers != self.layers_after_split:
            n_copy = min(len(parent_net.hidden_layers),
                         len(net.hidden_layers))
            for j in range(n_copy):
                net.hidden_layers[j].load_state_dict(
                    parent_net.hidden_layers[j].state_dict())
            try:
                if (parent_net.output_layer.in_features
                        == net.output_layer.in_features):
                    net.output_layer.load_state_dict(
                        parent_net.output_layer.state_dict())
            except Exception as e:
                print(f"    > Warning: output weight transfer failed: {e}")
        else:
            net.load_state_dict(parent_net.state_dict())
        return net

    def _initialize_children_from_parent(self, parent_net, children,
                                         n_init_epochs=200):
        """Fix 2: Train each child to match parent output in its domain."""
        print(f"    > Initializing {len(children)} children "
              f"({n_init_epochs} epochs each)...")
        for ci, child in enumerate(children):
            b = child['bounds']
            child_net = child['net']
            x_left, x_right = b[2], b[3]
            t_left, t_right = b[0], b[1]

            # Sample points in child's domain
            n_pts = 2000
            x_init = (torch.rand((n_pts, 1), device=device)
                      * (x_right - x_left) + x_left)
            t_init = (torch.rand((n_pts, 1), device=device)
                      * (t_right - t_left) + t_left)

            # Get parent's output (target)
            with torch.no_grad():
                u_target = parent_net(x_init, t_init).detach()

            # Train child to match parent
            opt = torch.optim.Adam(child_net.parameters(), lr=1e-3)
            for ep in range(n_init_epochs):
                opt.zero_grad()
                u_child = child_net(x_init, t_init)
                loss = torch.mean(torch.square(u_child - u_target))
                loss.backward()
                opt.step()

            with torch.no_grad():
                u_final = child_net(x_init, t_init)
                final_mse = torch.mean(
                    torch.square(u_final - u_target)).item()
            print(f"      Child {ci} [{x_left:.3f},{x_right:.3f}]x"
                  f"[{t_left:.3f},{t_right:.3f}]: "
                  f"init_loss={final_mse:.6f}")

    def _create_children_spatial(self, subdomain, x_split):
        """Spatial split -> 2 children."""
        b = subdomain['bounds']
        lv = subdomain.get('level', 0) + 1
        parent = subdomain['net']
        return [
            {'bounds': (b[0], b[1], b[2], x_split),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
            {'bounds': (b[0], b[1], x_split, b[3]),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
        ]

    def _create_children_temporal(self, subdomain, t_split):
        """Temporal split -> 2 children."""
        b = subdomain['bounds']
        lv = subdomain.get('level', 0) + 1
        parent = subdomain['net']
        return [
            {'bounds': (b[0], t_split, b[2], b[3]),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
            {'bounds': (t_split, b[1], b[2], b[3]),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
        ]

    def _create_children_quadtree(self, subdomain, x_split, t_split):
        """Quadtree split -> 4 children."""
        b = subdomain['bounds']
        lv = subdomain.get('level', 0) + 1
        parent = subdomain['net']
        return [
            {'bounds': (b[0], t_split, b[2], x_split),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
            {'bounds': (b[0], t_split, x_split, b[3]),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
            {'bounds': (t_split, b[1], b[2], x_split),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
            {'bounds': (t_split, b[1], x_split, b[3]),
             'net': self._create_child_net(parent),
             'level': lv, 'parent_id': None},
        ]

    # ------------------------------------------------------------------
    # Session 2 — Interface rebuild for arbitrary splits
    # ------------------------------------------------------------------
    def _rebuild_interfaces_from_subdomains(self):
        """Detect adjacent subdomains and create spatial/temporal interfaces."""
        self.interfaces = []
        n = len(self.subdomains)
        tol = 1e-6
        idx_counter = 0
        for i in range(n):
            bi = self.subdomains[i]['bounds']
            for j in range(i + 1, n):
                bj = self.subdomains[j]['bounds']
                # Spatial interface: i.x_right == j.x_left
                if abs(bi[3] - bj[2]) < tol:
                    t_lo = max(bi[0], bj[0])
                    t_hi = min(bi[1], bj[1])
                    if t_hi - t_lo > tol:
                        sv = torch.tensor(
                            0.0, device=device, dtype=torch.float32,
                            requires_grad=True)
                        self.interfaces.append({
                            'position': bi[3], 'left_idx': i,
                            'right_idx': j, 'type': 'spatial_shock',
                            'direction': 'spatial', 'shock_speed': sv})
                        idx_counter += 1
                # Spatial interface: j.x_right == i.x_left
                elif abs(bj[3] - bi[2]) < tol:
                    t_lo = max(bi[0], bj[0])
                    t_hi = min(bi[1], bj[1])
                    if t_hi - t_lo > tol:
                        sv = torch.tensor(
                            0.0, device=device, dtype=torch.float32,
                            requires_grad=True)
                        self.interfaces.append({
                            'position': bi[2], 'left_idx': j,
                            'right_idx': i, 'type': 'spatial_shock',
                            'direction': 'spatial', 'shock_speed': sv})
                        idx_counter += 1
                # Temporal interface: i.t_right == j.t_left
                if abs(bi[1] - bj[0]) < tol:
                    x_lo = max(bi[2], bj[2])
                    x_hi = min(bi[3], bj[3])
                    if x_hi - x_lo > tol:
                        self.interfaces.append({
                            'position': bi[1], 'left_idx': i,
                            'right_idx': j, 'type': 'temporal',
                            'direction': 'temporal'})
                # Temporal interface: j.t_right == i.t_left
                elif abs(bj[1] - bi[0]) < tol:
                    x_lo = max(bi[2], bj[2])
                    x_hi = min(bi[3], bj[3])
                    if x_hi - x_lo > tol:
                        self.interfaces.append({
                            'position': bj[1], 'left_idx': j,
                            'right_idx': i, 'type': 'temporal',
                            'direction': 'temporal'})
        n_s = sum(1 for it in self.interfaces if it['direction'] == 'spatial')
        n_t = sum(1 for it in self.interfaces if it['direction'] == 'temporal')
        print(f"  - Rebuilt interfaces: {n_s} spatial, {n_t} temporal "
              f"({len(self.interfaces)} total)")

    # ------------------------------------------------------------------
    # Two-Stage AMR: Residual profile and split detection
    # ------------------------------------------------------------------
    def _compute_residual_profile(self, n_x=200, n_t=100):
        """Compute 1D spatial residual profile R(x) from the coarse PINN.

        Evaluates PDE residuals on a structured grid, squares them,
        and averages over time to get R(x) = mean_t[R(x,t)^2].

        Returns:
            (x_vals, R_x): tensors of shape (n_x,)
        """
        # Use the first (coarse) subdomain's network and bounds
        b = self.subdomains[0]['bounds']
        t_left, t_right = b[0], b[1]
        x_left, x_right = b[2], b[3]
        net = self.subdomains[0]['net']

        x_vals = torch.linspace(x_left, x_right, n_x, device=device)
        t_vals = torch.linspace(t_left, t_right, n_t, device=device)

        # Create full grid
        x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing='ij')
        x_flat = x_grid.reshape(-1, 1)
        t_flat = t_grid.reshape(-1, 1)

        # Compute residuals in chunks to avoid OOM
        chunk_size = 5000
        all_residuals = []
        for start in range(0, x_flat.shape[0], chunk_size):
            end = min(start + chunk_size, x_flat.shape[0])
            x_chunk = x_flat[start:end].detach().requires_grad_(True)
            t_chunk = t_flat[start:end].detach().requires_grad_(True)
            res = self._pde_residual(net, x_chunk, t_chunk,
                                     create_graph=False).detach()
            all_residuals.append(res)

        residuals = torch.cat(all_residuals, dim=0)
        # Reshape to (n_x, n_t), square, average over time
        res_grid = residuals.reshape(n_x, n_t)
        R_x = torch.mean(torch.square(res_grid), dim=1)  # shape (n_x,)

        print(f"  [Residual Profile] R(x) computed on {n_x}x{n_t} grid, "
              f"max={R_x.max().item():.6f}, mean={R_x.mean().item():.6f}")
        return x_vals, R_x

    def _find_split_positions_from_profile(self, x_vals, R_x,
                                           n_target_subdomains=None):
        """Determine split positions from spatial residual profile R(x).

        Args:
            x_vals: 1D tensor of x positions
            R_x: 1D tensor of residual values at x_vals
            n_target_subdomains: desired number of subdomains (None=auto)

        Returns:
            sorted list of split x-positions
        """
        x_np = x_vals.cpu().numpy()
        R_np = R_x.cpu().numpy()
        n_x = len(x_np)
        x_left, x_right = float(x_np[0]), float(x_np[-1])
        domain_width = x_right - x_left

        # Smooth R(x) for noise robustness (conv kernel size ~5% of points)
        kernel_size = max(3, n_x // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        R_smooth = np.convolve(R_np, kernel, mode='same')

        # Margin: exclude 10% from edges
        margin_idx = max(1, int(n_x * 0.10))

        if n_target_subdomains is None:
            # Auto-detect: find peaks above 30% of max
            R_max = np.max(R_smooth)
            threshold = 0.30 * R_max
            min_dist = max(1, int(n_x * 0.10))

            # Simple peak detection
            peaks = []
            for i in range(margin_idx, n_x - margin_idx):
                if (R_smooth[i] > threshold
                        and R_smooth[i] >= R_smooth[i - 1]
                        and R_smooth[i] >= R_smooth[i + 1]):
                    # Check min distance from existing peaks
                    if not peaks or (i - peaks[-1]) >= min_dist:
                        peaks.append(i)

            n_splits = len(peaks)
            n_splits = min(n_splits, self.max_subdomains - 1)
            if n_splits == 0:
                n_splits = 1
            print(f"  [Split Detection] Auto-detected {len(peaks)} peaks, "
                  f"using {n_splits} splits")
        else:
            n_splits = n_target_subdomains - 1
            if n_splits <= 0:
                return []

        # Find local minima (valleys between peaks) in smoothed profile
        minima = []
        for i in range(margin_idx, n_x - margin_idx):
            if (R_smooth[i] <= R_smooth[i - 1]
                    and R_smooth[i] <= R_smooth[i + 1]):
                minima.append((i, R_smooth[i]))

        if len(minima) >= n_splits:
            # Select the deepest (lowest value) minima
            minima.sort(key=lambda m: m[1])
            selected = sorted([minima[k][0] for k in range(n_splits)])
            split_positions = [float(x_np[idx]) for idx in selected]
        else:
            # Fallback: equal spacing
            print(f"  [Split Detection] Not enough minima ({len(minima)}), "
                  f"using equal spacing")
            split_positions = [
                x_left + (k + 1) * domain_width / (n_splits + 1)
                for k in range(n_splits)
            ]

        # Validate: enforce delta_x_min spacing
        validated = []
        for sp in split_positions:
            # Distance from edges
            if sp - x_left < self.delta_x_min:
                sp = x_left + self.delta_x_min
            if x_right - sp < self.delta_x_min:
                sp = x_right - self.delta_x_min
            # Distance from previously accepted splits
            if validated and sp - validated[-1] < self.delta_x_min:
                continue
            validated.append(sp)

        print(f"  [Split Detection] Split positions: "
              f"{[f'{s:.4f}' for s in validated]}")
        return validated

    # ------------------------------------------------------------------
    # Part C — Data-driven shock indicator
    # ------------------------------------------------------------------
    def _compute_shock_indicators(self):
        """Compute shock indicators from observed sensor data.

        Uses speed gradients (not network output) to detect if the data
        contains localized features (shocks) that would benefit from
        domain decomposition. If data is smooth, disables decomposition.
        """
        x_data = self.x_data.cpu().numpy().flatten()
        t_data = self.t_data.cpu().numpy().flatten()
        u_data = self.u_data.cpu().numpy().flatten()

        # Group by unique sensor locations (x values)
        unique_x = np.sort(np.unique(x_data))
        unique_t = np.sort(np.unique(t_data))
        n_sensors = len(unique_x)

        # Build sensor time-series: u[sensor_idx, time_idx]
        sensor_series = {}
        for xi, xv in enumerate(unique_x):
            mask = np.isclose(x_data, xv, atol=1e-7)
            t_vals = t_data[mask]
            u_vals = u_data[mask]
            sort_idx = np.argsort(t_vals)
            sensor_series[xi] = (t_vals[sort_idx], u_vals[sort_idx])

        # Spatial gradients: |du/dx| between consecutive sensors
        spatial_grads = []
        for i in range(n_sensors - 1):
            t1, u1 = sensor_series[i]
            t2, u2 = sensor_series[i + 1]
            dx = unique_x[i + 1] - unique_x[i]
            if dx < 1e-10:
                continue
            # Find common time steps
            common_t = np.intersect1d(t1, t2)
            if len(common_t) == 0:
                continue
            mask1 = np.isin(t1, common_t)
            mask2 = np.isin(t2, common_t)
            du_dx = np.abs(u1[mask1] - u2[mask2]) / dx
            spatial_grads.append(np.mean(du_dx))

        # Temporal gradients: |du/dt| at each sensor
        temporal_grads = []
        for xi in range(n_sensors):
            t_s, u_s = sensor_series[xi]
            if len(t_s) < 2:
                continue
            dt = np.diff(t_s)
            du = np.abs(np.diff(u_s))
            valid = dt > 1e-10
            if np.any(valid):
                du_dt = du[valid] / dt[valid]
                temporal_grads.append(np.mean(du_dt))

        # Compute shock indicators (max/mean ratio)
        if len(spatial_grads) > 0:
            sg = np.array(spatial_grads)
            spatial_shock = np.max(sg) / (np.mean(sg) + 1e-10)
        else:
            spatial_shock = 0.0

        if len(temporal_grads) > 0:
            tg = np.array(temporal_grads)
            temporal_shock = np.max(tg) / (np.mean(tg) + 1e-10)
        else:
            temporal_shock = 0.0

        # Store for reporting
        self._spatial_shock_indicator = spatial_shock
        self._temporal_shock_indicator = temporal_shock

        decomp_needed = (spatial_shock > self.shock_indicator_threshold
                         or temporal_shock > self.shock_indicator_threshold)

        print(f"\n  [Shock Indicator] spatial={spatial_shock:.3f}, "
              f"temporal={temporal_shock:.3f} "
              f"(threshold={self.shock_indicator_threshold})")

        if decomp_needed:
            print(f"  [Shock Indicator] Data has localized features "
                  f"-> decomposition ENABLED")
        else:
            print(f"  [Shock Indicator] Data is smooth "
                  f"-> decomposition DISABLED")
            self.use_spatial_decomp = False
            self.use_temporal_decomp = False
            if self.w_pde_smooth is not None:
                old_pde = self.w_pde.item()
                self.w_pde = torch.tensor(
                    self.w_pde_smooth, requires_grad=True,
                    device=device, dtype=torch.float32)
                w_data_new = 1.0 - self.w_pde_smooth - self.w_int.item()
                self.w_data = torch.tensor(
                    w_data_new, requires_grad=True,
                    device=device, dtype=torch.float32)
                print(f"  [Shock Indicator] PDE weight: {old_pde:.4f} "
                      f"-> {self.w_pde_smooth:.4f}, "
                      f"data weight -> {w_data_new:.4f}")

    # ------------------------------------------------------------------
    # Session 2 — Spatial-only decomposition (v1 backward compat)
    # ------------------------------------------------------------------
    def _spatial_only_decomposition_step(self, residual_threshold=0.001):
        """Spatial-only decomposition: split only the WORST subdomain."""
        if len(self.subdomains) >= self.max_subdomains:
            print(f"  - Max subdomains ({self.max_subdomains}) reached, "
                  f"skipping split.")
            return False

        # Cooldown check
        if hasattr(self, '_last_split_epoch') and hasattr(self, 'split_cooldown'):
            # Note: current_epoch not passed here, checked in caller
            pass

        # Evaluate ALL subdomains, split only the WORST one
        best_idx = -1
        best_residual = -1.0
        best_split_point = None

        for i, subdomain in enumerate(self.subdomains):
            b = subdomain['bounds']
            x_left, x_right = b[2], b[3]
            t_left, t_right = b[0], b[1]

            if (x_right - x_left) < self.delta_x_min:
                continue

            x_eval = torch.rand((5000, 1), device=device) * (
                x_right - x_left) + x_left
            t_eval = torch.rand((5000, 1), device=device) * (
                t_right - t_left) + t_left
            residuals = self._pde_residual(
                subdomain['net'], x_eval, t_eval,
                create_graph=False).detach()
            abs_residuals = torch.abs(residuals).squeeze()
            mean_residual = torch.mean(
                torch.square(residuals)).item()

            # Heterogeneity check
            mean_abs = torch.mean(abs_residuals).item()
            std_abs = torch.std(abs_residuals).item()
            cv = std_abs / (mean_abs + 1e-10)

            print(f"    SD {i}: residual={mean_residual:.4f}, cv={cv:.3f}")

            if cv < self.heterogeneity_threshold:
                print(f"    SD {i}: cv too low, skip")
                continue

            if mean_residual > residual_threshold and mean_residual > best_residual:
                with torch.no_grad():
                    max_idx = torch.argmax(
                        torch.squeeze(torch.abs(residuals)))
                    split_point = x_eval[max_idx][0].item()
                margin = (x_right - x_left) * 0.2
                split_point = np.clip(split_point,
                                      x_left + margin, x_right - margin)
                best_idx = i
                best_residual = mean_residual
                best_split_point = split_point

        if best_idx < 0:
            print("  - No subdomain exceeds threshold.")
            return False

        # Split only the worst subdomain
        subdomain = self.subdomains[best_idx]
        b = subdomain['bounds']
        x_left, x_right = b[2], b[3]
        t_left, t_right = b[0], b[1]
        print(f"    > SPATIAL split SD {best_idx} at x={best_split_point:.4f} "
              f"(residual: {best_residual:.4f})")

        parent_net = subdomain['net']
        children = self._create_children_spatial(subdomain, best_split_point)

        # Fix 2: Train children to match parent output
        self._initialize_children_from_parent(
            parent_net, children, n_init_epochs=200)

        # Replace split subdomain with children
        new_subdomains = []
        for i, sd in enumerate(self.subdomains):
            if i == best_idx:
                new_subdomains.extend(children)
            else:
                new_subdomains.append(sd)

        self.subdomains = sorted(
            new_subdomains, key=lambda sd: sd['bounds'][2])

        self.interfaces = []
        for i in range(len(self.subdomains) - 1):
            interface_x = self.subdomains[i + 1]['bounds'][2]
            shock_speed_var = torch.tensor(
                0.0, device=device, dtype=torch.float32,
                requires_grad=True)
            self.interfaces.append({
                'position': interface_x,
                'left_idx': i, 'right_idx': i + 1,
                'type': 'spatial_shock', 'direction': 'spatial',
                'shock_speed': shock_speed_var
            })

        self._num_splits_done += 1
        n_intf = len(self.interfaces)
        print(f"  - Now: {len(self.subdomains)} subdomains, {n_intf} interfaces")
        print("  - Rebuilding optimizer with LR warmup (1e-5 -> 1e-4)...")
        net_params = []
        for sd in self.subdomains:
            net_params.extend(list(sd['net'].parameters()))
        self.optimizer_model = torch.optim.Adam(net_params, lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)
        self._post_split_warmup_remaining = 500
        self._pre_split_lr = 1e-4
        print("  - Preserving loss weights and optimizer state.")
        return True

    # ------------------------------------------------------------------
    # Session 2 — Anisotropic decomposition (use_temporal_decomp=True)
    # ------------------------------------------------------------------
    def _adaptive_domain_decomposition_step(self, residual_threshold=0.001,
                                                current_epoch=None):
        """
        Dispatches to v1 spatial-only or v2 anisotropic logic.
        Session 3: gated by use_spatial_decomp / use_temporal_decomp flags.
        v4: cooldown, progressive threshold, interface cost awareness.
        """
        print("Performing adaptive domain decomposition (PDD)...")
        if not self.use_spatial_decomp and not self.use_temporal_decomp:
            print("  - All decomp flags disabled, skipping.")
            return False
        if len(self.subdomains) >= self.max_subdomains:
            print(f"  - Max subdomains ({self.max_subdomains}) reached, "
                  f"skipping split.")
            return False

        # v4 Fix 1: Cooldown check
        if current_epoch is not None:
            epochs_since = current_epoch - self._last_split_epoch
            if epochs_since < self.split_cooldown:
                print(f"  - Cooldown: {epochs_since}/{self.split_cooldown} "
                      f"epochs since last split, skipping.")
                return False

        # v4 Fix 3: Interface cost awareness
        n_ifaces = len(self.interfaces)
        n_subs = len(self.subdomains)
        if n_ifaces >= 2 * n_subs:
            print(f"  - Interface cost: {n_ifaces} interfaces >= "
                  f"2 * {n_subs} subdomains, skipping split.")
            return False

        # v4 Fix 2: Progressive threshold (relative to baseline if available)
        if self.baseline_residual is not None and self.baseline_residual > 0:
            progressive_threshold = self.baseline_residual * self.tau_relative * (
                1.3 ** self._num_splits_done)
            print(f"  - Relative threshold: {self.baseline_residual:.6f} * "
                  f"{self.tau_relative} * 1.3^{self._num_splits_done} = "
                  f"{progressive_threshold:.6f}")
        else:
            progressive_threshold = residual_threshold * (
                1.3 ** self._num_splits_done)
            print(f"  - Fallback threshold: {residual_threshold:.4f} * "
                  f"1.3^{self._num_splits_done} = {progressive_threshold:.4f}")

        if not self.use_temporal_decomp:
            if self.use_spatial_decomp:
                return self._spatial_only_decomposition_step(
                    progressive_threshold)
            return False

        # --- v5: Evaluate ALL subdomains, split only the WORST one ---
        best_idx = -1
        best_total = -1.0
        best_info = None  # (eta_t, eta_x, r_k)

        for i, subdomain in enumerate(self.subdomains):
            b = subdomain['bounds']
            delta_x = b[3] - b[2]
            delta_t = b[1] - b[0]

            if delta_x < self.delta_x_min and delta_t < self.delta_t_min:
                continue

            eta_t, eta_x, r_k = self._compute_anisotropic_indicators(subdomain)
            total_ind = eta_t + eta_x

            # Part C: Heterogeneity check for anisotropic path
            x_eval = torch.rand((5000, 1), device=device) * delta_x + b[2]
            t_eval = torch.rand((5000, 1), device=device) * delta_t + b[0]
            res = self._pde_residual(
                subdomain['net'], x_eval, t_eval,
                create_graph=False).detach()
            abs_res = torch.abs(res).squeeze()
            mean_abs = torch.mean(abs_res).item()
            std_abs = torch.std(abs_res).item()
            cv = std_abs / (mean_abs + 1e-10)

            print(f"    SD {i}: eta_t={eta_t:.4f}, eta_x={eta_x:.4f}, "
                  f"r_k={r_k:.4f}, total={total_ind:.4f}, cv={cv:.3f}")

            if cv < self.heterogeneity_threshold:
                print(f"    SD {i}: cv={cv:.3f} < {self.heterogeneity_threshold}, "
                      f"residual is uniform, skip")
                continue

            if total_ind > progressive_threshold and total_ind > best_total:
                best_idx = i
                best_total = total_ind
                best_info = (eta_t, eta_x, r_k)

        if best_idx < 0:
            print("  - No subdomain exceeds progressive threshold.")
            return False

        # Split only the worst subdomain
        subdomain = self.subdomains[best_idx]
        eta_t, eta_x, r_k = best_info
        b = subdomain['bounds']
        delta_x = b[3] - b[2]
        delta_t = b[1] - b[0]

        can_split_x = (self.use_spatial_decomp
                       and delta_x >= 2 * self.delta_x_min)
        can_split_t = delta_t >= 2 * self.delta_t_min

        children = None
        if r_k > self.gamma_upper and can_split_t:
            t_split = self._find_split_position_t(subdomain)
            print(f"    > TEMPORAL split SD {best_idx} at t={t_split:.4f} "
                  f"(r_k={r_k:.2f} > {self.gamma_upper})")
            children = self._create_children_temporal(subdomain, t_split)
        elif r_k < self.gamma_lower and can_split_x:
            x_split = self._find_split_position_x(subdomain)
            print(f"    > SPATIAL split SD {best_idx} at x={x_split:.4f} "
                  f"(r_k={r_k:.2f} < {self.gamma_lower})")
            children = self._create_children_spatial(subdomain, x_split)
        elif can_split_x and can_split_t:
            x_split = self._find_split_position_x(subdomain)
            t_split = self._find_split_position_t(subdomain)
            print(f"    > QUADTREE split SD {best_idx} at x={x_split:.4f}, "
                  f"t={t_split:.4f} (r_k={r_k:.2f})")
            children = self._create_children_quadtree(
                subdomain, x_split, t_split)
        elif can_split_x:
            x_split = self._find_split_position_x(subdomain)
            print(f"    > SPATIAL split (fallback) SD {best_idx} "
                  f"at x={x_split:.4f}")
            children = self._create_children_spatial(subdomain, x_split)
        elif delta_t >= 2 * self.delta_t_min:
            t_split = self._find_split_position_t(subdomain)
            print(f"    > TEMPORAL split (fallback) SD {best_idx} "
                  f"at t={t_split:.4f}")
            children = self._create_children_temporal(subdomain, t_split)

        if children is None:
            print(f"  - Worst SD {best_idx} cannot be split "
                  f"(too small in both dims).")
            return False

        # Fix 2: Train children to match parent output
        self._initialize_children_from_parent(
            subdomain['net'], children, n_init_epochs=200)

        # Replace the split subdomain with its children
        new_subdomains = []
        for i, sd in enumerate(self.subdomains):
            if i == best_idx:
                new_subdomains.extend(children)
            else:
                new_subdomains.append(sd)

        self.subdomains = new_subdomains
        self._rebuild_interfaces_from_subdomains()
        self._num_splits_done += 1
        n_sp = sum(1 for it in self.interfaces
                   if it.get('direction') == 'spatial')
        n_tp = sum(1 for it in self.interfaces
                   if it.get('direction') == 'temporal')
        print(f"  - Now: {len(self.subdomains)} subdomains, "
              f"{n_sp} spatial + {n_tp} temporal = "
              f"{len(self.interfaces)} interfaces")
        print("  - Rebuilding optimizer with LR warmup (1e-5 -> 1e-4)...")
        net_params = []
        for sd in self.subdomains:
            net_params.extend(list(sd['net'].parameters()))
        self.optimizer_model = torch.optim.Adam(net_params, lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)
        self._post_split_warmup_remaining = 500
        self._pre_split_lr = 1e-4
        print("  - Preserving loss weights and optimizer state.")
        return True

    # Temporal interface loss (Session 2: use overlapping x bounds)
    def _temporal_interface_loss(self, interface):
        """L_cont_t = MSE(u_after(x, t_n) - u_before(x, t_n))"""
        t_n = interface['position']
        net_before = self.subdomains[interface['left_idx']]['net']
        net_after = self.subdomains[interface['right_idx']]['net']

        # Session 2: use overlapping x range of the two subdomains
        b_before = self.subdomains[interface['left_idx']]['bounds']
        b_after = self.subdomains[interface['right_idx']]['bounds']
        xl = max(b_before[2], b_after[2])
        xr = min(b_before[3], b_after[3])
        x_points = torch.rand((200, 1), device=device) * (xr - xl) + xl
        t_points = torch.full((200, 1), t_n, device=device,
                              dtype=torch.float32)

        u_before = net_before(x_points, t_points)
        u_after = net_after(x_points, t_points)
        return torch.mean(torch.square(u_after - u_before))

    def _interface_loss(self):
        """
        Session 3: dispatches by direction, auto-detects shock vs smooth,
        applies R-H + entropy when shock (and flags enabled).
        """
        if not self.interfaces:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for interface in self.interfaces:
            if interface.get('direction') == 'temporal':
                total_loss = total_loss + self.w_int_temporal_scale * self._temporal_interface_loss(
                    interface)
                continue

            # --- Spatial interface ---
            net_left = self.subdomains[interface['left_idx']]['net']
            net_right = self.subdomains[interface['right_idx']]['net']
            x_interface = torch.full((200, 1), interface['position'],
                                     device=device, dtype=torch.float32)

            b_l = self.subdomains[interface['left_idx']]['bounds']
            b_r = self.subdomains[interface['right_idx']]['bounds']
            t_lo = max(b_l[0], b_r[0])
            t_hi = min(b_l[1], b_r[1])
            t_interface = torch.rand((200, 1), device=device) * (
                t_hi - t_lo) + t_lo

            x_interface = x_interface.detach().requires_grad_(True)
            t_interface_g = t_interface.detach().requires_grad_(True)

            u_left = net_left(x_interface, t_interface_g)
            u_right = net_right(x_interface, t_interface_g)

            u_x_left = torch.autograd.grad(
                u_left.sum(), x_interface, create_graph=True,
                retain_graph=True)[0]
            u_x_right = torch.autograd.grad(
                u_right.sum(), x_interface, create_graph=True,
                retain_graph=True)[0]

            rho_left = self._u_to_rho(u_left)
            rho_right = self._u_to_rho(u_right)

            # Auto-detect shock: |rho_L - rho_R| > delta_shock
            mean_jump = torch.mean(torch.abs(rho_left - rho_right))

            # C0 + C1 continuity loss (smooth interface)
            loss_c0 = torch.mean(torch.square(u_left - u_right))
            loss_c1 = (torch.mean(torch.square(u_x_left - u_x_right))
                        if u_x_left is not None and u_x_right is not None
                        else torch.tensor(0.0, device=device))
            loss_smooth = loss_c0 + loss_c1

            # R-H + entropy loss (shock interface)
            s = interface['shock_speed']
            flux_left = self._flux_function(rho_left)
            flux_right = self._flux_function(rho_right)
            rh_res = (s * (rho_left - rho_right)
                      - (flux_left - flux_right))
            loss_rh = torch.mean(torch.square(rh_res))

            loss_shock = loss_rh
            if self.use_entropy:
                lambda_L = self._characteristic_speed(rho_left)
                lambda_R = self._characteristic_speed(rho_right)
                loss_ent = torch.mean(
                    torch.square(torch.relu(s - lambda_L))
                    + torch.square(torch.relu(lambda_R - s)))
                loss_shock = loss_shock + self.w_entropy * loss_ent

            # Select based on auto-detect and use_rh_interface flag
            if self.use_rh_interface:
                loss_spatial = torch.where(
                    mean_jump > self.delta_shock,
                    loss_shock, loss_smooth)
            else:
                loss_spatial = loss_smooth

            total_loss = total_loss + loss_spatial
        return total_loss

    # ------------------------------------------------------------------
    # Session 3 — Task 9: Causal PDE loss
    # ------------------------------------------------------------------
    def _causal_pde_loss(self, net, x_col, t_col):
        """
        PDE loss with causal weighting (Wang et al. 2024).
        w_i = exp(-epsilon * sum_{j<i} L_pde(t_j)), weights detached.
        """
        residuals = self._pde_residual(net, x_col, t_col)
        res_sq = torch.squeeze(torch.square(residuals))

        sort_idx = torch.argsort(torch.squeeze(t_col))
        res_sq_sorted = res_sq[sort_idx]

        n = res_sq_sorted.shape[0]
        n_bins = self.n_causal_bins
        bin_size = n // n_bins

        bin_losses = []
        for j in range(n_bins):
            start = j * bin_size
            end = (j + 1) * bin_size if j < n_bins - 1 else n
            bin_losses.append(torch.mean(res_sq_sorted[start:end]))

        # Compute causal weights (detach: weights are constants)
        # Clamp cumulative to prevent exp underflow (exp(-50) ~ 1.9e-22)
        weights = []
        cumulative = torch.tensor(0.0, device=device, dtype=torch.float32)
        clamp_val = torch.tensor(50.0, device=device, dtype=torch.float32)
        for j in range(n_bins):
            w = torch.exp(-self.causal_epsilon * torch.minimum(
                cumulative, clamp_val))
            weights.append(w.detach())
            cumulative = cumulative + bin_losses[j].detach()

        weighted_loss = sum(w * l for w, l in zip(weights, bin_losses))
        total_weight = sum(weights)
        return weighted_loss / (total_weight + 1e-10)

    # ------------------------------------------------------------------
    # Part C: Viscous PDE residual (Huang & Agarwal 2023)
    # ------------------------------------------------------------------
    def _pde_residual_viscous(self, net, x, t, create_graph=True):
        """
        Artificial viscosity PDE residual:
            R_viscous = R_original - epsilon_norm * u_xx
        where epsilon_norm = epsilon * t_range / x_range^2
        u_xx computed via two sequential autograd calls.
        """
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.detach().requires_grad_(True)
        u = net(x, t)
        u_x, u_t = torch.autograd.grad(
            u.sum(), [x, t], create_graph=True)
        if u_x is None or u_t is None:
            return torch.zeros_like(x)
        # Second derivative u_xx
        u_xx = torch.autograd.grad(
            u_x.sum(), x, create_graph=create_graph)[0]
        if u_xx is None:
            u_xx = torch.zeros_like(x)
        raw_residual = (self.pde_coeff_A * u_x
                        - self.pde_coeff_B * u * u_x - u_t)
        x_range = getattr(self, 'x_range_physical', 1.0) or 1.0
        t_range = getattr(self, 't_range_physical', 1.0) or 1.0
        epsilon_norm = self.viscous_epsilon * t_range / (x_range ** 2)
        viscous_residual = raw_residual - epsilon_norm * u_xx
        return viscous_residual / self.pde_norm_factor

    def _loss_fn(self, collocation_points_dict, data_batch):
        x_data_b, t_data_b, u_data_b = data_batch
        u_pred_data = self.predict(x_data_b, t_data_b)
        loss_data = torch.mean(torch.square(u_pred_data - u_data_b))

        loss_pde = torch.tensor(0.0, device=device, dtype=torch.float32)
        if self.subdomains:
            num_valid = 0
            for i, subdomain in enumerate(self.subdomains):
                key = f'subdomain_{i}'
                if key not in collocation_points_dict:
                    continue
                x_col, t_col = collocation_points_dict[key]
                # Session 3: causal weighting
                if self.use_causal_weighting:
                    loss_pde = loss_pde + self._causal_pde_loss(
                        subdomain['net'], x_col, t_col)
                elif self.use_viscous_pde:
                    residual = self._pde_residual_viscous(
                        subdomain['net'], x_col, t_col)
                    loss_pde = loss_pde + torch.mean(torch.square(residual))
                else:
                    residual = self._pde_residual(
                        subdomain['net'], x_col, t_col)
                    loss_pde = loss_pde + torch.mean(torch.square(residual))
                num_valid += 1
            if num_valid > 0:
                loss_pde = loss_pde / num_valid

        loss_int = self._interface_loss()
        total_loss = (self.w_data * loss_data
                      + self.w_pde * loss_pde
                      + self.w_int * loss_int)
        return total_loss, {'data': loss_data, 'pde': loss_pde,
                            'int': loss_int}

    def _train_step(self, collocation_points_dict, data_batch):
        # Zero all grads
        self.optimizer_model.zero_grad()
        if self.use_adaptive_loss:
            self.optimizer_weights.zero_grad()
        for interface in self.interfaces:
            if 'shock_speed' in interface:
                sv = interface['shock_speed']
                if sv.grad is not None:
                    sv.grad.zero_()

        total_loss, individual_losses = self._loss_fn(
            collocation_points_dict, data_batch)

        if self.use_adaptive_loss:
            loss_data_w = self.w_data * individual_losses['data']
            loss_pde_w = self.w_pde * individual_losses['pde']
            loss_int_w = self.w_int * individual_losses['int']
            loss_list = [loss_data_w, loss_pde_w]
            if self.interfaces:
                loss_list.append(loss_int_w)
            weights_loss = torch.var(torch.stack(loss_list), correction=0)

        # Both backward passes BEFORE any optimizer step
        # (optimizer.step() modifies params in-place, invalidating the graph)
        total_loss.backward(retain_graph=self.use_adaptive_loss)
        if self.use_adaptive_loss:
            self.optimizer_weights.zero_grad()
            weights_loss.backward()

        # Fix 3: Global gradient clipping (always on, max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(
            self.optimizer_model.param_groups[0]['params'], max_norm=5.0)

        # Now step all optimizers
        self.optimizer_model.step()
        self.scheduler.step()

        # Fix 1: LR warmup — restore LR after warmup period
        if self._post_split_warmup_remaining > 0:
            self._post_split_warmup_remaining -= 1
            if self._post_split_warmup_remaining == 0:
                for pg in self.optimizer_model.param_groups:
                    pg['lr'] = self._pre_split_lr
                print(f"  [LR Warmup] Restored LR to {self._pre_split_lr}")

        # Manual SGD for shock vars
        with torch.no_grad():
            for interface in self.interfaces:
                if 'shock_speed' in interface:
                    sv = interface['shock_speed']
                    if sv.grad is not None:
                        sv.data -= 1e-3 * sv.grad
                        sv.grad.zero_()

        # Session 3: skip weight update when adaptive loss disabled
        if self.use_adaptive_loss:
            self.optimizer_weights.step()
            with torch.no_grad():
                self.w_data.clamp_(min=0.0)
                self.w_pde.clamp_(min=0.0)
                self.w_int.clamp_(min=0.0)

        return total_loss, individual_losses

    def _adaptive_sampling_step(self, collocation_points,
                                num_new_points=2000):
        print("  - Performing adaptive sampling (RAR)...")
        for i, subdomain in enumerate(self.subdomains):
            b = subdomain['bounds']
            # MODIFIED: 2D bounds
            x_left, x_right = b[2], b[3]
            t_left, t_right = b[0], b[1]
            x_cand = torch.rand((5000, 1), device=device) * (
                x_right - x_left) + x_left
            t_cand = torch.rand((5000, 1), device=device) * (
                t_right - t_left) + t_left
            x_cand_g = x_cand.detach().requires_grad_(True)
            t_cand_g = t_cand.detach().requires_grad_(True)
            residuals = self._pde_residual(
                subdomain['net'], x_cand_g, t_cand_g,
                create_graph=False).detach()
            _, top_indices = torch.topk(
                torch.squeeze(torch.abs(residuals)),
                k=num_new_points)
            new_x = x_cand[top_indices]
            new_t = t_cand[top_indices]
            x_col, t_col = collocation_points[f'subdomain_{i}']
            collocation_points[f'subdomain_{i}'] = (
                torch.cat([x_col, new_x], dim=0),
                torch.cat([t_col, new_t], dim=0))
        return collocation_points

    def _initialize_collocation_points(self, N_f):
        """
        Session 2: area-proportional allocation when temporal decomp enabled.
        """
        print("  - Initializing collocation points...")
        collocation_points = {}

        if self.use_temporal_decomp:
            total_area = sum(
                (sd['bounds'][3] - sd['bounds'][2])
                * (sd['bounds'][1] - sd['bounds'][0])
                for sd in self.subdomains)
            if total_area == 0:
                total_area = 1.0
        else:
            total_area = None  # not used
            total_x_width = (self.domain_bounds[1][0]
                             - self.domain_bounds[0][0])
            if total_x_width == 0:
                total_x_width = 1.0

        for i, subdomain in enumerate(self.subdomains):
            b = subdomain['bounds']
            x_left, x_right = b[2], b[3]
            t_left, t_right = b[0], b[1]

            if self.use_temporal_decomp:
                area = (x_right - x_left) * (t_right - t_left)
                num_points = max(2000, int(N_f * (area / total_area)))
            else:
                subdomain_width = x_right - x_left
                num_points = max(2000,
                                 int(N_f * (subdomain_width / total_x_width)))

            lb_sub = np.array([x_left, t_left])
            ub_sub = np.array([x_right, t_right])
            points = lb_sub + (ub_sub - lb_sub) * lhs(2, num_points)
            x_col = torch.tensor(points[:, 0:1], dtype=torch.float32,
                                 device=device)
            t_col = torch.tensor(points[:, 1:2], dtype=torch.float32,
                                 device=device)
            collocation_points[f'subdomain_{i}'] = (x_col, t_col)
        return collocation_points

    def train(self, epochs, batch_size, N_f, adaptive_sampling_freq,
              domain_decomp_freq, num_new_points, residual_threshold,
              decomp_epoch_max=14000):
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]

        N_f_batch_size = 2048
        collocation_points = self._initialize_collocation_points(N_f)
        start_time = time.time()

        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])

            collocation_batch_dict = {}
            n_subs = max(1, len(self.subdomains))
            effective_bs = max(512, N_f_batch_size // n_subs)
            for i, subdomain in enumerate(self.subdomains):
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

            total_loss, losses = self._train_step(
                collocation_batch_dict, data_batch)

            # Fix 2: Increment epochs-since-split counter
            if self._epochs_since_split is not None:
                self._epochs_since_split += 1

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                w_vals = (f"W(d,p,i): {self.w_data.item():.2f}, "
                          f"{self.w_pde.item():.2f}, "
                          f"{self.w_int.item():.2f}")
                print(f'Epoch {epoch+1} | Loss: {total_loss.item():.3e} '
                      f'| {w_vals} | Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(losses['int'].item())
                start_time = time.time()

            # Part C: Measure baseline PDE residual at epoch 1000
            if (epoch + 1) == 1000 and self.baseline_residual is None:
                all_residuals = []
                for key, (x_c, t_c) in collocation_points.items():
                    idx_i = int(key.split('_')[1])
                    if idx_i < len(self.subdomains):
                        net = self.subdomains[idx_i]['net']
                        x_tmp = x_c.detach().requires_grad_(True)
                        t_tmp = t_c.detach().requires_grad_(True)
                        res = self._pde_residual(
                            net, x_tmp, t_tmp, create_graph=False)
                        all_residuals.append(
                            torch.square(res).detach())
                if all_residuals:
                    baseline = torch.mean(torch.cat(all_residuals)).item()
                    self.baseline_residual = baseline
                    print(f"  [Baseline] PDE residual at epoch 1000: "
                          f"{baseline:.6f}")

            # Session 3: RAR gated by use_rar flag
            if (self.use_rar
                    and (epoch + 1) % adaptive_sampling_freq == 0
                    and epoch < epochs - 1):
                collocation_points = self._adaptive_sampling_step(
                    collocation_points, num_new_points=num_new_points)

            # Part C: Data-driven shock indicator check (once, before first decomp)
            if (not self._shock_check_done
                    and (self.use_spatial_decomp or self.use_temporal_decomp)
                    and (epoch + 1) >= 5000):
                self._shock_check_done = True
                self._compute_shock_indicators()

            # Session 3: decomp gated by spatial/temporal flags
            # v4: epoch guards [5000, 14000], cooldown handled inside
            if ((self.use_spatial_decomp or self.use_temporal_decomp)
                    and (epoch + 1) % domain_decomp_freq == 0
                    and (epoch + 1) >= 5000
                    and (epoch + 1) <= decomp_epoch_max
                    and epoch < epochs - 1):
                if self._adaptive_domain_decomposition_step(
                        residual_threshold=residual_threshold,
                        current_epoch=epoch + 1):
                    collocation_points = self._initialize_collocation_points(
                        N_f)
                    self._last_split_epoch = epoch + 1
                    self._epochs_since_split = 0
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return history

    # ------------------------------------------------------------------
    # Two-Stage AMR Training
    # ------------------------------------------------------------------
    def train_two_stage(self, total_epochs, stage1_epochs, batch_size, N_f,
                        adaptive_sampling_freq=2500, num_new_points=2500,
                        residual_threshold=1e-3, n_target_subdomains=None,
                        force_decomp=False, min_subdomains=None,
                        sensor_split=False, stage2_lr=1e-4,
                        child_noise_std=0.0):
        """Two-stage AMR training: coarse PINN → decompose → fine-tune.

        Stage 1: Train coarse global PINN (no decomp, no RAR, fixed weights)
        Stage 2: Analyze residuals, decompose domain, initialize children
        Stage 3: Fine-tune with RAR on decomposed domain

        Args:
            total_epochs: total training epochs (stage1 + stage3)
            stage1_epochs: epochs for coarse PINN training
            batch_size: data batch size
            N_f: number of collocation points
            adaptive_sampling_freq: RAR frequency in stage 3
            num_new_points: points added per RAR step
            residual_threshold: (unused, kept for API compat)
            n_target_subdomains: target subdomains (None=auto from residual)
            force_decomp: if True, decompose even when no shock detected
                          (uses minimal 2-subdomain split)
            min_subdomains: minimum number of subdomains after decomposition;
                           if residual-based splits yield fewer, override with
                           equally spaced splits to reach this minimum
            sensor_split: if True, when no shock detected, split at sensor
                         midpoint positions (like cPINN/B6) instead of using
                         residual-based positions
        """
        history = {'total': [], 'data': [], 'pde': [], 'int': [], 'epochs': []}
        n_data = self.x_data.shape[0]
        N_f_batch_size = 2048
        stage3_epochs = total_epochs - stage1_epochs

        # Save original flags
        orig_use_adaptive_loss = self.use_adaptive_loss
        orig_use_rar = self.use_rar
        orig_use_spatial_decomp = self.use_spatial_decomp
        orig_use_temporal_decomp = self.use_temporal_decomp

        # ==============================================================
        # Stage 1: Coarse PINN
        # ==============================================================
        print(f"\n{'=' * 60}")
        print(f"[S1] STAGE 1: Coarse PINN training ({stage1_epochs} epochs)")
        print(f"{'=' * 60}")

        # Disable all decomp/RAR/adaptive loss
        self.use_adaptive_loss = False
        self.use_rar = False
        self.use_spatial_decomp = False
        self.use_temporal_decomp = False

        # Set fixed weights
        self.w_data = torch.tensor(0.85, requires_grad=False,
                                   device=device, dtype=torch.float32)
        self.w_pde = torch.tensor(0.05, requires_grad=False,
                                  device=device, dtype=torch.float32)
        self.w_int = torch.tensor(0.10, requires_grad=False,
                                  device=device, dtype=torch.float32)

        collocation_points = self._initialize_collocation_points(N_f)
        start_time = time.time()

        for epoch in range(stage1_epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])

            collocation_batch_dict = {}
            n_subs = max(1, len(self.subdomains))
            effective_bs = max(512, N_f_batch_size // n_subs)
            for i, subdomain in enumerate(self.subdomains):
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

            total_loss, losses = self._train_step(
                collocation_batch_dict, data_batch)

            if self._epochs_since_split is not None:
                self._epochs_since_split += 1

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f'[S1] Epoch {epoch+1}/{stage1_epochs} | '
                      f'Loss: {total_loss.item():.3e} | '
                      f'Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(losses['int'].item())
                start_time = time.time()

        print(f"\n[S1] Stage 1 complete. Final loss: {total_loss.item():.3e}")

        # ==============================================================
        # Stage 2: Analyze and Decompose
        # ==============================================================
        print(f"\n{'=' * 60}")
        print(f"[S2] STAGE 2: Residual analysis and domain decomposition")
        print(f"{'=' * 60}")

        # Check if shock exists in data
        self._compute_shock_indicators()
        shock_detected = (
            hasattr(self, '_spatial_shock_indicator')
            and hasattr(self, '_temporal_shock_indicator')
            and (self._spatial_shock_indicator > self.shock_indicator_threshold
                 or self._temporal_shock_indicator > self.shock_indicator_threshold)
        )

        if not shock_detected and not force_decomp:
            print("[S2] No shock detected -> skipping decomposition")
            # Restore spatial decomp to False (no splits needed)
            self.use_spatial_decomp = False
        else:
            if not shock_detected and force_decomp:
                if sensor_split:
                    # Split at sensor midpoints (like cPINN/B6)
                    print("[S2] No shock detected -> sensor-location "
                          "decomposition (sensor_split=True)")
                    sensor_x = sorted(
                        torch.unique(self.x_data).cpu().numpy().tolist())
                    b = self.subdomains[0]['bounds']
                    x_left, x_right = b[2], b[3]

                    # Use sensor midpoints as split positions
                    split_positions = []
                    for i in range(len(sensor_x) - 1):
                        midpoint = (sensor_x[i] + sensor_x[i + 1]) / 2.0
                        if x_left < midpoint < x_right:
                            split_positions.append(midpoint)
                    split_positions = sorted(split_positions)
                    print(f"[S2] Sensor split positions: {len(split_positions)}"
                          f" splits -> {len(split_positions)+1} subdomains")
                else:
                    print("[S2] No shock detected -> minimal decomposition "
                          "(force_decomp=True)")
                    if n_target_subdomains is None:
                        n_target_subdomains = 2
            else:
                print("[S2] Shock detected -> computing residual profile...")

            # For non-sensor-split paths, compute residual-based positions
            if not (not shock_detected and force_decomp and sensor_split):
                # Compute residual profile from coarse PINN
                x_vals, R_x = self._compute_residual_profile(
                    n_x=200, n_t=100)

                # Find split positions
                split_positions = self._find_split_positions_from_profile(
                    x_vals, R_x, n_target_subdomains=n_target_subdomains)

                # Enforce min_subdomains if specified
                if min_subdomains is not None:
                    actual_subs = len(split_positions) + 1
                    if actual_subs < min_subdomains:
                        print(f"[S2] Residual-based splits gave "
                              f"{actual_subs} subdomains, overriding "
                              f"to {min_subdomains}")
                        split_positions = \
                            self._find_split_positions_from_profile(
                                x_vals, R_x,
                                n_target_subdomains=min_subdomains)

            if split_positions:
                coarse_sub = self.subdomains[0]
                b = coarse_sub['bounds']
                coarse_net = coarse_sub['net']

                # Build new subdomains from split positions
                edges = [b[2]] + split_positions + [b[3]]
                children = []
                for k in range(len(edges) - 1):
                    child_bounds = (b[0], b[1], edges[k], edges[k + 1])
                    child_net = self._create_child_net(coarse_net)
                    children.append({
                        'bounds': child_bounds,
                        'net': child_net,
                        'level': 1,
                        'parent_id': None,
                    })

                print(f"[S2] Created {len(children)} subdomains from "
                      f"{len(split_positions)} splits")

                # Initialize children from parent
                self._initialize_children_from_parent(
                    coarse_net, children, n_init_epochs=200)

                # Add noise to child parameters if requested
                if child_noise_std > 0:
                    print(f"[S2] Adding Gaussian noise "
                          f"N(0, {child_noise_std}) to child params")
                    with torch.no_grad():
                        for ch in children:
                            for param in ch['net'].parameters():
                                param.data += (torch.randn_like(param.data)
                                               * child_noise_std)

                # Replace subdomains
                self.subdomains = children
                self._num_splits_done = len(split_positions)

                # Rebuild interfaces
                self._rebuild_interfaces_from_subdomains()
            else:
                print("[S2] No valid split positions found")

        # Rebuild optimizer for all subnet parameters
        all_params = []
        for sd in self.subdomains:
            all_params.extend(list(sd['net'].parameters()))
        self.optimizer_model = torch.optim.Adam(all_params, lr=stage2_lr)
        print(f"[S2] Stage 2 optimizer lr={stage2_lr}")
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)
        self._post_split_warmup_remaining = 0  # no warmup needed

        # ==============================================================
        # Stage 3: Fine-tune
        # ==============================================================
        print(f"\n{'=' * 60}")
        print(f"[S3] STAGE 3: Fine-tuning ({stage3_epochs} epochs)")
        print(f"{'=' * 60}")

        # Restore RAR, keep decomp and adaptive loss off
        self.use_rar = orig_use_rar
        self.use_spatial_decomp = False  # no further splits
        self.use_adaptive_loss = False   # fixed weights

        # Re-initialize collocation points for new subdomain structure
        collocation_points = self._initialize_collocation_points(N_f)
        start_time = time.time()

        for epoch in range(stage3_epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])

            collocation_batch_dict = {}
            n_subs = max(1, len(self.subdomains))
            effective_bs = max(512, N_f_batch_size // n_subs)
            for i, subdomain in enumerate(self.subdomains):
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

            total_loss, losses = self._train_step(
                collocation_batch_dict, data_batch)

            if self._epochs_since_split is not None:
                self._epochs_since_split += 1

            global_epoch = stage1_epochs + epoch + 1
            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f'[S3] Epoch {epoch+1}/{stage3_epochs} '
                      f'(global {global_epoch}) | '
                      f'Loss: {total_loss.item():.3e} | '
                      f'Time: {elapsed:.2f}s')
                history['epochs'].append(global_epoch)
                history['total'].append(total_loss.item())
                history['data'].append(losses['data'].item())
                history['pde'].append(losses['pde'].item())
                history['int'].append(losses['int'].item())
                start_time = time.time()

            # RAR in stage 3
            if (self.use_rar
                    and (epoch + 1) % adaptive_sampling_freq == 0
                    and epoch < stage3_epochs - 1):
                collocation_points = self._adaptive_sampling_step(
                    collocation_points, num_new_points=num_new_points)

        print(f"\n[S3] Stage 3 complete. Final loss: {total_loss.item():.3e}")
        return history

    # ------------------------------------------------------------------
    # cPINN: flux continuity + solution average interface loss
    # ------------------------------------------------------------------
    def _cpinn_interface_loss(self):
        """cPINN interface coupling for normalized LWR.

        Flux F(u) = -A*u + B*u^2/2 from conservation form u_t + F(u)_x = 0.
        Spatial: MSE(F(u_L) - F(u_R)) + MSE(u_k - u_avg).
        Temporal: solution average only (C0 continuity).
        """
        if not self.interfaces:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        A = self.pde_coeff_A
        B = self.pde_coeff_B
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for interface in self.interfaces:
            if interface.get('direction') == 'temporal':
                total_loss = total_loss + self._temporal_interface_loss(
                    interface)
                continue

            net_L = self.subdomains[interface['left_idx']]['net']
            net_R = self.subdomains[interface['right_idx']]['net']

            x_if = torch.full((200, 1), interface['position'],
                              device=device, dtype=torch.float32)
            b_l = self.subdomains[interface['left_idx']]['bounds']
            b_r = self.subdomains[interface['right_idx']]['bounds']
            t_lo = max(b_l[0], b_r[0])
            t_hi = min(b_l[1], b_r[1])
            t_if = torch.rand((200, 1), device=device) * (
                t_hi - t_lo) + t_lo

            u_L = net_L(x_if, t_if)
            u_R = net_R(x_if, t_if)

            # Flux continuity: F(u) = -A*u + B*u^2/2
            F_L = -A * u_L + B * u_L.pow(2) / 2.0
            F_R = -A * u_R + B * u_R.pow(2) / 2.0
            loss_flux = torch.mean(torch.square(F_L - F_R))

            # Solution average
            u_avg = (u_L + u_R) / 2.0
            loss_avg = (torch.mean(torch.square(u_L - u_avg))
                        + torch.mean(torch.square(u_R - u_avg)))

            total_loss = total_loss + loss_flux + loss_avg

        return total_loss

    # ------------------------------------------------------------------
    # XPINN: residual continuity + solution average interface loss
    # ------------------------------------------------------------------
    def _xpinn_interface_loss(self):
        """XPINN interface coupling: residual continuity + solution average.

        Residual continuity: MSE(R_L - R_R) for PDE residual.
        Solution average: MSE(u_k - u_avg).
        Applied to both spatial and temporal interfaces.
        """
        if not self.interfaces:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for interface in self.interfaces:
            net_L = self.subdomains[interface['left_idx']]['net']
            net_R = self.subdomains[interface['right_idx']]['net']

            if interface.get('direction') == 'temporal':
                b_l = self.subdomains[interface['left_idx']]['bounds']
                b_r = self.subdomains[interface['right_idx']]['bounds']
                x_lo = max(b_l[2], b_r[2])
                x_hi = min(b_l[3], b_r[3])
                x_pts = (torch.rand((200, 1), device=device)
                         * (x_hi - x_lo) + x_lo)
                t_pts = torch.full((200, 1), interface['position'],
                                   device=device, dtype=torch.float32)
            else:
                b_l = self.subdomains[interface['left_idx']]['bounds']
                b_r = self.subdomains[interface['right_idx']]['bounds']
                t_lo = max(b_l[0], b_r[0])
                t_hi = min(b_l[1], b_r[1])
                x_pts = torch.full((200, 1), interface['position'],
                                   device=device, dtype=torch.float32)
                t_pts = (torch.rand((200, 1), device=device)
                         * (t_hi - t_lo) + t_lo)

            # Residual continuity (_pde_residual handles grad internally)
            R_L = self._pde_residual(net_L, x_pts, t_pts)
            R_R = self._pde_residual(net_R, x_pts, t_pts)
            loss_res = torch.mean(torch.square(R_L - R_R))

            # Solution average
            u_L = net_L(x_pts.detach(), t_pts.detach())
            u_R = net_R(x_pts.detach(), t_pts.detach())
            u_avg = (u_L + u_R) / 2.0
            loss_avg = (torch.mean(torch.square(u_L - u_avg))
                        + torch.mean(torch.square(u_R - u_avg)))

            total_loss = total_loss + loss_res + loss_avg

        return total_loss

    # ------------------------------------------------------------------
    # cPINN online training
    # ------------------------------------------------------------------
    def train_cpinn(self, epochs, batch_size, N_f, n_subdomains=3):
        """Online cPINN training with equal-spaced subdomains.

        No coarse pre-training stage. Subdomains are initialized with random
        weights and trained from epoch 0 with flux continuity coupling.

        Args:
            epochs: total training epochs
            batch_size: data mini-batch size
            N_f: total collocation points (split proportionally)
            n_subdomains: number of equal-spaced spatial subdomains
        """
        history = {'total': [], 'data': [], 'pde': [], 'int': [],
                   'epochs': []}
        n_data = self.x_data.shape[0]
        N_f_batch_size = 2048

        # --- Create equal-spaced subdomains with random init ---
        b = self.subdomains[0]['bounds']
        t_left, t_right = b[0], b[1]
        x_left, x_right = b[2], b[3]
        layers_sub = (self.layers_after_split
                      if self.layers_after_split else self.layers)

        edges = np.linspace(x_left, x_right, n_subdomains + 1).tolist()
        self.subdomains = []
        for k in range(n_subdomains):
            self.subdomains.append({
                'bounds': (t_left, t_right, edges[k], edges[k + 1]),
                'net': BaseNet(layers_sub, activation='tanh').to(device),
                'level': 0, 'parent_id': None,
            })

        self._rebuild_interfaces_from_subdomains()
        print(f"[cPINN] {n_subdomains} equal-spaced subdomains, "
              f"{len(self.interfaces)} interfaces")

        # --- Optimizer (single Adam for all subnets) ---
        all_params = []
        for sd in self.subdomains:
            all_params.extend(list(sd['net'].parameters()))
        self.optimizer_model = torch.optim.Adam(all_params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)

        # Fixed weights
        self.w_data = torch.tensor(
            0.85, requires_grad=False, device=device, dtype=torch.float32)
        self.w_pde = torch.tensor(
            0.05, requires_grad=False, device=device, dtype=torch.float32)
        self.w_int = torch.tensor(
            0.10, requires_grad=False, device=device, dtype=torch.float32)

        # --- Collocation points ---
        collocation_points = self._initialize_collocation_points(N_f)

        # --- Training loop ---
        start_time = time.time()
        total_loss = torch.tensor(0.0, device=device)

        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])

            collocation_batch_dict = {}
            n_subs = len(self.subdomains)
            effective_bs = max(512, N_f_batch_size // n_subs)
            for i in range(n_subs):
                x_col, t_col = collocation_points[f'subdomain_{i}']
                n_pts = x_col.shape[0]
                if n_pts == 0:
                    continue
                bs = min(effective_bs, n_pts)
                ci = torch.randint(0, n_pts, (bs,), device=device)
                collocation_batch_dict[f'subdomain_{i}'] = (
                    x_col[ci], t_col[ci])

            if not collocation_batch_dict:
                continue

            self.optimizer_model.zero_grad()

            # Data loss
            x_d, t_d, u_d = data_batch
            u_pred = self.predict(x_d, t_d)
            loss_data = torch.mean(torch.square(u_pred - u_d))

            # PDE loss per subdomain
            loss_pde = torch.tensor(
                0.0, device=device, dtype=torch.float32)
            num_valid = 0
            for i, sd in enumerate(self.subdomains):
                key = f'subdomain_{i}'
                if key not in collocation_batch_dict:
                    continue
                xc, tc = collocation_batch_dict[key]
                res = self._pde_residual(sd['net'], xc, tc)
                loss_pde = loss_pde + torch.mean(torch.square(res))
                num_valid += 1
            if num_valid > 0:
                loss_pde = loss_pde / num_valid

            # cPINN interface loss
            loss_int = self._cpinn_interface_loss()

            total_loss = (self.w_data * loss_data
                          + self.w_pde * loss_pde
                          + self.w_int * loss_int)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            self.optimizer_model.step()
            self.scheduler.step()

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f'[cPINN] Epoch {epoch+1}/{epochs} | '
                      f'Loss: {total_loss.item():.3e} | '
                      f'Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(loss_data.item())
                history['pde'].append(loss_pde.item())
                history['int'].append(loss_int.item())
                start_time = time.time()

        print(f"\n[cPINN] Training complete. "
              f"Final loss: {total_loss.item():.3e}")
        return history

    # ------------------------------------------------------------------
    # XPINN online training
    # ------------------------------------------------------------------
    def train_xpinn(self, epochs, batch_size, N_f,
                    n_spatial=2, n_temporal=2):
        """Online XPINN training with space-time grid subdomains.

        No coarse pre-training stage. Subdomains form an n_spatial x n_temporal
        grid and are initialized with random weights. Residual continuity
        coupling at all interfaces.

        Args:
            epochs: total training epochs
            batch_size: data mini-batch size
            N_f: total collocation points (split by area)
            n_spatial: number of spatial subdivisions
            n_temporal: number of temporal subdivisions
        """
        history = {'total': [], 'data': [], 'pde': [], 'int': [],
                   'epochs': []}
        n_data = self.x_data.shape[0]
        N_f_batch_size = 2048

        # --- Create grid subdomains with random init ---
        b = self.subdomains[0]['bounds']
        t_left, t_right = b[0], b[1]
        x_left, x_right = b[2], b[3]
        layers_sub = (self.layers_after_split
                      if self.layers_after_split else self.layers)

        x_edges = np.linspace(x_left, x_right, n_spatial + 1).tolist()
        t_edges = np.linspace(t_left, t_right, n_temporal + 1).tolist()

        self.subdomains = []
        for j in range(n_temporal):
            for i in range(n_spatial):
                self.subdomains.append({
                    'bounds': (t_edges[j], t_edges[j + 1],
                               x_edges[i], x_edges[i + 1]),
                    'net': BaseNet(layers_sub, activation='tanh').to(device),
                    'level': 0, 'parent_id': None,
                })

        self._rebuild_interfaces_from_subdomains()
        print(f"[XPINN] {n_spatial}x{n_temporal} grid = "
              f"{len(self.subdomains)} subdomains, "
              f"{len(self.interfaces)} interfaces")

        # Enable temporal decomp for area-based collocation allocation
        orig_temporal_decomp = self.use_temporal_decomp
        self.use_temporal_decomp = True

        # --- Optimizer ---
        all_params = []
        for sd in self.subdomains:
            all_params.extend(list(sd['net'].parameters()))
        self.optimizer_model = torch.optim.Adam(all_params, lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model, step_size=5000, gamma=0.9)

        # Fixed weights
        self.w_data = torch.tensor(
            0.85, requires_grad=False, device=device, dtype=torch.float32)
        self.w_pde = torch.tensor(
            0.05, requires_grad=False, device=device, dtype=torch.float32)
        self.w_int = torch.tensor(
            0.10, requires_grad=False, device=device, dtype=torch.float32)

        # --- Collocation points (area-proportional) ---
        collocation_points = self._initialize_collocation_points(N_f)

        # --- Training loop ---
        start_time = time.time()
        total_loss = torch.tensor(0.0, device=device)

        for epoch in range(epochs):
            idx = torch.randint(0, n_data, (min(batch_size, n_data),),
                                device=device)
            data_batch = (self.x_data[idx], self.t_data[idx],
                          self.u_data[idx])

            collocation_batch_dict = {}
            n_subs = len(self.subdomains)
            effective_bs = max(512, N_f_batch_size // n_subs)
            for i in range(n_subs):
                x_col, t_col = collocation_points[f'subdomain_{i}']
                n_pts = x_col.shape[0]
                if n_pts == 0:
                    continue
                bs = min(effective_bs, n_pts)
                ci = torch.randint(0, n_pts, (bs,), device=device)
                collocation_batch_dict[f'subdomain_{i}'] = (
                    x_col[ci], t_col[ci])

            if not collocation_batch_dict:
                continue

            self.optimizer_model.zero_grad()

            # Data loss
            x_d, t_d, u_d = data_batch
            u_pred = self.predict(x_d, t_d)
            loss_data = torch.mean(torch.square(u_pred - u_d))

            # PDE loss per subdomain
            loss_pde = torch.tensor(
                0.0, device=device, dtype=torch.float32)
            num_valid = 0
            for i, sd in enumerate(self.subdomains):
                key = f'subdomain_{i}'
                if key not in collocation_batch_dict:
                    continue
                xc, tc = collocation_batch_dict[key]
                res = self._pde_residual(sd['net'], xc, tc)
                loss_pde = loss_pde + torch.mean(torch.square(res))
                num_valid += 1
            if num_valid > 0:
                loss_pde = loss_pde / num_valid

            # XPINN interface loss
            loss_int = self._xpinn_interface_loss()

            total_loss = (self.w_data * loss_data
                          + self.w_pde * loss_pde
                          + self.w_int * loss_int)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            self.optimizer_model.step()
            self.scheduler.step()

            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f'[XPINN] Epoch {epoch+1}/{epochs} | '
                      f'Loss: {total_loss.item():.3e} | '
                      f'Time: {elapsed:.2f}s')
                history['epochs'].append(epoch + 1)
                history['total'].append(total_loss.item())
                history['data'].append(loss_data.item())
                history['pde'].append(loss_pde.item())
                history['int'].append(loss_int.item())
                start_time = time.time()

        self.use_temporal_decomp = orig_temporal_decomp
        print(f"\n[XPINN] Training complete. "
              f"Final loss: {total_loss.item():.3e}")
        return history

    def predict(self, x_pred, t_pred):
        """
        Fix 4: Ensemble prediction at spatial interfaces.
        Within a blending zone of half_width=0.02 around each spatial
        interface, predictions from both neighboring subnets are averaged
        using a smooth linear blend, reducing sensitivity to individual
        subnet quality near boundaries.
        """
        blend_half_width = 0.02  # in normalized coordinates

        # Collect spatial interface positions
        spatial_interfaces = []
        for intf in self.interfaces:
            if intf.get('direction') == 'spatial':
                spatial_interfaces.append(intf)

        # If no interfaces (single subdomain), use simple path
        if not spatial_interfaces:
            u_pred = torch.zeros_like(x_pred)
            for i, subdomain in enumerate(self.subdomains):
                b = subdomain['bounds']
                x_left, x_right = b[2], b[3]
                t_left, t_right = b[0], b[1]
                is_last = (i == len(self.subdomains) - 1)
                if is_last:
                    cond_x = (x_pred >= x_left) & (x_pred <= x_right)
                else:
                    cond_x = (x_pred >= x_left) & (x_pred < x_right)
                cond_t = (t_pred >= t_left) & (t_pred <= t_right)
                mask = cond_x & cond_t
                if mask.any():
                    mask_flat = mask.view(-1)
                    u_sub_vals = subdomain['net'](
                        x_pred[mask_flat], t_pred[mask_flat])
                    u_full = torch.zeros_like(x_pred)
                    u_full[mask_flat] = u_sub_vals
                    u_pred = torch.where(mask, u_full, u_pred)
            return u_pred

        # With interfaces: use accumulator approach for blending
        u_sum = torch.zeros_like(x_pred)
        w_sum = torch.zeros_like(x_pred)

        for i, subdomain in enumerate(self.subdomains):
            b = subdomain['bounds']
            x_left, x_right = b[2], b[3]
            t_left, t_right = b[0], b[1]

            # Extend evaluation range by blend_half_width
            eval_x_left = x_left - blend_half_width
            eval_x_right = x_right + blend_half_width
            cond_x = (x_pred >= eval_x_left) & (x_pred <= eval_x_right)
            cond_t = (t_pred >= t_left) & (t_pred <= t_right)
            mask = cond_x & cond_t

            if not mask.any():
                continue

            mask_flat = mask.view(-1)
            x_sub = x_pred[mask_flat]
            t_sub = t_pred[mask_flat]
            u_sub_vals = subdomain['net'](x_sub, t_sub)

            # Compute weight: 1.0 inside core, linear ramp in blend zone
            weight = torch.ones_like(x_sub)

            # Left blend zone: ramp from 0 to 1
            left_blend = (x_sub < x_left) & (x_sub >= eval_x_left)
            if left_blend.any():
                alpha = ((x_sub[left_blend] - eval_x_left)
                         / (blend_half_width + 1e-10))
                weight[left_blend] = alpha.squeeze()

            # Right blend zone: ramp from 1 to 0
            right_blend = (x_sub > x_right) & (x_sub <= eval_x_right)
            if right_blend.any():
                alpha = ((eval_x_right - x_sub[right_blend])
                         / (blend_half_width + 1e-10))
                weight[right_blend] = alpha.squeeze()

            # Accumulate weighted predictions
            u_contrib = torch.zeros_like(x_pred)
            w_contrib = torch.zeros_like(x_pred)
            u_contrib[mask_flat] = u_sub_vals * weight
            w_contrib[mask_flat] = weight

            u_sum = u_sum + u_contrib
            w_sum = w_sum + w_contrib

        # Normalize by total weight
        u_pred = u_sum / (w_sum + 1e-10)
        return u_pred


# ==============================================================================
# 3. Visualization Helpers
# ==============================================================================
def plot_results(model_name, Exact, U_pred, T_plot, X_plot,
                 u_min, u_max, model, x_train_plot, t_train_plot,
                 x_max_orig, x_min_orig, t_max_orig, num_sensors,
                 save_dir='figure'):
    """
    MODIFIED: Interface visualization handles both spatial and temporal types.
    """
    t_max_hours = t_max_orig / 1800.0
    hour_ticks = np.linspace(
        0, t_max_orig, int(np.ceil(t_max_hours)) + 1)
    hour_labels = [f'{h:.0f}' for h in np.linspace(
        0, t_max_hours, int(np.ceil(t_max_hours)) + 1)]

    total_miles = 4.0
    mile_ticks = np.linspace(x_min_orig, x_max_orig,
                             int(total_miles) + 1)
    mile_labels = [f'{m:.0f}' for m in np.linspace(
        0, total_miles, int(total_miles) + 1)]

    def setup_plot(ax, title):
        ax.set_title(title, fontsize=30)
        ax.set_xlabel('Time (hour)', fontsize=25)
        ax.set_ylabel('Distance (mile)', fontsize=25)
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(hour_labels)
        ax.set_yticks(mile_ticks)
        ax.set_yticklabels(mile_labels)
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.scatter(t_train_plot, x_train_plot,
                   c='black', marker='x', s=20, alpha=0.8,
                   label='Sensor Locations')
        # MODIFIED: handle both spatial and temporal interfaces
        if hasattr(model, 'interfaces') and model.interfaces:
            for interface in model.interfaces:
                direction = interface.get('direction', 'spatial')
                if direction == 'spatial':
                    # MODIFIED: use .item() if position is a torch.Tensor
                    pos = interface['position']
                    if isinstance(pos, torch.Tensor):
                        pos = pos.item()
                    pos_unnorm = (pos
                                  * (x_max_orig - x_min_orig) + x_min_orig)
                    ax.axhline(y=pos_unnorm, color='white',
                               linestyle='--', linewidth=2.5,
                               label="Spatial Interface")
                elif direction == 'temporal':
                    # NEW: temporal interface visualization
                    pos = interface['position']
                    if isinstance(pos, torch.Tensor):
                        pos = pos.item()
                    pos_unnorm = pos * t_max_orig
                    ax.axvline(x=pos_unnorm, color='cyan',
                               linestyle=':', linewidth=2.5,
                               label="Temporal Interface")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=25)

    # Ground Truth
    fig, ax = plt.subplots(figsize=(16, 12))
    h = ax.pcolormesh(T_plot, X_plot, Exact, cmap='jet',
                      vmin=u_min, vmax=u_max, shading='nearest')
    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    setup_plot(ax, f'Ground Truth Speed (mph) - {model_name}')
    plt.tight_layout()
    fname = (f"{model_name.replace(' ', '_').replace('+', 'and')}"
             f"_sensors_{num_sensors}_1_GroundTruth.png")
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)

    # Prediction
    fig, ax = plt.subplots(figsize=(16, 12))
    h = ax.pcolormesh(T_plot, X_plot, U_pred, cmap='jet',
                      vmin=u_min, vmax=u_max, shading='nearest')
    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    setup_plot(ax, f'Predicted Speed (mph) - {model_name}')
    plt.tight_layout()
    fname = (f"{model_name.replace(' ', '_').replace('+', 'and')}"
             f"_sensors_{num_sensors}_2_Prediction.png")
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)

    # Error
    fig, ax = plt.subplots(figsize=(16, 12))
    error = np.abs(Exact - U_pred)
    h = ax.pcolormesh(T_plot, X_plot, error, cmap='hot', shading='nearest')
    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    setup_plot(ax, f'Absolute Error (mph) - {model_name}')
    plt.tight_layout()
    fname = (f"{model_name.replace(' ', '_').replace('+', 'and')}"
             f"_sensors_{num_sensors}_3_Error.png")
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_loss_history(model_name, history, num_sensors, save_dir='figure'):
    if not history['epochs']:
        return
    fig = plt.figure(figsize=(16, 12))
    plt.plot(history['epochs'], history['total'], 'k',
             label='Total Loss', linewidth=2.5)
    plt.plot(history['epochs'], history['data'], 'r--',
             label='Data Loss', linewidth=2)
    plt.plot(history['epochs'], history['pde'], 'b--',
             label='PDE Loss', linewidth=2)
    if any(i > 0 for i in history['int']):
        plt.plot(history['epochs'], history['int'], 'g--',
                 label='Interface Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss (Log Scale)', fontsize=25)
    plt.title(f'Training Loss History for {model_name} '
              f'({num_sensors} Sensors)', fontsize=30)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    fname = (f"{model_name.replace(' ', '_').replace('+', 'and')}"
             f"_sensors_{num_sensors}_loss.png")
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # -------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------
    N_EPOCHS = 50000
    N_f = 50000
    BATCH_SIZE = 4096
    N_COLLOCATION_NEW = 2500
    FREQ_DECOMPOSITION = 5000
    FREQ_SAMPLING = 2500
    THRESHOLD_RESIDUAL = 1e-3
    LAYERS = [2, 256, 128, 128, 128, 1]
    LAYERS_AFTER_SPLIT = [2, 256, 128, 128, 1]
    NUM_SENSORS = 5

    # MODIFIED: Weight initialization — paper specifies (1/3, 1/3, 1/3)
    # v1 used (100, 1, 1) which is inconsistent with the paper.
    W_DATA_INIT = 1.0 / 3.0
    W_PDE_INIT = 1.0 / 3.0
    W_INT_INIT = 1.0 / 3.0

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    os.makedirs('figure_accident', exist_ok=True)
    os.makedirs('result_accident', exist_ok=True)

    params_dict = {
        'N_EPOCHS': N_EPOCHS, 'N_f_initial': N_f, 'BATCH_SIZE': BATCH_SIZE,
        'N_COLLOCATION_NEW_RAR': N_COLLOCATION_NEW,
        'FREQ_DECOMPOSITION': FREQ_DECOMPOSITION,
        'FREQ_SAMPLING_RAR': FREQ_SAMPLING,
        'THRESHOLD_RESIDUAL_DECOMP': THRESHOLD_RESIDUAL,
        'LAYERS_INITIAL': str(LAYERS),
        'LAYERS_AFTER_SPLIT': str(LAYERS_AFTER_SPLIT),
        'NUM_SENSORS': NUM_SENSORS,
        'W_DATA_INIT': W_DATA_INIT,
        'W_PDE_INIT': W_PDE_INIT,
        'W_INT_INIT': W_INT_INIT
    }
    params_df = pd.DataFrame([params_dict])
    params_csv_path = f'result_accident/hyperparameters_sensors_{NUM_SENSORS}.csv'
    params_df.to_csv(params_csv_path, index=False)
    print(f"Saved hyperparameters to: {params_csv_path}")

    # -------------------------------------------------
    # Data Loading and Preprocessing
    # -------------------------------------------------
    try:
        data = pd.read_csv('./data/20221121_accident.csv', header=0)
        data = data[['t', 'x', 'speed']]
    except FileNotFoundError:
        print("Error: CSV file not found. Check './data/20221121_accident.csv'")
        exit()

    # Convert ft/s to mph
    FT_PER_S_TO_MPH = 3600.0 / 5280.0
    data['speed'] = data['speed'] * FT_PER_S_TO_MPH

    data = data.astype(np.float32)
    x_min_orig, x_max_orig = data['x'].min(), data['x'].max()
    t_min_orig, t_max_orig = data['t'].min(), data['t'].max()
    u_min_orig, u_max_orig = data['speed'].min(), data['speed'].max()

    # MODIFIED: Estimate free-flow speed from I-24 data
    v_f_estimated = estimate_free_flow_speed(data['speed'],
                                              method='percentile_85')
    print(f"\n{'='*60}")
    print(f"FREE-FLOW SPEED ESTIMATION (I-24 data)")
    print(f"  Method: 85th percentile")
    print(f"  v_f = {v_f_estimated:.2f} mph")
    print(f"  (v1 used NGSIM hardcoded 46.64 mph)")
    print(f"  u_min = {u_min_orig:.2f} mph, u_max = {u_max_orig:.2f} mph")
    print(f"  x_range = {x_max_orig - x_min_orig:.1f} ft, "
          f"t_range = {t_max_orig - t_min_orig:.1f} s")
    print(f"{'='*60}\n")

    # Normalization parameters for PDE coefficients
    norm_params = {
        'v_f_physical': v_f_estimated,
        'u_min_physical': u_min_orig,
        'u_max_physical': u_max_orig,
        'x_range_physical': float(x_max_orig - x_min_orig),
        't_range_physical': float(t_max_orig - t_min_orig),
    }

    # Min-max normalization
    data['x_norm'] = (data['x'] - x_min_orig) / (x_max_orig - x_min_orig)
    data['t_norm'] = (data['t'] - t_min_orig) / (t_max_orig - t_min_orig)
    data['speed_norm'] = ((data['speed'] - u_min_orig)
                          / (u_max_orig - u_min_orig))

    # Ground Truth
    x_axis = np.sort(data['x_norm'].unique())
    t_axis = np.sort(data['t_norm'].unique())
    X, T = np.meshgrid(x_axis, t_axis)

    Exact_normalized = data.pivot_table(
        index='t_norm', columns='x_norm',
        values='speed_norm').values.astype(np.float32)
    Exact_unnormalized = (Exact_normalized * (u_max_orig - u_min_orig)
                          + u_min_orig)

    # Training data
    lb = np.array([data['x_norm'].min(), data['t_norm'].min()])
    ub = np.array([data['x_norm'].max(), data['t_norm'].max()])
    domain_bounds = (lb, ub)

    ideal_locations = np.linspace(lb[0], ub[0], NUM_SENSORS)
    actual_sensor_locations = [
        x_axis[np.argmin(np.abs(x_axis - loc))] for loc in ideal_locations]
    sensor_locations_norm = np.unique(actual_sensor_locations)
    sensor_data_df = data[data['x_norm'].isin(sensor_locations_norm)]

    x_sensor = sensor_data_df[['x_norm']].values
    t_sensor = sensor_data_df[['t_norm']].values
    u_sensor = sensor_data_df[['speed_norm']].values

    data_points = (torch.tensor(x_sensor, dtype=torch.float32, device=device),
                   torch.tensor(t_sensor, dtype=torch.float32, device=device),
                   torch.tensor(u_sensor, dtype=torch.float32, device=device))

    # -------------------------------------------------
    # Benchmark
    # -------------------------------------------------
    models_to_run = {
        "Basic NN": SimpleNN,
        "Basic PINN": VanillaPinnLWR,
        "Adaptive Weight PINN": AdaptiveWeightPinnLWR,
        "RAR PINN": RARPinnLWR,
        "ADA-PINN (Weight + RAR)": AdaPinnLWR,
        "STDPINN": StdpinnLWR,
        "ADA-STDPINN": AdaStdpinnLWR
    }

    benchmark_results = []

    for name, model_class in models_to_run.items():
        print("=" * 60)
        print(f"STARTING BENCHMARK: {name}")
        print("=" * 60)

        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        # MODIFIED: pass norm_params and updated weight inits
        model_init_params = {
            'layers': LAYERS,
            'w_data_init': W_DATA_INIT,
            'w_pde_init': W_PDE_INIT,
            'w_int_init': W_INT_INIT,
            'sensor_locations': sensor_locations_norm,
            'layers_after_split': LAYERS_AFTER_SPLIT,
            'use_temporal_decomp': False,   # NEW: v1 compat default
            **norm_params                    # NEW: PDE coefficients
        }

        if name == "Basic NN":
            model = model_class(data_points, **model_init_params)
        else:
            model = model_class(domain_bounds, data_points,
                                **model_init_params)

        # Count parameters
        params_count = 0
        if hasattr(model, 'subdomains') and model.subdomains:
            for sd in model.subdomains:
                params_count += sum(p.numel() for p in sd['net'].parameters())
        elif hasattr(model, 'net'):
            params_count += sum(p.numel() for p in model.net.parameters())

        if hasattr(model, 'interfaces') and model.interfaces:
            for interface in model.interfaces:
                if ('shock_speed' in interface
                        and interface['shock_speed'].requires_grad):
                    params_count += 1

        if (hasattr(model, 'w_data')
                and isinstance(model.w_data, torch.Tensor)
                and model.w_data.requires_grad):
            params_count += 1
        if (hasattr(model, 'w_pde')
                and isinstance(model.w_pde, torch.Tensor)
                and model.w_pde.requires_grad):
            params_count += 1
        if (hasattr(model, 'w_int')
                and isinstance(model.w_int, torch.Tensor)
                and model.w_int.requires_grad):
            params_count += 1

        start_total_time = time.time()

        train_params = {
            'epochs': N_EPOCHS, 'batch_size': BATCH_SIZE, 'N_f': N_f,
            'adaptive_sampling_freq': FREQ_SAMPLING,
            'domain_decomp_freq': FREQ_DECOMPOSITION,
            'num_new_points': N_COLLOCATION_NEW,
            'residual_threshold': THRESHOLD_RESIDUAL
        }
        history = model.train(**train_params)

        total_time = time.time() - start_total_time

        # Prediction
        X_flat, T_flat = X.flatten()[:, None], T.flatten()[:, None]
        pred_batch_size = 8192
        num_pred_samples = X_flat.shape[0]
        predictions = []
        with torch.no_grad():
            for i in range(0, num_pred_samples, pred_batch_size):
                x_batch = X_flat[i:i + pred_batch_size]
                t_batch = T_flat[i:i + pred_batch_size]
                u_batch = model.predict(
                    torch.tensor(x_batch, dtype=torch.float32, device=device),
                    torch.tensor(t_batch, dtype=torch.float32, device=device))
                predictions.append(u_batch)

        u_pred_normalized = torch.cat(predictions, dim=0)
        if u_pred_normalized.ndim > 1 and u_pred_normalized.shape[1] == 1:
            u_pred_normalized = u_pred_normalized.squeeze(-1)

        u_pred = (u_pred_normalized.detach().cpu().numpy()
                  .reshape(Exact_unnormalized.shape)
                  * (u_max_orig - u_min_orig) + u_min_orig)
        Exact, U_pred = Exact_unnormalized, u_pred

        mse = np.mean((Exact - U_pred) ** 2)
        rmse = np.sqrt(mse)
        relative_l2_error = (np.linalg.norm(Exact - U_pred)
                             / np.linalg.norm(Exact))

        results = {
            "Model": name, "Parameters": params_count,
            "MSE": mse, "RMSE": rmse,
            "Relative L2 Error": relative_l2_error,
            "Time (s)": total_time,
            "Subdomains": (len(model.subdomains)
                           if hasattr(model, 'subdomains') else 1)
        }
        benchmark_results.append(results)

        print(f"\n{'-'*50}")
        print(f"Final Metrics for {name} (Speed in mph)")
        print(f"{'-'*50}")
        print(f"  Trainable Parameters  : {params_count:,}")
        print(f"  MSE                   : {mse:.6f}")
        print(f"  RMSE                  : {rmse:.6f}")
        print(f"  Relative L2 Error     : {relative_l2_error:.4%}")
        print(f"  Training Time         : {total_time:.2f} s")
        if hasattr(model, 'subdomains'):
            print(f"  Final subdomains      : {len(model.subdomains)}")
        print(f"{'-'*50}\n")

        X_plot = (X * (x_max_orig - x_min_orig)) + x_min_orig
        T_plot = (T * (t_max_orig - t_min_orig)) + t_min_orig
        x_train_plot = sensor_data_df['x'].values
        t_train_plot = sensor_data_df['t'].values

        plot_results(name, Exact, U_pred, T_plot, X_plot,
                     u_min_orig, u_max_orig, model,
                     x_train_plot, t_train_plot,
                     x_max_orig, x_min_orig, t_max_orig,
                     NUM_SENSORS, save_dir='figure_accident')
        plot_loss_history(name, history, NUM_SENSORS,
                          save_dir='figure_accident')

    # -------------------------------------------------
    # Final Comparison
    # -------------------------------------------------
    results_df = pd.DataFrame(benchmark_results)
    results_df = results_df.set_index('Model')[[
        'Parameters', 'MSE', 'RMSE', 'Relative L2 Error',
        'Time (s)', 'Subdomains']]

    print(f"\n\n{'='*80}")
    print("FINAL BENCHMARK COMPARISON (Speed in mph)")
    print("=" * 80)
    print(results_df.to_string(formatters={
        'Parameters': '{:,}'.format,
        'MSE': '{:,.6f}'.format,
        'RMSE': '{:,.6f}'.format,
        'Relative L2 Error': '{:,.4%}'.format,
        'Time (s)': '{:,.2f}'.format
    }))
    print("=" * 80)

    comparison_csv_path = (
        f'result_accident/benchmark_comparison_sensors_{NUM_SENSORS}.csv')
    results_df.to_csv(comparison_csv_path)
    print(f"\nSaved benchmark comparison to: {comparison_csv_path}")
