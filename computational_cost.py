#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 23:20:17 2025

@author: benjaminvolkert
"""

# ===== Computational cost of μ-sweep (quantum): simple fixed-exponent fits =====

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Mapping
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import more_robust_LO_reconstruction as LO
from haar_random_matrix_generation import haar_random_unitary
import quantum_fidelity_LO as fid
from partial_distinguishability import measurements_with_mu, make_mu_list


# -------------------------
# Lightweight timing helper
# -------------------------
class TimerAcc:
    def __init__(self): self.d = {}
    def add(self, key: str, dt: float): self.d[key] = self.d.get(key, 0.0) + dt
    def as_dict(self, scale: float = 1.0) -> Dict[str, float]:
        return {k: v / scale for k, v in self.d.items()}

def _total_cost(cost_dict: Dict[str, float]) -> float:
    return float(sum(cost_dict.values()))


# ---------------------------------------------------
# μ-sweep (quantum) — computational cost benchmark
# ---------------------------------------------------
def benchmark_mu_quantum_cost(
    N: int,
    mus: List[float],
    *,
    num_devices: int = 12,
    seed0: int = 123,
    mu_by_input_pair: Optional[Mapping[Tuple[int,int], float]] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Average seconds per inner trial for the μ-sweep (quantum only).
    Inner trial = one μ value for one device.
    Returns (average phase times dict, num_inner_trials).
    """
    t = TimerAcc()
    iters = 0

    with tqdm(total=num_devices * len(mus), desc=f"μ-sweep timing (N={N})", unit="dev·μ") as pbar:
        for _ in range(num_devices):
            # Haar device
            t0 = time.perf_counter()
            U = haar_random_unitary(N)
            t.add("haar", time.perf_counter() - t0)

            # Gauge fix once per device
            t0 = time.perf_counter()
            U_rb, _, _ = fid.make_real_bordered(U)
            t.add("gauge_fix", time.perf_counter() - t0)

            for mu in mus:
                # Measurements (SPP + TPP_μ)
                t0 = time.perf_counter()
                spp, tpp = measurements_with_mu(U_rb, mu_global=mu, mu_by_input_pair=mu_by_input_pair)
                t.add("meas_gen_mu", time.perf_counter() - t0)

                # LO reconstruction
                t0 = time.perf_counter()
                M2 = LO.laing_obrien_reconstruct(spp, tpp, N)
                t.add("reconstruct", time.perf_counter() - t0)

                # Fidelity (max over conjugation)
                t0 = time.perf_counter()
                _F_qu, _branch = fid.fidelity_max_over_conjugation(U_rb, M2)
                t.add("fidelity", time.perf_counter() - t0)

                iters += 1
                pbar.update(1)

    return t.as_dict(scale=iters), iters


# ---------------------------------------------------
# Sweep Ns and collect totals + reconstruction times
# ---------------------------------------------------
def total_cost_vs_N_mu(
    Ns: List[int],
    mus: List[float],
    *,
    num_devices: int = 12,
    seed0: int = 123,
    mu_by_input_pair: Optional[Mapping[Tuple[int,int], float]] = None,
) -> Tuple[List[int], List[float], List[float]]:
    xs, totals, recon_only = [], [], []
    base_rng = np.random.default_rng(seed0)
    for N in Ns:
        seedN = int(base_rng.integers(0, 2**31 - 1))
        phase_times, _iters = benchmark_mu_quantum_cost(
            N, mus,
            num_devices=num_devices,
            seed0=seedN,
            mu_by_input_pair=mu_by_input_pair,
        )
        xs.append(N)
        totals.append(_total_cost(phase_times))
        recon_only.append(phase_times.get("reconstruct", 0.0))
    return xs, totals, recon_only


# ---------------------------------------------------
# Fixed-exponent fits: a * N^k  (no intercept)
# ---------------------------------------------------
def best_coeff_fixed_k(Ns: List[int], ys: List[float], k: float) -> float:
    """Least-squares coefficient 'a' for fixed exponent k in y ≈ a * N^k."""
    N = np.asarray(Ns, dtype=float)
    Y = np.asarray(ys, dtype=float)
    mask = (N > 0) & (Y > 0)
    if mask.sum() == 0:
        return float("nan")
    X = N[mask] ** k
    a = float(np.dot(X, Y[mask]) / np.dot(X, X))
    return max(0.0, a)


# ---------------------------------------------------
# Plots (linear axes only)
# ---------------------------------------------------
def plot_combined_linear_with_fits(Ns: List[int], totals: List[float], recon: List[float]):
    """Combined plot with O(N^4) for total and O(N^2) for reconstruction (points only for measured)."""
    a4 = best_coeff_fixed_k(Ns, totals, k=4.0)
    a2 = best_coeff_fixed_k(Ns, recon,  k=2.0)
    fit_tot = (np.asarray(Ns, dtype=float) ** 4) * a4 if np.isfinite(a4) else None
    fit_rec = (np.asarray(Ns, dtype=float) ** 2) * a2 if np.isfinite(a2) else None

    plt.figure(figsize=(8.4, 4.8))
    
    # measured points only (no connecting lines)
    plt.plot(Ns, totals, "o", ms=7, color="C0", label="Total Simulation")
    if fit_tot is not None:
        plt.plot(Ns, fit_tot, ":", lw=2, color="C0", label="Fit ~ O($N^4$)")
    plt.plot(Ns, recon,  "s", ms=7, color="C3", label="Laing-O'Brien Reconstruction")
    if fit_rec is not None:
        plt.plot(Ns, fit_rec, ":", lw=2, color="C3", label="Fit ~ O($N^2$)")

    plt.xlabel("Interferometer size N")
    plt.ylabel("Seconds per inner trial")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Also print reconstruction share at each N
    pct = [100.0 * (r/t if t > 0 else 0.0) for r, t in zip(recon, totals)]
    for n, p in zip(Ns, pct):
        print(f"N={n:>2}: reconstruction share ~ {p:5.1f}%")


# ----------------
# Run
# ----------------
if __name__ == "__main__":
    # Adjust as needed
    Ns  = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    mus = make_mu_list(0.90, 1.00, 10)
    num_devices = 20

    xs, totals, recon = total_cost_vs_N_mu(
        Ns, mus,
        num_devices=num_devices,
        seed0=123
    )

    plot_combined_linear_with_fits(xs, totals, recon)