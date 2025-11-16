#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:26:44 2025

@author: benjaminvolkert
"""

# ===== Quantum gate fidelity F_qu vs μ (partial distinguishability), averaged over devices =====
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Mapping
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator

import more_robust_LO_reconstruction as LO
from haar_random_matrix_generation import haar_random_unitary
import quantum_fidelity_LO as fid
from partial_distinguishability import measurements_with_mu, make_mu_list


# ---------------------------------------------------
# Quantum gate fidelity F_qu vs μ for a single size N
# ---------------------------------------------------
def quantum_gate_fidelity_vs_mu(
    N: int,
    mus: List[float],
    *,
    num_devices: int = 12,
    seed0: int = 123,
    mu_by_input_pair: Optional[Mapping[Tuple[int,int], float]] = None,
) -> Dict[float, Dict[str, float]]:
    """
    For each device U~Haar(N) and each μ:
      - Gauge fix U → U_rb (real-bordered).
      - Build SPP/TPP_μ from U_rb.
      - Reconstruct M2 via Laing–O'Brien (LO) which internally polar-projects.
      - Compute quantum gate fidelity F_qu(U_rb, M2), maximized over complex conjugation.
    Returns per-μ statistics across devices: mean, between-device std, and SEM.
    """
    per_device_means: Dict[float, List[float]] = {mu: [] for mu in mus}

    with tqdm(total=num_devices * len(mus), desc=f"Quantum gate fidelity vs μ (N={N})", unit="dev·μ") as pbar:
        for _ in range(num_devices):
            U = haar_random_unitary(N)
            U_rb, _, _ = fid.make_real_bordered(U)

            F_by_mu: Dict[float, float] = {}
            for mu in mus:
                spp, tpp = measurements_with_mu(U_rb, mu_global=mu, mu_by_input_pair=mu_by_input_pair)
                M2 = LO.laing_obrien_reconstruct(spp, tpp, N)
                F_qu, _branch = fid.fidelity_max_over_conjugation(U_rb, M2)  # unpack (value, branch)
                F_by_mu[mu] = F_qu
                pbar.update(1)

            for mu in mus:
                per_device_means[mu].append(F_by_mu[mu])

    summary: Dict[float, Dict[str, float]] = {}
    for mu in mus:
        arr = np.asarray(per_device_means[mu], dtype=float)
        mean = float(np.mean(arr))
        between_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sem = between_std / np.sqrt(max(len(arr), 1))
        summary[mu] = {"mean": mean, "std_between": between_std, "sem": sem}
    return summary


# ---------------------------------------------------
# Multi-size runner and optional persistence helpers
# ---------------------------------------------------
def quantum_gate_fidelity_vs_mu_multiN(
    Ns: List[int],
    mus: List[float],
    *,
    num_devices: int = 12,
    seed0: int = 123
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """Run F_qu vs μ for multiple N and return nested results {N: {μ: stats}}."""
    all_results: Dict[int, Dict[float, Dict[str, float]]] = {}
    base_rng = np.random.default_rng(seed0)
    for N in Ns:
        seedN = int(base_rng.integers(0, 2**31 - 1))
        all_results[N] = quantum_gate_fidelity_vs_mu(
            N, mus,
            num_devices=num_devices,
            seed0=seedN
        )
    return all_results


# ---------------------------------------------------
# Plotting
# ---------------------------------------------------
def plot_F_qu_vs_mu_multiN(
    results_by_N: Dict[int, Dict[float, Dict[str, float]]],
    mus: List[float],
    title: str = "Quantum gate fidelity F_qu vs μ (SEM across devices)",
):
    """Errorbar plot of mean F_qu vs μ for each N, with SEM across devices."""
    plt.figure(figsize=(7.2, 4.6))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    for i, (N, summary) in enumerate(sorted(results_by_N.items())):
        means = [summary[mu]["mean"] for mu in mus]
        sems  = [summary[mu]["sem"]  for mu in mus]
        mk = markers[i % len(markers)]
        plt.errorbar(mus, means, yerr=sems, fmt=mk+"-", capsize=3, label=f"N={N}")
    plt.xlabel("Partial distinguishability μ")
    plt.ylabel(r"Averaged quantum gate fidelity $\overline{F}$")
    # plt.title(title)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.025)) # Set y-axis ticks every 0.025
    plt.grid(True, alpha=0.3)
    plt.legend(title="Interferometer size")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.show()


# ----------------
# Simulation
# ----------------
if __name__ == "__main__":
    Ns  = [3, 6, 9]
    mus = make_mu_list(0.9, 1.00, 15)

    results_by_N = quantum_gate_fidelity_vs_mu_multiN(
        Ns, mus,
        num_devices=1000,   # increase for tighter SEM
        seed0=123
    )

    # Display results
    for N in Ns:
        print(f"\n=== N={N} ===")
        for mu in mus:
            r = results_by_N[N][mu]
            print(f"μ={mu:.2f}  mean={r['mean']:.6f}  between-std={r['std_between']:.6f}  SEM={r['sem']:.6f}")

    plot_F_qu_vs_mu_multiN(results_by_N, mus)