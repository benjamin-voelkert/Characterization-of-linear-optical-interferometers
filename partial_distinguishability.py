#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:24:03 2025

@author: benjaminvolkert
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Mapping
import numpy as np

# -----------------------
# String key constructors
# -----------------------
def _key_spp(j: int, i: int) -> str:
    """
    Build a single-photon probability key "Pji" using 1-based port indices.

    Args:
        j: Output port index (1-based).
        i: Input  port index (1-based).

    Returns:
        A string key like "P31" for P_{3,1}.
    """
    return f"P{j}{i}"

def _key_tpp(o1: int, i1: int, o2: int, i2: int) -> str:
    """
    Build a two-photon probability key "Po1i1o2i2" using 1-based port indices.

    Args:
        o1: Output port for photon 1 (1-based).
        i1:  Input port for photon 1 (1-based).
        o2: Output port for photon 2 (1-based).
        i2:  Input port for photon 2 (1-based).

    Returns:
        A string key like "P3142" for P_{o1=3,i1=1,o2=4,i2=2}.
    """
    return f"P{o1}{i1}{o2}{i2}"

# -----------------------------
# Single-photon (SPP) modeling
# -----------------------------
def singles_probs(U: np.ndarray) -> Dict[str, float]:
    """
    Compute single-photon transition probabilities (SPP).

    Convention:
      - Keys are 1-based: "Pji" corresponds to P_{j,i}.
      - NumPy indexing is 0-based; we subtract 1 for array access.
      - T_{j,i} = |U_{j,i}|^2, with j = output, i = input.

    Args:
        U: Unitary (N x N) as a complex NumPy array.

    Returns:
        Dict mapping "Pji" -> probability float.
    """
    U = np.asarray(U)
    N = U.shape[0]
    return {
        _key_spp(j, i): float(np.abs(U[j - 1, i - 1]) ** 2)
        for i in range(1, N + 1)
        for j in range(1, N + 1)
    }

# -------------------------------------------
# Two-photon (TPP) with partial distinguishability
# -------------------------------------------
def two_photon_probs_partial_dist(
    U: np.ndarray,
    *,
    mu_global: float = 1.0,
    mu_by_input_pair: Optional[Mapping[Tuple[int, int], float]] = None,
) -> Dict[str, float]:
    """
    Two-photon transition probabilities with partial (in)distinguishability.

    Model:
        Q_μ = μ * Q_indist + (1 - μ) * Q_dist,
      where
        Q_indist = |U[o1,i1] U[o2,i2] + U[o1,i2] U[o2,i1]|^2,
        Q_dist   = |U[o1,i1]|^2 |U[o2,i2]|^2 + |U[o1,i2]|^2 |U[o2,i1]|^2.

    Notes:
      - Only collision-free events are included: i1 != i2 and o1 != o2.
      - Keys use 1-based indices: "Po1i1o2i2".
      - If mu_by_input_pair is provided, it overrides mu_global per ordered pair (i1, i2).
      - μ is clamped to [0, 1].

    Args:
        U: Unitary (N x N) as a complex NumPy array.
        mu_global: Default indistinguishability μ for all input pairs.
        mu_by_input_pair: Optional per-(i1, i2) μ overrides (1-based indices).

    Returns:
        Dict mapping "Po1i1o2i2" -> probability float.
    """
    U = np.asarray(U, dtype=complex)
    N = U.shape[0]
    tpp: Dict[str, float] = {}

    for i1 in range(1, N + 1):
        for i2 in range(1, N + 1):
            if i1 == i2:
                # Exclude input collisions
                continue

            # Select μ for this ordered input pair and clamp to [0, 1]
            mu = mu_global if mu_by_input_pair is None else float(
                mu_by_input_pair.get((i1, i2), mu_global)
            )
            mu = max(0.0, min(1.0, mu))

            for o1 in range(1, N + 1):
                for o2 in range(1, N + 1):
                    if o1 == o2:
                        # Exclude output collisions
                        continue

                    # Shorthand amplitudes (0-based array access)
                    a = U[o1 - 1, i1 - 1] * U[o2 - 1, i2 - 1]
                    b = U[o1 - 1, i2 - 1] * U[o2 - 1, i1 - 1]

                    # Indistinguishable and distinguishable contributions
                    Q_indist = np.abs(a + b) ** 2
                    Q_dist = (
                        (np.abs(U[o1 - 1, i1 - 1]) ** 2) * (np.abs(U[o2 - 1, i2 - 1]) ** 2)
                        + (np.abs(U[o1 - 1, i2 - 1]) ** 2) * (np.abs(U[o2 - 1, i1 - 1]) ** 2)
                    )

                    # μ-mixture
                    tpp[_key_tpp(o1, i1, o2, i2)] = float(mu * Q_indist + (1.0 - mu) * Q_dist)

    return tpp

# -------------------------------
# Aggregate measurement interface
# -------------------------------
def measurements_with_mu(
    U: np.ndarray,
    *,
    mu_global: float = 1.0,
    mu_by_input_pair: Optional[Mapping[Tuple[int, int], float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute both SPP and TPP_μ dictionaries from U and μ settings.

    Args:
        U: Unitary (N x N).
        mu_global: Default μ for all input pairs.
        mu_by_input_pair: Optional overrides for specific ordered (i1, i2) pairs.

    Returns:
        (spp, tpp_mu):
            spp    : dict "Pji"     -> |U[j,i]|^2
            tpp_mu : dict "Po1i1o2i2" -> μ*Q_indist + (1-μ)*Q_dist
    """
    return singles_probs(U), two_photon_probs_partial_dist(
        U, mu_global=mu_global, mu_by_input_pair=mu_by_input_pair
    )

# ----------------------
# μ grid convenience API
# ----------------------
def make_mu_list(mu_min: float, mu_max: float, num_points: int) -> List[float]:
    """
    Uniformly spaced μ values in [mu_min, mu_max].

    Args:
        mu_min: Lower bound (inclusive).
        mu_max: Upper bound (inclusive).
        num_points: Number of grid points.

    Returns:
        List of floats of length num_points.
    """
    return list(np.linspace(mu_min, mu_max, num_points))