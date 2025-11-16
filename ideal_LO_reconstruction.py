#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:25:51 2025

@author: benjaminvolkert
"""

import numpy as np
from numpy.linalg import solve

# ---------------------------------------
# Laing-O'Brien reconstruction protocol
# ---------------------------------------
def laing_obrien_reconstruct(spp, tpp, n, eps=1e-18):
    """
    Reconstruct an n x n unitary matrix using the Laing-O'Brien protocol.
    
    Parameters
    ----------
    spp : dict
        Single-photon probabilities, e.g. {'Pij': value}
    tpp : dict
        Two-photon probabilities, e.g. {'Pghjk': value}
    n : int
        Number of modes
    eps : float
        Small tolerance for error checking
    
    Returns
    -------
    M2 : np.ndarray
        Reconstructed unitary of shape (n, n).
    """
    # Helper to compute the coefficient x_ij
    def coeff_x(g, h, j, k):
        """ x_{ghjk} = sqrt(P_jk * P_gh / (P_jh * P_gk)) """
        Pgh, Pjk, Pjh, Pgk = f"P{g}{h}", f"P{j}{k}", f"P{j}{h}", f"P{g}{k}"
        try:
            val = (spp[Pjk] * spp[Pgh]) / (spp[Pjh] * spp[Pgk])
            if val <= 0:
                raise ValueError(f"Invalid coefficient value for modes {(g,h,j,k)}.")
            return np.sqrt(val), (np.sqrt(val) + np.sqrt(val)**(-1))
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Division error for modes {(g,h,j,k)}.")
            
    # Visibility
    def visibility(g, h, j, k):
        """ V_{ghjk} = (C_{ghjk} - Q_{ghjk}) / C_{ghjk} """
        C = (spp[f"P{j}{k}"] * spp[f"P{g}{h}"]) + (spp[f"P{g}{k}"] * spp[f"P{j}{h}"])
        Q = tpp[f"P{g}{h}{j}{k}"]
        return (C - Q) / C if abs(C) > eps else 0.0

    # Delta for phase angle 
    def delta(g, h, j, k):
        """ cos(beta) value necessary for phase angle calculation """
        return -0.5 * visibility(g, h, j, k) * coeff_x(g, h, j, k)[1]

    # General sign determination
    def determine_phase_sign(g, h, alpha_abs, angles):
        """Determine the sign of alpha_{gh} using the generalized rules"""
        if g == 2 and h == 2:
            return 1  # case 1

        elif h == 2 and g > 2:  # case 2: second column
            d = delta(g, h, 2, 1)
            a = angles[1, 1]  # alpha_22
            s1 = abs(d - np.cos(-a - alpha_abs))
            s2 = abs(d - np.cos(-a + alpha_abs))
            return np.sign(s1 - s2)

        elif g == 2 and h > 2:  # case 3: second row
            d = delta(g, h, 1, 2)
            a = angles[1, 1]  # alpha_22
            s1 = abs(d - np.cos(-a - alpha_abs))
            s2 = abs(d - np.cos(-a + alpha_abs))
            return np.sign(s1 - s2)

        else:  # case 4: everything else
            d = delta(g, h, 2, 2)
            a22 = angles[1, 1]
            ag2 = angles[g - 1, 1]
            a2h = angles[1, h - 1]
            arg_sum = a22 - a2h - ag2
            s1 = abs(d - np.cos(arg_sum - alpha_abs))
            s2 = abs(d - np.cos(arg_sum + alpha_abs))
            return np.sign(s1 - s2)

    # Initialize the coefficient matrix M_mu
    M_mu = np.ones((n, n), dtype=complex)

    angles = np.zeros((n, n))
    angles[0, 0] = 0.0  # fix phase for the (1,1) element
    X_values = {}
    Y_values = {}

    # Build X_values for arbitrary g, h, j, k 
    for g in range(1, n + 1):
        for h in range(1, n + 1):
            X_values[(g, h)] = coeff_x(g, h, 1, 1)[0]
            Y_values[(g, h)] = coeff_x(g, h, 1, 1)[1]

    # Solve for column intensities
    P_target = np.zeros(n, dtype=complex)
    P_target[0] = 1.0

    for g in range(n):
        for h in range(n):
            if g == 0:
                M_mu[g, h] = X_values[(h + 1, g + 1)]
            elif h == 0:
                M_mu[g, h] = X_values[(g + 1, h + 1)]
            else:
                vis = visibility(g + 1, h + 1, 1, 1)
                arg = -0.5 * vis * Y_values[(g + 1, h + 1)]
                arg = np.clip(arg, -1.0, 1.0)            
                angle_candidate = float(abs(np.arccos(arg)))
                angles[g, h] = determine_phase_sign(g + 1, h + 1, angle_candidate, angles) * angle_candidate
                M_mu[g, h] = X_values[(g + 1, h + 1)] * np.exp(1j * angles[g, h])

    # Solve for squared transition probabilities
    t_squared_row = solve(M_mu, P_target) # transisiton probabilities first row
    t_squared_column = solve(M_mu.conj().T, P_target) # transition probabilities first column

    # Reconstruct M2
    M2 = np.zeros((n, n), dtype=complex)

    # First row and column: take square roots of transition probabilities
    t_row = np.sqrt(np.clip(np.real(t_squared_row), 0.0, None))     
    t_col = np.sqrt(np.clip(np.real(t_squared_column), 0.0, None))  

    # Fill first row and first column
    M2[0, :] = t_row
    M2[:, 0] = t_col

    # Fill the rest
    for g in range(1, n):
        for h in range(1, n):
            base = M2[0, 0]
            # If the absolute reference is tiny, nudge it to avoid blow-ups 
            if abs(base) < eps:
                base = eps
            
            # If either multiplicative factor is ~0, the entry should be ~0 — set it explicitly.
            if (abs(M2[0, h]) < eps) or (abs(M2[g, 0]) < eps):
                M2[g, h] = 0.0
            else:
                M2[g, h] = (
                    X_values[(g + 1, h + 1)]
                    * (M2[g, 0] * M2[0, h] / base)
                    * np.exp(1j * angles[g, h])
                )
    
    return M2