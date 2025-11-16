#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:49:51 2025

@author: benjaminvolkert
"""
import numpy as np

# ---------------------------------------------------
# Quantum gate fidelity between two matrices
# ---------------------------------------------------
def quantum_fidelity_basic(A, B):
    """
    Compute a quantum gate fidelity between two matrices A and B:
        F = |Tr(A^† B)| / d

    Parameters
    ----------
    A, B : ndarray, shape (d, d)
        Unitary (or approximately unitary) matrices to compare.

    Returns
    -------
    fidelity : float
        Value in [0,1], where 1 means identical processes.
    """
    d = A.shape[0]
    # Compute normalized trace overlap between A and B
    return float(np.abs(np.trace(A.conj().T @ B)) / d)

# ---------------------------------------------------
# Classical fidelity between two probability distributions
# ---------------------------------------------------
def classical_fidelity(dist_p: dict, dist_q: dict) -> float:
    keys = set(dist_p.keys()) | set(dist_q.keys())
    # Collect the union of all possible outcomes
    """
    Compute the classical fidelity (Bhattacharyya coefficient) 
    between two discrete probability distributions.

    Parameters
    ----------
    dist_p, dist_q : dict
        Keys   = outcomes
        Values = probabilities (non-negative, sum to ~1)

    Returns
    -------
    fidelity : float
        Value in [0,1], where 1 means identical distributions.
    """
    bc = 0.0
    # Loop over outcomes and sum overlap contributions
    for k in keys:
        p = float(dist_p.get(k, 0.0))
        q = float(dist_q.get(k, 0.0))
        if p > 0.0 and q > 0.0:
            bc += np.sqrt(p * q)
    return float(bc)



