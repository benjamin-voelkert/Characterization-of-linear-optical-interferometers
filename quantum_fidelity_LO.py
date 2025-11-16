#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:29:56 2025

@author: benjaminvolkert
"""

import numpy as np
import fidelity_of_reconstruction as fid

# ---------------------------------------------------
# Make a unitary matrix real-bordered
# ---------------------------------------------------
def make_real_bordered(U, eps=0.0):
    """
    Transform a unitary matrix U into a 'real-bordered' form via diagonal rephasing:
        U_rb = D_left @ U @ D_right
    such that the first column and first row of U_rb are real and nonnegative.

    Parameters
    ----------
    U : ndarray, shape (d, d)
        Unitary (or approximately unitary) matrix.
    eps : float, optional (default=0.0)
        Magnitude threshold below which phases are ignored (for numerical stability).

    Returns
    -------
    U_rb : ndarray, shape (d, d)
        Real-bordered version of U.
    D_left : ndarray, shape (d, d)
        Left diagonal unitary used for rephasing.
    D_right : ndarray, shape (d, d)
        Right diagonal unitary used for rephasing.
    """
    U = np.asarray(U, dtype=complex)

    # Step 1: Left rephasing to make first column real and nonnegative
    col = U[:, 0]
    phases_left = np.where(np.abs(col) > eps, -np.angle(col), 0.0)
    D_left = np.diag(np.exp(1j * phases_left))
    U1 = D_left @ U

    # Step 2: Right rephasing to make first row real and nonnegative
    row = U1[0, :]
    phases_right = np.where(np.abs(row) > eps, -np.angle(row), 0.0)
    D_right = np.diag(np.exp(1j * phases_right))
    U_rb = U1 @ D_right

    return U_rb, D_left, D_right

# ---------------------------------------------------
# Fidelity maximization over complex conjugation
# ---------------------------------------------------
def fidelity_max_over_conjugation(A, B):
    """
    Compute quantum fidelity between matrix A and B,
    returning the larger value obtained from comparing
    A with either B or its complex conjugate B.conj().

    Parameters
    ----------
    A, B : ndarray, shape (d, d)
        Matrices to compare (usually unitary or close to unitary).

    Returns
    -------
    fidelity : float
        The maximum fidelity value in [0,1].
    branch : int
        0 if best match was with B, 1 if best match was with B.conj().
    """
    B0 = B          # original matrix
    B1 = B.conj()   # complex conjugate

    # Compute fidelities against both variants
    F_plain = fid.quantum_fidelity_basic(A, B0)
    F_conj  = fid.quantum_fidelity_basic(A, B1)

    # Return the maximum fidelity and which branch was used
    if F_conj > F_plain:
        return F_conj, 1  # branch 1: used B.conj()
    else:
        return F_plain, 0  # branch 0: used B