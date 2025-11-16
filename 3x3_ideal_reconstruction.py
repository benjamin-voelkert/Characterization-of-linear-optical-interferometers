#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:31:33 2025

@author: benjaminvolkert
"""

import numpy as np
from haar_random_matrix_generation import haar_random_unitary
import measurement_data_simulation as mds
import ideal_LO_reconstruction as LO
import quantum_fidelity_LO as fid

# ---------------------------------------
# Round matrix
# ---------------------------------------
def round_complex_matrix(M, decimals=3, tiny=1e-12):
    """
    Round a complex-valued matrix elementwise.

    Parameters
    ----------
    M : array_like
        Input complex matrix.
    decimals : int
        Number of decimals to round to.
    tiny : float
        Threshold below which entries are set to exactly zero.

    Returns
    -------
    Mr : np.ndarray
        Rounded complex matrix.
    """
    M = np.asarray(M, dtype=complex)
    real = np.round(M.real, decimals)
    imag = np.round(M.imag, decimals)
    Mr = real + 1j * imag

    # Zero-out tiny entries for nicer printing
    Mr[np.abs(Mr) < tiny] = 0.0
    return Mr

# ---------------------------------------
# Add input/output port loss
# ---------------------------------------
def apply_loss(U, eta_in=None, eta_out=None, seed=None):
    """
    Applies mode-dependent input/output loss to a unitary matrix U.
    Returns a non-unitary matrix: D_out @ U @ D_in
    """
    n = U.shape[0]
    rng = np.random.default_rng(seed)

    # Generate random input/output losses if not given
    if eta_in is None:
        eta_in = rng.random(n)   # values between 0 and 1
    if eta_out is None:
        eta_out = rng.random(n)  # values between 0 and 1

    # Build diagonal attenuation matrices
    D_in = np.diag(np.sqrt(eta_in))
    D_out = np.diag(np.sqrt(eta_out))

    # Apply losses
    return D_out @ U @ D_in, D_in, D_out

# ---------------------------------------------------
# Example
# ---------------------------------------------------
if __name__ == "__main__":
    N = 3
    U = haar_random_unitary(N) #generate Haar-random unitary
    U_rb,_,_ = fid.make_real_bordered(U) # make it real-bordered
    U_lossy, L_in, L_out = apply_loss(U) # apply loss at input and output ports
    spp, tpp = mds.compute_transition_probabilities(U_lossy) # compute measurement data
    M2 = LO.laing_obrien_reconstruct(spp, tpp, N) # reconstruct the matrix from noisy measurement data
    
    # Quantum gate fidelity of the reconstruction
    F, branch = fid.fidelity_max_over_conjugation(U_rb,M2)
    print(f"\nQuantum gate fidelity: {F:.6f}")
    
    # Print matrices
    print(round_complex_matrix(L_out))
    print()
    print(round_complex_matrix(U_rb))
    print()
    print(round_complex_matrix(L_in))
    print()
    print(round_complex_matrix(M2))
    
    







