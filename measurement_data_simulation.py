#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 20:17:46 2025

@author: benjaminvolkert
"""
import numpy as np
from thewalrus import perm
from haar_random_matrix_generation import haar_random_unitary


# ---------------------------------------
# Simulate  ideal single- and two-photon data relevant for Laing-O'Brien protocol:
# -> Count rates as a proportion are exactly the transition probabilities for
#    single photons (single_photon_probs)
# -> Coincidence count rates as a proportion (Q value in LO) of two photons 
#    injected in two different modes correspond exactly to the two-photon
#    coincidence probabilities (two_photon_probs)
# ---------------------------------------


# ---------------------------------------
# Compute single- and two-photon transition probabilities
# ---------------------------------------
def compute_transition_probabilities(U):
    """
    Compute single-photon and two-photon (collision-free) transition probabilities
    for an N x N unitary U, in the sense of Tichy et al.:
    indistinguishable bosons -> probabilities are |perm(submatrix)|^2.

    Returns
    -------
    single_photon_probs : dict
        Keys 'Pji' (1-based indices). Value = |U[j-1, i-1]|^2.
        Interpreted as: photon entering input i exits at output j.
    two_photon_probs : dict
        Keys 'Po1i1o2i2' (1-based indices). Value = |perm(U_sub)|^2,
        where U_sub = U[[o1-1, o2-1], :][:, [i1-1, i2-1]] is a 2x2 submatrix.
        Only *collision-free* terms are included (i1 != i2 and o1 != o2).
    """
    # ---- basic validation ----
    U = np.asarray(U)
    assert U.ndim == 2 and U.shape[0] == U.shape[1], "U must be square."
    N = U.shape[0]

    # ---- single-photon probabilities: Pji = |U[j,i]|^2 ----
    single_photon_probs = {}
    for i in range(N):        # input mode index (0-based)
        for j in range(N):    # output mode index (0-based)
            single_photon_probs[f"P{j+1}{i+1}"] = np.abs(U[j, i])**2

    # ---- two-photon probabilities (collision-free, indistinguishable bosons) ----
    # Two photons injected in distinct inputs i1 != i2 and detected in distinct outputs o1 != o2.
    # Probability amplitude ~ perm(2x2 submatrix of U) ; probability = |amplitude|^2.
    two_photon_probs = {}
    for i1 in range(N):
        for i2 in range(N):
            if i1 == i2:
                continue  # exclude double-occupancy inputs
            for o1 in range(N):
                for o2 in range(N):
                    if o1 == o2:
                        continue  # exclude bunching (same output)
                    sub = U[np.ix_([o1, o2], [i1, i2])]
                    amp = perm(sub)
                    two_photon_probs[f"P{o1+1}{i1+1}{o2+1}{i2+1}"] = np.abs(amp)**2

    return single_photon_probs, two_photon_probs

# ---------------------------------------
# Demo
# ---------------------------------------
if __name__ == "__main__":
    N = 3
    U = haar_random_unitary(N)
    print(U)

    spp, tpp = compute_transition_probabilities(U)
    print("\nSingle-photon probabilities (top 12 by prob):")
    shown_1 = 0
    for k, v in sorted(spp.items(), key=lambda tp: -tp[-1]):
        print(f"  {str(k):<12}: {v:.6f}")
        shown_1 += 1
        if shown_1 >= 12 and len(spp) > 12:
            print(f"  ... ({len(spp)-12} more outcomes)")
            break

    print("\nTwo-photon (collision-free) probabilities (top 12 by prob):")
    shown_2 = 0
    for k, v in sorted(tpp.items(), key=lambda tp: -tp[-1]):
        print(f"  {str(k):<12}: {v:.6f}")
        shown_2 += 1
        if shown_2 >= 12 and len(tpp) > 12:
            print(f"  ... ({len(tpp)-12} more outcomes)")
            break

    
    







