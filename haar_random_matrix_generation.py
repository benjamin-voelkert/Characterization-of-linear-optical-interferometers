#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 18:36:31 2025

@author: benjaminvolkert
"""

import numpy as np
from scipy.linalg import qr

# ---------------------------------------
# Generate Haar-random unitary matrix
# ---------------------------------------
def haar_random_unitary(n):
    """
    Generate a Haar-random unitary matrix of size n x n.
    """
    # Ginibre matrix (complex normal)
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
    
    # QR decomposition using scipy
    q, r = qr(z)
    
    # Normalize the diagonal of R to get a Haar-distributed unitary
    d = np.diag(r)          # extract diagonal of R
    ph = d / np.abs(d)      # phase factors (unit modulus complex numbers)
    q = q * ph              # broadcast multiplication normalizes columns

    return q

# ---------------------------------------------------
# Example usage + sanity check
# ---------------------------------------------------
if __name__ == "__main__":
    n = 3
    U = haar_random_unitary(n)

    print("Haar-random unitary matrix U:\n", U)

    # Sanity check: verify unitarity (U^† U = I)
    identity = np.eye(n)
    check = U.conj().T @ U   # U^† U
    error = np.linalg.norm(check - identity)  # Frobenius norm of deviation

    print("\nUnitarity check (‖U†U - I‖):", error)
    if error < 1e-12:
        print("U is unitary within numerical precision.")
    else:
        print("U is NOT unitary (numerical error too large).")