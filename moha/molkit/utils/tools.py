r"""Utilities for molecular Hamiltonians."""

import numpy as np


def from_geminal(two_body_gem, n_orb):
    """
    Inverse of MolHam.to_geminal().

    Parameters
    ----------
    two_body_gem : (n_gem, n_gem) ndarray
        Matrix in the geminal basis.
    n_orb : int
        Number of spin orbitals.

    Returns
    -------
    V : (n_orb, n_orb, n_orb, n_orb) ndarray
        Fully antisymmetrised two-electron tensor V_{ijkl}.
    """
    n_gem = n_orb * (n_orb - 1) // 2
    if two_body_gem.shape != (n_gem, n_gem):
        raise ValueError(f"Shape mismatch: got {two_body_gem.shape}")

    # Generate flattened pair list exactly like to_geminal
    pairs = [(i, j) for i in range(n_orb) for j in range(i + 1, n_orb)]
    V = np.zeros((n_orb, n_orb, n_orb, n_orb))

    for A, (i, j) in enumerate(pairs):
        for B, (k, l) in enumerate(pairs):
            val = 0.25 * two_body_gem[A, B]  # undo the factor 0.5 * 4 = 2

            # Apply antisymmetric filling
            V[i, j, k, l] = val
            V[j, i, k, l] = -val
            V[i, j, l, k] = -val
            V[j, i, l, k] = val
            V[k, l, i, j] = val
            V[l, k, i, j] = -val
            V[k, l, j, i] = -val
            V[l, k, j, i] = val

    return V
