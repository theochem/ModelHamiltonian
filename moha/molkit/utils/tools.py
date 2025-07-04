r"""Utilities for molecular Hamiltonians."""

import numpy as np


def _build_pair_lut(n_orb):
    """Return  two arrays  idx2pair[A]=(p,q)  and  pair2idx[p,q]=A."""
    n_gem = n_orb * (n_orb - 1) // 2
    idx2pair = np.empty((n_gem, 2), dtype=int)
    pair2idx = -np.ones((n_orb, n_orb), dtype=int)

    A = 0
    for p in range(n_orb):
        for q in range(p + 1, n_orb):
            idx2pair[A] = (p, q)
            pair2idx[p, q] = A
            pair2idx[q, p] = A    #
            A += 1
    return idx2pair, pair2idx


def from_geminal(two_body_gem, n_orb, type='rdm2'):
    """
    Inverse of to_geminal().

    Parameters
    ----------
    two_body_gem : (n_gem, n_gem) ndarray
        Matrix in the geminal basis.
    n_orb : int
        Number of spin orbitals.
    type : {'rdm2', 'h2'}
        Must match the `type` used in to_geminal().

    Returns
    -------
    V : (n_orb, n_orb, n_orb, n_orb) ndarray
        Two-electron tensor in spin-orbital physics notation,
        fully antisymmetrised:  V_{ijkl} = -V_{jikl} = -V_{ijlk} = V_{jilk}
    """
    n_gem = n_orb * (n_orb - 1) // 2
    if two_body_gem.shape != (n_gem, n_gem):
        raise ValueError("geminal tensor shape mismatch: " f"got {
                         two_body_gem.shape}, expected {(n_gem, n_gem)}")

    idx2pair, pair2idx = _build_pair_lut(n_orb)

    V = np.zeros((n_orb, n_orb, n_orb, n_orb))

    for A in range(n_gem):
        p, q = idx2pair[A]
        for B in range(n_gem):
            r, s = idx2pair[B]
            val = two_body_gem[A, B]

            if type == 'rdm2':
                # to_geminal used factor 2 = 0.5*4  → divide by 2 to invert
                val *= 0.5
            # type == 'h2' : no scaling (it was a sum/diff already)

            # fill the 8 permutations required by antisymmetry
            V[p, q, r, s] = val
            V[q, p, r, s] = -val
            V[p, q, s, r] = -val
            V[q, p, s, r] = val
            V[r, s, p, q] = val
            V[s, r, p, q] = -val
            V[r, s, q, p] = -val
            V[s, r, q, p] = val

    return V
