r"""Utilities for spin operations on molecular Hamiltonians."""

import numpy as np


def antisymmetrize_two_body(
        tensor: np.ndarray,
        *,
        inplace: bool = True) -> np.ndarray:
    r"""Antisymmetrize a two-electron integral tensor in spin-orbital basis.

    Parameters
    ----------
    tensor : np.ndarray
        Four-index array ``(2n, 2n, 2n, 2n)`` produced by :func:`spinize_H`.
    inplace : bool, default True
        If *True* modify *tensor* in place, else return a new antisymmetrised
        copy.

    Returns
    -------
    np.ndarray
        Antisymmetrised tensor obeying

        .. math::

           V_{pqrs} = -V_{pqsr} = -V_{qprs} = V_{qpsr},

        Mixed-spin blocks (αβ αβ and βα βα) are left unchanged because the
        exchange integral between opposite spins vanishes :
        contentReference[oaicite:1]{index=1}.

    Notes
    -----
    The operation applied is

    .. math::

        V^{\\sigma\\sigma\\sigma\\sigma}_{pqrs}
            \\;\\;\\leftarrow\\;\\;
            \\tfrac12\\,\\bigl(V^{\\sigma\\sigma\\sigma\\sigma}_{pqrs}
                              -V^{\\sigma\\sigma\\sigma\\sigma}_{pqsr}\\bigr),

    for ``σ = α`` and ``σ = β``.  All other spin sectors are returned
    untouched.

    """
    if not inplace:
        tensor = tensor.copy()

    nspin = tensor.shape[0]
    if nspin % 2:
        raise ValueError("spin-orbital tensor size must be even (2 n)")

    n = nspin // 2                       # number of spatial orbitals

    # αααα block
    aa = tensor[:n, :n, :n, :n]
    aa -= np.swapaxes(aa, 2, 3)
    aa *= 0.5

    # ββββ block
    bb = tensor[n:, n:, n:, n:]
    bb -= np.swapaxes(bb, 2, 3)
    bb *= 0.5

    return tensor


def get_spin_blocks(two_body_spin, n_spatial):
    """Return the main spin blocks of a two-body spin-orbital tensor.

    Parameters
    ----------
    two_body_spin : np.ndarray
        (2n, 2n, 2n, 2n) array of two-electron integrals in spin-orbital basis.
    n_spatial : int
        Number of spatial orbitals (n), where spin-orbitals = 2 * n.

    Returns
    -------
    dict
        Dictionary with spin blocks:
        - 'aaaa': alpha-alpha-alpha-alpha
        - 'bbbb': beta-beta-beta-beta
        - 'abab': alpha-beta-alpha-beta

    """
    return {
        "aaaa": two_body_spin[:n_spatial, :n_spatial, :n_spatial, :n_spatial],
        "bbbb": two_body_spin[n_spatial:, n_spatial:, n_spatial:, n_spatial:],
        "abab": two_body_spin[:n_spatial, n_spatial:, :n_spatial, n_spatial:],
    }
