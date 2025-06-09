r"""Utilities for spin operations on molecular Hamiltonians."""

import numpy as np


def spinize_H(one_body: np.ndarray,
              two_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert the one/two body terms from spatial to spin-orbital basis.

    Parameters
    ----------
    one_body : np.ndarray
        One-body term in spatial basis in physics notation
    two_body : np.ndarray
        Two-body term in spatial basis in physics notation

    Returns
    -------
    one_body_spin : np.ndarray
        One-body term in spin-orbital basis in physics notation
    two_body_spin : np.ndarray
        Two-body term in spin-orbital basis in physics notation

    Notes
    -----
    Rules for the conversion:
    - The one-body term is converted as follows:
        - :math:`h_{pq}^{\\alpha \\alpha}=h_{pq}^{\\beta \\beta}
        =h_{pq}^{\\text{spatial}}`
        - :math:`h_{pq}^{\\alpha \\beta}=h_{pq}^{\\beta \\alpha}=0`
    - The two-body term is converted as follows:
        - :math:`V_{pqrs}^{\\alpha \\alpha \\alpha \\alpha}=\\
                 V_{pqrs}^{\\alpha \\beta \\alpha \\beta}=\\
                 V_{pqrs}^{\\beta \\alpha \\beta \\alpha}=\\
                 V_{pqrs}^{\\text{spatial}}`

    """
    one_body = np.asarray(one_body)
    two_body = np.asarray(two_body)

    if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
        raise ValueError("one_body must be square (n, n)")
    n = one_body.shape[0]
    if two_body.shape != (n, n, n, n):
        raise ValueError("two_body must have shape (n, n, n, n) with same n")

    one_body_spin = np.zeros((2 * n, 2 * n), dtype=one_body.dtype)
    one_body_spin[:n, :n] = one_body   # αα block
    one_body_spin[n:, n:] = one_body   # ββ block

    two_body_spin = np.zeros((2 * n, 2 * n, 2 * n, 2 * n),
                             dtype=two_body.dtype)
    # αααα
    two_body_spin[:n, :n, :n, :n] = two_body
    # ββββ
    two_body_spin[n:, n:, n:, n:] = two_body
    # αβαβ
    two_body_spin[:n, n:, :n, n:] = two_body
    # βαβα
    two_body_spin[n:, :n, n:, :n] = two_body

    return one_body_spin, two_body_spin


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
