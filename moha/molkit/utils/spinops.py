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

    n = nspin // 2      # number of spatial orbitals

    # αααα block
    aa = tensor[:n, :n, :n, :n].copy()
    aa = aa - np.einsum('pqrs->qprs', aa) - np.einsum('pqrs->pqsr', aa) +\
             np.einsum('pqrs->rspq', aa) - np.einsum('pqrs->srpq', aa) - np.einsum('pqrs->rsqp', aa) +\
             np.einsum('pqrs->qpsr', aa) + np.einsum('pqrs->srqp', aa)

    # ββββ block
    bb = tensor[n:, n:, n:, n:].copy()
    bb = bb - np.einsum('pqrs->qprs', bb) - np.einsum('pqrs->pqsr', bb) +\
                np.einsum('pqrs->rspq', bb) - np.einsum('pqrs->srpq', bb) - np.einsum('pqrs->rsqp', bb) +\
                np.einsum('pqrs->qpsr', bb) + np.einsum('pqrs->srqp', bb)


    tensor[:n, :n, :n, :n] = aa / 8
    tensor[n:, n:, n:, n:] = bb / 8

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


def upscale_one_body(one_body, n_elec):
    r"""
    Upscale the 1 body term to the 2 body term.

    Specifically, the one-body term is upscaled to a 4-dimensional tensor.

    Parameters
    ----------
    one_body : np.ndarray
        One-body term in spin-orbital basis in physics notation.
    n_elec : int
        Number of electrons in the system

    Returns
    -------
    one_body_up : np.ndarray
        Upscaled one-body integrals

    Notes
    -----
    The upscaling is done by the following formula:

    .. math::

        \\frac{1}{2 (n-1)}(\\mathbf{h}_{pq}\\delta_{rs} +
        \\mathbf{h}_{rs}\\delta_{pq})

    where:
    - :math:`\\mathbf{h}_{pq}` and :math:`\\mathbf{h}_{rs}` are the one-body
    - :math:`\\delta_{rs}` and :math:`\\delta_{pq}` are Kronecker deltas
    - :math:`n` is the number of electrons in the system

    The resulting upscaled one-body term is a 4-dimensional tensor.

    """
    n = one_body.shape[0]
    eye = np.eye(n)
    one_body_up = 0.5 * (np.kron(one_body, eye) +
                         np.kron(eye, one_body)) / (n_elec - 1)

    return one_body_up.reshape(n, n, n, n)
