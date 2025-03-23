r"""MoHa utilities submodule."""

import numpy as np

from scipy.sparse import diags, csr_matrix, issparse

import re

import scipy.sparse as sp

__all__ = [
    "convert_indices",
    "get_atom_type",
    "expand_sym",
]


def convert_indices(N, *args):
    r"""
    Convert indices from 4d array to 2d numpy array and vice-versa.

    :param N: size of corresponding _4d_ matrix: int
    :param args: indices i, j, k, l or p,q: int
    :return: list of converted indices: list
    """
    if len(args) == 4:
        # Check if indices are right
        for elem in args:
            if not isinstance(elem, int) and not isinstance(elem, np.int64) \
                    and not isinstance(elem, np.int32):
                raise TypeError("Wrong indices")
            if elem >= N:
                raise TypeError("index is greater than size of the matrix")
        # converting indices
        i, j, k, l_ = args
        p = int(i * N + j)
        q = int(k * N + l_)
        return [p, q]

    elif len(args) == 2:
        # check indices
        for elem in args:
            if not isinstance(elem, int) and not isinstance(elem, np.int32):
                raise TypeError("Wrong indices")
            if elem >= N**2:
                raise TypeError("index is greater than size of the matrix")

        # converting indices
        p, q = args
        i, k = p // N, q // N
        j, l_ = p % N, q % N
        return [i, j, k, l_]
    else:
        raise TypeError("Wrong indices")


def expand_sym(sym, integral, nbody):
    r"""
    Restore permutational symmetry of one- and two-body terms.

    Parameters
    ----------
    sym: int
        integral symmetry, one of 1 (no symmetry), 2, 4 or 8.
    integral: scipy.sparce.csr_matrix
        2-D sparse array, the {one,two}-body integrals
    nbody: int
        number of particle variables in the integral,
        one of 1 (one-body) or 2 (two-body)

    Returns
    -------
    integral: scipy.sparse.csr_matrix
        2d array with the symmetry 1

    Notes
    -----
        Given the one- or two-body Hamiltonian matrix terms,
        :math:`h_{i,j}` and :math:`g_{ij,kl}` respectively,
        the supported permutational symmetries are:
        sym = 2:
        :math:`h_{i,j} = h_{j,i}`
        :math:`g_{ij,kl} = g_{kl,ij}`
        sym = 4:
        :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji}`
        sym = 8:
        :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji} =
         g_{kj,il} = g_(il,kj) = g_(li,jk) = g_(jk,li)`
        sym = 1 corresponds to no-symmetry
        where it is assumed the integrals are over real orbitals.

        The input Hamiltonian terms are expected to be sparse
        arrays of dimensions :math:`(N,N)` or
        :math:`(N^2, N^2)` for the one- and two-body integrals respectively.
        :math:`N` represents the number of basis functions,
        which may be either of spatial or spin-orbital type.
        This function applies to the input array the
        permutations indicated by the symmetry parameter `sym`
        to adds the missing terms.
        Phicisist notation is used for the two-body integrals:
         :math:`<pq|rs>` and further details of the
        permutations considered can be
        found in [this site]
        (http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.html).
    """
    if sym not in [1, 2, 4, 8]:
        raise ValueError("Wrong input symmetry")
    if nbody not in [1, 2]:
        raise ValueError(f"`nbody` must be an integer, "
                         f"either 1 or 2, but {nbody} given")
    if sym == 1:
        return integral

    # Expanding Symmetries
    if nbody == 1:
        if not sym == 2:
            raise ValueError("Wrong 1-body term symmetry")
        h_ii = diags(integral.diagonal()).copy()
        integral = integral + integral.T - h_ii
    else:
        # getting nonzero elements from the 2d _sparse_ array
        integral = integral.tolil()
        pq_array, rs_array = integral.nonzero()
        n = int(np.sqrt(integral.shape[0]))

        for pq, rs in zip(pq_array, rs_array):
            p, q, r, s = convert_indices(n, int(pq), int(rs))
            if sym >= 2:
                # 1. Transpose: <pq|rs>=<rs|pq>
                rs, pq = convert_indices(n, r, s, p, q)
                integral[rs, pq] = integral[pq, rs]
            if sym >= 4:
                # 2. Permute dummy indices
                # (swap variables of particles 1 and 2):
                # <p_1 q_2|r_1 s_2> = <q_1 p_2|s_1 r_2>
                qp, sr = convert_indices(n, q, p, s, r)
                integral[qp, sr] = integral[pq, rs]
                integral[sr, qp] = integral[rs, pq]
            if sym == 8:
                # 3. Permute orbitals of the same variable,
                # e.g. <p_1 q_2|r_1 s_2> = <r_1 q_2|p_1 s_2>
                rq, ps = convert_indices(n, r, q, p, s)
                sp, qr = convert_indices(n, s, p, q, r)
                integral[rq, ps] = integral[pq, rs]
                integral[ps, rq] = integral[rs, pq]
                integral[sp, qr] = integral[qp, sr]
                integral[qr, sp] = integral[sr, qp]
        integral = integral.tocsr()
    return integral


def fill_o2(o2):
    """
    Fill the 2-body matrix with the missing elements.

    Parameters
    ----------
    o2: np.ndarray
        4-D array, the two-body integrals

    Returns
    -------
    o2: np.ndarray
        4d array with the symmetry 1
    """
    # loop over nonzero elements
    for i, j, k, l in np.nonzero(o2):
        o2[j, i, k, l] = - o2[i, j, k, l]
        o2[j, i, l, k] = o2[i, j, k, l]
        o2[i, j, l, k] = -o2[i, j, k, l]
    return o2




def enforce_pg_symmetry(eri: np.ndarray) -> np.ndarray:
    """Enforces point-group symmetry."""
    return 0.5 * (eri + eri.transpose(2, 3, 0, 1))

def enforce_hermitian_symmetry(eri: np.ndarray) -> np.ndarray:
    """Enforces Hermitian symmetry."""
    return 0.5 * (eri + eri.transpose(2, 3, 0, 1).conj())

def apply_permutational_symmetry(eri: np.ndarray) -> np.ndarray:
    """Enforces permutational symmetry."""
    eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    eri = 0.5 * (eri + eri.transpose(3, 2, 1, 0))
    return eri

def enforce_trs_symmetry(eri: np.ndarray) -> np.ndarray:
    """Enforces time-reversal symmetry."""
    return 0.5 * (eri + eri.transpose(2, 3, 0, 1).conj())

def antisymmetrize_two_electron_integrals(
    eri: np.ndarray,
    enforce_pg_symmetry: bool = False,
    enforce_hermitian: bool = False,
    enforce_spin_symmetry_func = None,
    enforce_permutational_symmetry: bool = False,
    enforce_trs: bool = False,
    n_spin_orbitals: int = None
) -> np.ndarray:
    """
    Antisymmetrizes two-electron integrals with optional symmetries.

    Args:
        eri: Two-electron integrals, either as a dense array or sparse matrix.
        enforce_pg_symmetry: Enforce point-group symmetry.
        enforce_hermitian: Enforce Hermitian symmetry.
        enforce_spin_symmetry_func: Function to enforce spin symmetry.
        enforce_permutational_symmetry: Enforce permutational symmetry.
        enforce_trs: Enforce time-reversal symmetry.
        n_spin_orbitals: Number of spin orbitals (required for spin symmetry).

    Returns:
        Antisymmetrized two-electron integrals as a dense array.
    """

    # Convert sparse matrix to dense if necessary
    if sp.issparse(eri):
        eri = eri.toarray()

    assert isinstance(eri, np.ndarray), f"ERI should be a NumPy array, but got {type(eri)}"

    # Reshape if necessary
    if eri.ndim == 2:
        n_orbitals = int(np.sqrt(eri.shape[0]))
        if n_orbitals ** 2 != eri.shape[0]:
            raise ValueError(f"Cannot reshape ERI of shape {eri.shape} into (n, n, n, n).")
        eri = eri.reshape(n_orbitals, n_orbitals, n_orbitals, n_orbitals)
    elif eri.ndim != 4:
        raise ValueError(f"ERI must be a 4D array or a 2D array that can be reshaped into 4D, but got shape {eri.shape}.")

    # Antisymmetrization (Pauli exclusion principle)
    antisymmetrized_eri = 0.5 * (eri - eri.transpose(1, 0, 2, 3))
    antisymmetrized_eri -= 0.5 * (eri.transpose(0, 1, 3, 2) - eri.transpose(1, 0, 3, 2))

    # Apply optional symmetries
    if enforce_pg_symmetry:
        antisymmetrized_eri = enforce_pg_symmetry(antisymmetrized_eri)

    if enforce_hermitian:
        antisymmetrized_eri = enforce_hermitian_symmetry(antisymmetrized_eri)

    if enforce_permutational_symmetry:
        antisymmetrized_eri = apply_permutational_symmetry(antisymmetrized_eri)

    if enforce_trs:
        antisymmetrized_eri = enforce_trs_symmetry(antisymmetrized_eri)

    if enforce_spin_symmetry_func and n_spin_orbitals:
        antisymmetrized_eri = enforce_spin_symmetry_func(antisymmetrized_eri, n_spin_orbitals)

    return antisymmetrized_eri
