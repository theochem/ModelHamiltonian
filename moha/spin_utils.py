r"""Fock-space spin operator utilities.

This module implements spin-1/2 operators using the Fock-space occupation-number
representation.  Each lattice site k hosts one alpha (spin-up) orbital and one
beta (spin-down) orbital.  A many-body basis state is encoded as a pair of
integers (alpha_bits, beta_bits) where bit k of alpha_bits (beta_bits) is 1 iff
orbital k_alpha (k_beta) is occupied.

Spin operators are expressed via fermionic creation / annihilation operators:

    S_k^+  = a†_{k,alpha} a_{k,beta}
    S_k^-  = a†_{k,beta}  a_{k,alpha}
    S_k^Z  = (1/2)(n_{k,alpha} - n_{k,beta})

Fermionic sign convention
--------------------------
When applying a†_i or a_i to a bit-string state, the sign is (-1)^(number of
occupied orbitals *before* position i in the canonical ordering).  The canonical
ordering used here is:

    0_alpha, 1_alpha, ..., (n-1)_alpha, 0_beta, 1_beta, ..., (n-1)_beta

So for orbital (site=k, spin='alpha') the global index is k, and for
(site=k, spin='beta') the global index is n_sites + k.

References
----------
Helgaker, Jorgensen, Olsen – "Molecular Electronic-Structure Theory" (2000),
Chapter 1: Occupation-number representation.
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple

__all__ = [
    "fock_basis",
    "spin_half_basis",
    "apply_creation",
    "apply_annihilation",
    "spin_plus_matrix",
    "spin_minus_matrix",
    "spin_z_matrix",
    "verify_commutations",
    "build_xyz_hamiltonian",
]

# ---------------------------------------------------------------------------
# Type alias: a basis state is (alpha_bits: int, beta_bits: int)
# ---------------------------------------------------------------------------
State = Tuple[int, int]


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def fock_basis(n_sites: int, n_alpha: int, n_beta: int) -> List[State]:
    r"""Return all Fock-space basis states with exactly n_alpha spin-up and
    n_beta spin-down electrons distributed over n_sites sites.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    n_alpha : int
        Number of spin-up (alpha) electrons  (0 <= n_alpha <= n_sites).
    n_beta : int
        Number of spin-down (beta) electrons (0 <= n_beta  <= n_sites).

    Returns
    -------
    basis : list of (int, int)
        Each element is (alpha_bits, beta_bits) - a pair of integers whose
        bit patterns encode the occupied orbitals.

    Notes
    -----
    The Nα, Nβ sector corresponds to a fixed *seniority* class when
    considered at half-filling.  For example, with n_sites = L the sector
    (n_alpha = L, n_beta = 0) contains the single maximum-seniority state
    with all spins pointing up.

    Examples
    --------
    >>> basis = fock_basis(2, 1, 1)   # 2 sites, 1 alpha, 1 beta electron
    >>> len(basis)
    4
    """
    if not (0 <= n_alpha <= n_sites and 0 <= n_beta <= n_sites):
        raise ValueError(
            f"n_alpha and n_beta must be between 0 and n_sites={n_sites}."
        )

    # All ways to choose n_alpha sites for alpha electrons
    alpha_configs = [
        sum(1 << site for site in occ)
        for occ in combinations(range(n_sites), n_alpha)
    ]
    # All ways to choose n_beta sites for beta electrons
    beta_configs = [
        sum(1 << site for site in occ)
        for occ in combinations(range(n_sites), n_beta)
    ]

    basis = [(a, b) for a in alpha_configs for b in beta_configs]
    return basis


def spin_half_basis(n_sites: int) -> List[State]:
    r"""Return the physical spin-1/2 basis: exactly one electron per site.

    This enforces the single-occupancy constraint: each site k has either an
    alpha electron OR a beta electron, never both and never neither.  The
    resulting basis has 2^n_sites states, one for each classical spin
    configuration on the lattice.

    This is the correct subspace for spin-1/2 operators because:
    - S_k^+ = a†_{k,alpha} a_{k,beta}: requires a beta electron to remove and
      no alpha electron present at site k (otherwise Pauli exclusion blocks it).
    - The operation maps one valid spin-1/2 state to another.

    Parameters
    ----------
    n_sites : int

    Returns
    -------
    basis : list of (int, int), length 2**n_sites
        States ordered by (alpha_bits, beta_bits).

    Notes
    -----
    The *seniority* s of a spin-1/2 state is the number of sites where the
    electron is unpaired relative to some reference pairing.  In this basis,
    the maximum-seniority states are those where all electrons have the same
    spin (all up or all down).  The Nα, Nβ sub-sector (Nα + Nβ = n_sites) has
    seniority s = Nα - Nβ when Nα > Nβ (or |Nα - Nβ| in general).

    Examples
    --------
    >>> basis = spin_half_basis(2)
    >>> len(basis)
    4
    >>> basis   # (alpha_bits, beta_bits): 1 means site 0, 2 means site 1
    [(0, 3), (1, 2), (2, 1), (3, 0)]
    """
    basis = []
    for alpha_bits in range(1 << n_sites):
        # beta occupies exactly the sites NOT occupied by alpha
        beta_bits = ((1 << n_sites) - 1) & ~alpha_bits
        basis.append((alpha_bits, beta_bits))
    return basis


def _state_index(basis: List[State], state: State) -> int:
    """Return the index of *state* in *basis*, or -1 if absent."""
    try:
        return basis.index(state)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Fermionic sign helper
# ---------------------------------------------------------------------------

def _fermionic_sign(bits: int, target_bit: int) -> int:
    r"""Compute the fermionic sign (±1) for (de)occupying orbital *target_bit*.

    The sign is (-1)^(number of set bits *strictly below* target_bit) in the
    bit-string *bits*.

    Parameters
    ----------
    bits : int
        Bit-string representing current occupancies.
    target_bit : int
        Index of the orbital being created / annihilated.

    Returns
    -------
    int
        +1 or -1.
    """
    # Count occupied orbitals with index < target_bit
    mask = (1 << target_bit) - 1
    n_below = bin(bits & mask).count("1")
    return (-1) ** n_below


# ---------------------------------------------------------------------------
# Elementary creation / annihilation operators
# ---------------------------------------------------------------------------

def _apply_creation_raw(
    alpha_bits: int,
    beta_bits: int,
    site: int,
    spin: str,
) -> Tuple[int, int, float]:
    r"""Apply a†_{site,spin} to raw bitstrings; returns new bitstrings + sign.

    Returns (new_alpha, new_beta, sign) or (-1, -1, 0.0) if result is zero.
    """
    if spin == "alpha":
        if (alpha_bits >> site) & 1:
            return -1, -1, 0.0
        sign = _fermionic_sign(alpha_bits, site)
        return alpha_bits | (1 << site), beta_bits, float(sign)
    elif spin == "beta":
        if (beta_bits >> site) & 1:
            return -1, -1, 0.0
        n_alpha_total = bin(alpha_bits).count("1")
        sign = (-1) ** n_alpha_total * _fermionic_sign(beta_bits, site)
        return alpha_bits, beta_bits | (1 << site), float(sign)
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin!r}")


def _apply_annihilation_raw(
    alpha_bits: int,
    beta_bits: int,
    site: int,
    spin: str,
) -> Tuple[int, int, float]:
    r"""Apply a_{site,spin} to raw bitstrings; returns new bitstrings + sign.

    Returns (new_alpha, new_beta, sign) or (-1, -1, 0.0) if result is zero.
    """
    if spin == "alpha":
        if not (alpha_bits >> site) & 1:
            return -1, -1, 0.0
        sign = _fermionic_sign(alpha_bits, site)
        return alpha_bits & ~(1 << site), beta_bits, float(sign)
    elif spin == "beta":
        if not (beta_bits >> site) & 1:
            return -1, -1, 0.0
        n_alpha_total = bin(alpha_bits).count("1")
        sign = (-1) ** n_alpha_total * _fermionic_sign(beta_bits, site)
        return alpha_bits, beta_bits & ~(1 << site), float(sign)
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin!r}")


def apply_creation(
    basis: List[State],
    state_idx: int,
    site: int,
    spin: str,
) -> Tuple[int, float]:
    r"""Apply a†_{site, spin} to a basis state and return (new_idx, sign).

    The canonical orbital ordering is::

        0_alpha, 1_alpha, ..., (n-1)_alpha, 0_beta, 1_beta, ..., (n-1)_beta

    Parameters
    ----------
    basis : list of (int, int)
    state_idx : int
    site : int
    spin : {'alpha', 'beta'}

    Returns
    -------
    new_idx : int  (−1 if zero)
    sign : float
    """
    a, b = basis[state_idx]
    na, nb, sign = _apply_creation_raw(a, b, site, spin)
    if na == -1:
        return -1, 0.0
    return _state_index(basis, (na, nb)), sign


def apply_annihilation(
    basis: List[State],
    state_idx: int,
    site: int,
    spin: str,
) -> Tuple[int, float]:
    r"""Apply a_{site, spin} to a basis state and return (new_idx, sign).

    Parameters
    ----------
    basis : list of (int, int)
    state_idx : int
    site : int
    spin : {'alpha', 'beta'}

    Returns
    -------
    new_idx : int  (−1 if zero)
    sign : float
    """
    a, b = basis[state_idx]
    na, nb, sign = _apply_annihilation_raw(a, b, site, spin)
    if na == -1:
        return -1, 0.0
    return _state_index(basis, (na, nb)), sign


# ---------------------------------------------------------------------------
# Spin operator matrices
# ---------------------------------------------------------------------------

def _build_operator_matrix(
    basis: List[State],
    action_fn,
) -> np.ndarray:
    """Build a dim×dim matrix for a one-site operator by looping over basis.

    ``action_fn(basis, state_idx)`` should return (new_idx, amplitude).
    """
    dim = len(basis)
    mat = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        new_i, amp = action_fn(basis, j)
        if new_i >= 0:
            mat[new_i, j] = amp
    return mat


def spin_plus_matrix(basis: List[State], site: int) -> np.ndarray:
    r"""Matrix representation of S_site^+ = a†_{site,alpha} a_{site,beta}.

    Acting on a basis state, S^+ flips one beta electron to alpha at *site*.
    Note: the intermediate state after applying a_{site,beta} lies in a
    different (Nα, Nβ-1) sector, so we apply both operators on the raw
    bitstrings and look up only the final state in the original basis.

    Parameters
    ----------
    basis : list of (int, int)
    site : int

    Returns
    -------
    np.ndarray, shape (dim, dim)
    """
    dim = len(basis)
    mat = np.zeros((dim, dim), dtype=complex)
    for j, (a_bits, b_bits) in enumerate(basis):
        # Step 1: annihilate beta at site (on raw bits)
        na1, nb1, sign1 = _apply_annihilation_raw(a_bits, b_bits, site, "beta")
        if na1 == -1:
            continue
        # Step 2: create alpha at site (on raw bits from step 1)
        na2, nb2, sign2 = _apply_creation_raw(na1, nb1, site, "alpha")
        if na2 == -1:
            continue
        # Look up the FINAL state in the original basis
        new_i = _state_index(basis, (na2, nb2))
        if new_i >= 0:
            mat[new_i, j] = sign1 * sign2
    return mat


def spin_minus_matrix(basis: List[State], site: int) -> np.ndarray:
    r"""Matrix representation of S_site^- = a†_{site,beta} a_{site,alpha}.

    Parameters
    ----------
    basis : list of (int, int)
    site : int

    Returns
    -------
    np.ndarray, shape (dim, dim)
    """
    dim = len(basis)
    mat = np.zeros((dim, dim), dtype=complex)
    for j, (a_bits, b_bits) in enumerate(basis):
        # Step 1: annihilate alpha at site
        na1, nb1, sign1 = _apply_annihilation_raw(a_bits, b_bits, site, "alpha")
        if na1 == -1:
            continue
        # Step 2: create beta at site
        na2, nb2, sign2 = _apply_creation_raw(na1, nb1, site, "beta")
        if na2 == -1:
            continue
        new_i = _state_index(basis, (na2, nb2))
        if new_i >= 0:
            mat[new_i, j] = sign1 * sign2
    return mat


def spin_z_matrix(basis: List[State], site: int) -> np.ndarray:
    r"""Matrix representation of S_site^Z = (1/2)(n_{site,alpha} - n_{site,beta}).

    This operator is diagonal in the occupation-number basis.

    Parameters
    ----------
    basis : list of (int, int)
    site : int

    Returns
    -------
    np.ndarray, shape (dim, dim), real diagonal matrix
    """
    dim = len(basis)
    mat = np.zeros((dim, dim), dtype=float)
    for j, (a_bits, b_bits) in enumerate(basis):
        n_alpha = (a_bits >> site) & 1
        n_beta  = (b_bits >> site) & 1
        mat[j, j] = 0.5 * (n_alpha - n_beta)
    return mat


# ---------------------------------------------------------------------------
# Commutation verification
# ---------------------------------------------------------------------------

def verify_commutations(
    basis: List[State],
    site: int,
    tol: float = 1e-10,
) -> dict:
    r"""Numerically verify the spin-1/2 algebra at a single site.

    Checks:
        [S^+, S^-] = 2 S^Z
        [S^Z, S^+] = +S^+
        [S^Z, S^-] = -S^-

    Parameters
    ----------
    basis : list of (int, int)
    site : int
    tol : float
        Tolerance for the Frobenius-norm check.

    Returns
    -------
    results : dict
        Keys are the relation names; values are dicts with keys
        ``'max_error'`` (float) and ``'passed'`` (bool).
    """
    Sp = spin_plus_matrix(basis, site)
    Sm = spin_minus_matrix(basis, site)
    Sz = spin_z_matrix(basis, site)

    relations = {
        "[S+, S-] = 2Sz": (Sp @ Sm - Sm @ Sp, 2 * Sz),
        "[Sz, S+] = +S+": (Sz @ Sp - Sp @ Sz, Sp),
        "[Sz, S-] = -S-": (Sz @ Sm - Sm @ Sz, -Sm),
    }

    results = {}
    for name, (lhs, rhs) in relations.items():
        diff = lhs - rhs
        err = np.linalg.norm(diff)
        results[name] = {"max_error": err, "passed": err < tol}
    return results


# ---------------------------------------------------------------------------
# XYZ Hamiltonian builder (prototype, not yet in HamiltonianAPI)
# ---------------------------------------------------------------------------

def build_xyz_hamiltonian(
    n_sites: int,
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    mu: np.ndarray,
    basis: List[State],
) -> np.ndarray:
    r"""Build the XYZ Heisenberg Hamiltonian matrix in the given Fock basis.

    .. math::
        \hat{H}_{XYZ} = \sum_{p \ne q}
            \frac{J^x_{pq} + J^y_{pq}}{2} S_p^+ S_q^-
          + \frac{J^x_{pq} - J^y_{pq}}{4} (S_p^+ S_q^+ + S_p^- S_q^-)
          + J^z_{pq} S_p^Z S_q^Z
          + \sum_p \mu_p S_p^Z

    When ``Jx == Jy`` the pairing (DM-like) terms vanish and the Hamiltonian
    reduces to the familiar XXZ form:

    .. math::
        \hat{H}_{XXZ} = \sum_{p \ne q}
            J^{eq}_{pq} S_p^+ S_q^-
          + J^{ax}_{pq} S_p^Z S_q^Z
          + \sum_p \mu_p S_p^Z

    with :math:`J^{eq} = J^x = J^y` and :math:`J^{ax} = J^z`.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    Jx, Jy, Jz : np.ndarray, shape (n_sites, n_sites)
        Coupling matrices (symmetric; diagonal is ignored).
    mu : np.ndarray, shape (n_sites,)
        On-site Zeeman field.
    basis : list of (int, int)
        Fock basis states returned by :func:`fock_basis`.

    Returns
    -------
    H : np.ndarray, shape (dim, dim), complex
        Full Hamiltonian matrix in the Fock basis.

    Notes
    -----
    This is a **prototype** implementation intended to demonstrate correctness
    on small systems (L ≤ 4).  The Fock-space dimension grows as
    C(L, Nα) × C(L, Nβ), which becomes prohibitively large for L > ~10.
    Production implementation will exploit sparsity and symmetry.
    """
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)

    # Pre-build spin operator matrices for all sites
    Sp = [spin_plus_matrix(basis, k)  for k in range(n_sites)]
    Sm = [spin_minus_matrix(basis, k) for k in range(n_sites)]
    SZ = [spin_z_matrix(basis, k)     for k in range(n_sites)]

    for p in range(n_sites):
        for q in range(n_sites):
            if p == q:
                continue
            j_eq   = 0.5 * (Jx[p, q] + Jy[p, q])   # hopping / flip-flop
            j_pair = 0.25 * (Jx[p, q] - Jy[p, q])   # pairing (DM-like)
            j_z    = Jz[p, q]

            H += j_eq   * (Sp[p] @ Sm[q])
            H += j_pair * (Sp[p] @ Sp[q] + Sm[p] @ Sm[q])
            H += j_z    * (SZ[p] @ SZ[q])

        # Zeeman term
        H += mu[p] * SZ[p]

    return H
