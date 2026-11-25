"""Test the spatial integrals of the PPP Hamiltonian against exact diagonalization."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from moha import HamPPP


def _annihilators(n_orb):
    """Jordan-Wigner annihilation operators, alpha spin orbitals first."""
    n_so = 2 * n_orb
    ident, z, a = np.eye(2), np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = []
    for k in range(n_so):
        mat = np.eye(1)
        for factor in [z] * k + [a] + [ident] * (n_so - k - 1):
            mat = np.kron(mat, factor)
        ops.append(mat)
    return ops


def _lowest_energy(n_orb, nalpha, nbeta, e0, h_spatial, v_spatial):
    """Lowest energy of e0 + sum h_pq a+_p a_q + sum v_pqrs a+_p a+_q a_s a_r."""
    ann = _annihilators(n_orb)
    cre = [op.T for op in ann]
    dim = 2 ** (2 * n_orb)
    ham = e0 * np.eye(dim)
    for p, q in product(range(n_orb), repeat=2):
        for spin in (0, n_orb):
            ham = ham + h_spatial[p, q] * cre[p + spin] @ ann[q + spin]
    for p, q, r, s in product(range(n_orb), repeat=4):
        coeff = v_spatial[p, q, r, s]
        if coeff == 0:
            continue
        for spin1, spin2 in product((0, n_orb), repeat=2):
            pair = cre[p + spin1] @ cre[q + spin2]
            ham = ham + coeff * pair @ ann[s + spin2] @ ann[r + spin1]
    num_a = sum(cre[k] @ ann[k] for k in range(n_orb))
    num_b = sum(cre[k + n_orb] @ ann[k + n_orb] for k in range(n_orb))
    sector = np.where(
        (np.diag(num_a).round() == nalpha) & (np.diag(num_b).round() == nbeta)
    )[0]
    return np.linalg.eigvalsh(ham[np.ix_(sector, sector)])[0]


def _ppp_ground_state(n_orb, nalpha, nbeta, h, u_onsite, gamma, charges):
    """Lowest energy of the PPP Hamiltonian, built directly from its definition."""
    ann = _annihilators(n_orb)
    cre = [op.T for op in ann]
    dim = 2 ** (2 * n_orb)
    dens = [cre[k] @ ann[k] + cre[k + n_orb] @ ann[k + n_orb] for k in range(n_orb)]
    ham = np.zeros((dim, dim))
    for p, q in product(range(n_orb), repeat=2):
        for spin in (0, n_orb):
            ham += h[p, q] * cre[p + spin] @ ann[q + spin]
    for p in range(n_orb):
        ham += u_onsite[p] * (cre[p] @ ann[p]) @ (cre[p + n_orb] @ ann[p + n_orb])
    for p, q in product(range(n_orb), repeat=2):
        if p != q:
            ham += 0.5 * gamma[p, q] * (dens[p] - charges[p] * np.eye(dim)) @ (
                dens[q] - charges[q] * np.eye(dim)
            )
    num_a = sum(cre[k] @ ann[k] for k in range(n_orb))
    num_b = sum(cre[k + n_orb] @ ann[k + n_orb] for k in range(n_orb))
    sector = np.where(
        (np.diag(num_a).round() == nalpha) & (np.diag(num_b).round() == nbeta)
    )[0]
    return np.linalg.eigvalsh(ham[np.ix_(sector, sector)])[0]


def _make_ppp(sym):
    connectivity = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    u_onsite = np.array([2.0, 3.0, 1.0])
    gamma = np.array([[0.0, 0.8, 0.3], [0.8, 0.0, 0.5], [0.3, 0.5, 0.0]])
    charges = np.array([1.0, 1.0, 1.0])
    ham = HamPPP(
        connectivity,
        alpha=0.1,
        beta=-1.0,
        u_onsite=u_onsite,
        gamma=gamma,
        charges=charges,
        sym=sym,
    )
    h = np.asarray(ham.generate_one_body_integral("spatial basis", True))
    v = np.asarray(ham.generate_two_body_integral("spatial basis", True, sym=sym))
    e0 = ham.generate_zero_body_integral()
    ref_h = 0.1 * np.eye(3) - connectivity
    return h, v.reshape(3, 3, 3, 3), e0, (ref_h, u_onsite, gamma, charges)


@pytest.mark.parametrize("sym", [1, 2, 4, 8])
@pytest.mark.parametrize("nelec", [(1, 1), (2, 1), (2, 2)])
def test_ppp_spatial_integrals_reproduce_exact_energy(sym, nelec):
    """The spatial integrals must give the PPP energy for every value of sym."""
    h, v, e0, (ref_h, u_onsite, gamma, charges) = _make_ppp(sym)
    expected = _ppp_ground_state(3, *nelec, ref_h, u_onsite, gamma, charges)
    energy = _lowest_energy(3, *nelec, e0, h, v)
    assert_allclose(energy, expected, atol=1e-10)
