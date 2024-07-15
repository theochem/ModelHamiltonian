"""Testing t-JUV model."""

import numpy as np
from moha import *
from numpy.testing import assert_allclose, assert_equal


def test_tjuv_consistency_zero_body():
    r"""
    Checking consistency of TJUV model
    with Heisenberg and PPP model.
    """
    connectivity = np.array([[0, 1, 0, 0, 0, 1],
                            [1, 0, 1, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0],
                            [0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 0]])
    alpha = 0.0
    beta = -1.0
    u_onsite = np.array([1, 1, 1, 1, 1, 1])
    gamma = None
    charges = 1
    sym = 8
    J_eq = 1
    J_ax = 1

    # Initialize the HamTJUV object
    tjuv_hamiltonian = HamTJUV(connectivity=connectivity,
                               alpha=alpha,
                               beta=beta,
                               u_onsite=u_onsite,
                               gamma=gamma,
                               charges=charges,
                               sym=sym,
                               J_eq=J_eq,
                               J_ax=J_ax)

    # Generate the zero body integral
    tjuv_zero = tjuv_hamiltonian.generate_zero_body_integral()

    heisenberg = HamHeisenberg(
        J_eq=J_eq,
        J_ax=J_ax,
        mu=0,
        connectivity=connectivity)
    heisenberg_zero = heisenberg.generate_zero_body_integral()
    hpp = HamPPP(
        connectivity=connectivity,
        alpha=alpha,
        beta=beta,
        gamma=np.zeros(
            (2,
             2)),
        charges=None,
        sym=None,
        u_onsite=[
            1,
            1])
    hpp_zero = hpp.generate_zero_body_integral()
    assert_allclose(tjuv_zero, heisenberg_zero + hpp_zero)


def test_tjuv_consistency_one_body():
    r"""
    Checking consistency of TJUV model
    with Heisenberg and PPP model for one-body term.
    """
    connectivity = np.array([[0, 1, 0, 0, 0, 1],
                             [1, 0, 1, 0, 0, 0],
                             [0, 1, 0, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1],
                             [1, 0, 0, 0, 1, 0]])
    alpha = 0.0
    beta = -1.0
    u_onsite = np.array([1, 1, 1, 1, 1, 1])
    gamma = None
    charges = 1
    sym = 8
    J_eq = 1
    J_ax = 1

    # Initialize the HamTJUV object
    tjuv_hamiltonian = HamTJUV(connectivity=connectivity,
                               alpha=alpha,
                               beta=beta,
                               u_onsite=u_onsite,
                               gamma=gamma,
                               charges=charges,
                               sym=sym,
                               J_eq=J_eq,
                               J_ax=J_ax,)

    # Generate the one-body integral
    tjuv_one_body = tjuv_hamiltonian.generate_one_body_integral(
        basis='spatial basis', dense=True)

    heisenberg = HamHeisenberg(
        J_eq=J_eq,
        J_ax=J_ax,
        mu=0,
        connectivity=connectivity)
    heisenberg_one_body = heisenberg.generate_one_body_integral(
        basis='spatial basis', dense=True)

    hpp = HamPPP(
        connectivity=connectivity,
        alpha=alpha,
        beta=beta,
        gamma=np.zeros(
            (2,
             2)),
        charges=None,
        sym=None,
        u_onsite=u_onsite)
    hpp_one_body = hpp.generate_one_body_integral(
        basis='spatial basis', dense=True)

    # Assert that the TJUV one-body integral is close to the sum of Heisenberg
    # and PPP one-body integrals
    assert_allclose(tjuv_one_body, heisenberg_one_body + hpp_one_body)


def test_tjuv_consistency_two_body():
    r"""
    Checking consistency of TJUV model
    with Heisenberg and PPP model for two-body term.
    """
    connectivity = np.array([[0, 1, 0, 0, 0, 1],
                             [1, 0, 1, 0, 0, 0],
                             [0, 1, 0, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1],
                             [1, 0, 0, 0, 1, 0]])
    alpha = 0.0
    beta = -1.0
    u_onsite = np.array([1, 1, 1, 1, 1, 1])
    gamma = None
    charges = 1
    sym = 8
    J_eq = 1
    J_ax = 1

    # Initialize the HamTJUV object
    tjuv_hamiltonian = HamTJUV(connectivity=connectivity,
                               alpha=alpha,
                               beta=beta,
                               u_onsite=u_onsite,
                               gamma=gamma,
                               charges=charges,
                               sym=sym,
                               J_eq=J_eq,
                               J_ax=J_ax,)

    # Generate the one-body integral
    tjuv_one_body = tjuv_hamiltonian.generate_two_body_integral(
        basis='spinorbital basis', dense=True)

    heisenberg = HamHeisenberg(
        J_eq=J_eq,
        J_ax=J_ax,
        mu=0,
        connectivity=connectivity)
    heisenberg_one_body = heisenberg.generate_two_body_integral(
        basis='spinorbital basis', dense=True)

    hpp = HamPPP(
        connectivity=connectivity,
        alpha=alpha,
        beta=beta,
        gamma=None,
        charges=None,
        sym=None,
        u_onsite=u_onsite)
    hpp_one_body = hpp.generate_two_body_integral(
        basis='spinorbital basis', dense=True)

    # Assert that the TJUV one-body integral is close to the sum of Heisenberg
    # and PPP one-body integrals
    assert_allclose(tjuv_one_body, heisenberg_one_body + hpp_one_body)


def test_tjuv_energy_spectrum():
    # Define parameters for a simple 1D chain
    N = 4
    t = 1.0
    alpha = -t
    beta = 0.0
    u_onsite = np.zeros(N)
    gamma = None
    charges = None
    sym = None
    J_eq = None
    J_ax = None

    # Connectivity matrix for a 1D chain with periodic boundary conditions
    connectivity = np.zeros((N, N))
    for i in range(N):
        connectivity[i, (i + 1) % N] = 1
        connectivity[(i + 1) % N, i] = 1

    # Initialize the HamTJUV object
    tjuv_hamiltonian = HamTJUV(connectivity=connectivity,
                               alpha=alpha,
                               beta=beta,
                               u_onsite=u_onsite,
                               gamma=gamma,
                               charges=charges,
                               sym=sym,
                               J_eq=J_eq,
                               J_ax=J_ax)

    # Generate the one-body integral (Hamiltonian matrix)
    tjuv_one_body = tjuv_hamiltonian.generate_one_body_integral()

    # Calculate the eigenvalues (energy spectrum)
    energies, _ = np.linalg.eigh(tjuv_one_body)

    # Analytical energy levels for comparison
    k_vals = np.arange(N)
    analytical_energies = -2 * t * np.cos(2 * np.pi * k_vals / N)

    # Sort the energies for comparison
    energies_sorted = np.sort(energies)
    analytical_energies_sorted = np.sort(analytical_energies)

    # Assert that the numerical and analytical energies are close
    np.testing.assert_allclose(
        energies_sorted,
        analytical_energies_sorted,
        atol=1e-8)
