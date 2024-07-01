"""Testing file."""

import numpy as np
from moha import *
from numpy.testing import assert_allclose, assert_equal


def test_hub2():
    r"""
    2 site hubbard model with 2 electrons.

    Should return U=\frac{1}{2}\left[U-\sqrt{U^{2}+16 t^{2}}\right]$
    numerical result is -1.561552812
    """
    hubbard = HamHub([("C1", "C2", 1)],
                     alpha=0, beta=-1, u_onsite=np.array([1, 1]), sym=1)

    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(basis='spatial basis', dense=True)
    v = hubbard.generate_two_body_integral(sym=1,
                                           basis='spatial basis',
                                           dense=True)

    # constructing the exact two body integral
    v_true = np.zeros((2, 2, 2, 2,))
    v_true[0, 0, 0, 0] = 1
    v_true[1, 1, 1, 1] = 1

    assert_equal(h, np.array([[0., -1.],
                              [-1., 0.]]))
    assert_equal(v, v_true)


def test_hub4():
    r"""
    4 site hubbard model with periodic boundary conditions.

    The exact energy is Lieb-Wu equation:
    $\frac{E(U, d=1)}{t N_{s}}=-4 \int_{0}^{\infty} d x
    \frac{J_{0}(x) J_{1}(x)}{x[1+\exp (U x / 2)]}$
    """
    nsites = np.linspace(2, 8, 4).astype(int)
    for nsite in nsites:
        nelec = nsite // 2
        hubbard = HamPPP([(f"C{i}",
                           f"C{i + 1}",
                           1) for i in range(1,
                                             nsite)] + [(f"C{nsite}",
                                                         f"C{1}",
                                                         1)],
                         alpha=0,
                         beta=-1,
                         u_onsite=np.array([1 for i in range(nsite)]),
                         gamma=np.zeros((nsite,
                                         nsite)))
        ecore = hubbard.generate_zero_body_integral()
        h = hubbard.generate_one_body_integral(basis='spatial basis',
                                               dense=True)
        v = hubbard.generate_two_body_integral(sym=1,
                                               basis='spatial basis',
                                               dense=True)

        # constructing the exact one body integral
        h_true = np.diag(-1 * np.ones(nsite - 1), k=1)
        h_true[0, -1] = -1
        h_true += h_true.T
        assert_equal(h, h_true)

        # constructing the exact two body integral
        v_true = np.zeros((nsite, nsite, nsite, nsite))
        for i in range(nsite):
            v_true[i, i, i, i] = 1
        assert_equal(v, v_true)


def test_ethylene():
    r"""
    Ethylene Huckel model.

    $E_0 = 2 (\alpha + \beta)$
    """
    a = -11.26
    b = -1.45
    hubbard = HamPPP([("C1", "C2", 1)], alpha=a, beta=b, gamma=np.zeros(
        (2, 2)), charges=None, sym=None, u_onsite=[0, 0])
    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(basis='spinorbital basis',
                                           dense=True)
    v = hubbard.generate_two_body_integral(sym=1,
                                           basis='spinorbital basis',
                                           dense=True)

    test = np.zeros_like(h)
    test[:2, :2] = np.array([[a, b], [b, a]])
    test[2:, 2:] = np.array([[a, b], [b, a]])
    assert np.allclose(h, test)

    assert v.shape[0] == 4
    # assert np.allclose(v, test)
    assert ecore == 0.

    h = hubbard.generate_one_body_integral(basis='spatial basis', dense=True)
    v = hubbard.to_spatial(sym=1, dense=True, nbody=2)

    assert_allclose(v, np.zeros((2, 2, 2, 2)))
    assert np.allclose(h, np.array([[a, b], [b, a]]))


def test_4():
    r"""
    Cyclobutadiene, 4 site Huckel model with periodic boundary conditions.

    $E_0 = 2 (\alpha + 2 \beta) + 2 \alpha$
    """
    a = -5
    b = -0.5
    hubbard = HamPPP([("C1", "C2", 1),
                      ("C2", "C3", 1),
                      ("C3", "C4", 1),
                      ("C4", "C1", 1)],
                     alpha=a, beta=b)
    atoms_sites_lst, _ = hubbard.generate_connectivity_matrix()

    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(basis='spatial basis', dense=True)
    v = hubbard.generate_two_body_integral(sym=1,
                                           basis='spinorbital basis',
                                           dense=True)

    assert v.shape[0] == 8
    assert np.allclose(h, np.array([[a, b, 0., b],
                                    [b, a, b, 0.],
                                    [0., b, a, b],
                                    [b, 0., b, a]]))
    v = hubbard.to_spatial(sym=1, dense=True, nbody=2)

    assert v.shape[0] == 4


def test_ppp_api():
    r"""Six sites PPP model."""
    norb = 6
    connectivity = [("C1", "C2", 1),
                    ("C2", "C3", 1),
                    ("C3", "C4", 1),
                    ("C4", "C5", 1),
                    ("C5", "C6", 1),
                    ("C6", "C1", 1)]
    u_matrix = np.ones(norb)
    g_matrix = np.arange(36).reshape((norb, norb))
    charges = np.ones(norb)

    ham = HamPPP(connectivity, alpha=0., beta=-2.5, u_onsite=u_matrix,
                 gamma=g_matrix, charges=charges)
    h = ham.generate_one_body_integral(basis='spatial basis', dense=True)
    v = ham.generate_two_body_integral(sym=1,
                                       basis='spatial basis',
                                       dense=True)

    assert h.shape[0] == 6
    assert v.shape[0] == 6


def test_api_input():
    r"""Test the input of the API."""
    norb = 6
    connectivity = np.zeros((norb, norb))
    for i in range(norb - 1):
        connectivity[i, i + 1] = 1
        connectivity[i + 1, i] = 1
    connectivity[-1, 0] = 1

    u_matrix = np.ones(norb)
    g_matrix = np.arange(36).reshape((norb, norb))
    charges = np.ones(norb)

    ham = HamPPP(connectivity, alpha=0., beta=-2.5, u_onsite=u_matrix,
                 gamma=g_matrix, charges=charges)
    h = ham.generate_one_body_integral(basis='spinorbital basis', dense=True)
    v = ham.generate_two_body_integral(sym=1,
                                       basis='spinorbital basis',
                                       dense=True)

    assert h.shape[0] == 12
    assert v.shape[0] == 12


def test_spin_spatial_conversion():
    r"""Test the conversion between spin and spatial basis."""
    norb = 4
    connectivity = np.zeros((norb, norb))
    for i in range(norb - 1):
        connectivity[i, i + 1] = 1
        connectivity[i + 1, i] = 1
    connectivity[-1, 0] = 1
    connectivity[0, -1] = 1

    u_matrix = np.ones(norb)

    beta = -2.5
    ham = HamHub(connectivity, alpha=0., beta=beta, u_onsite=u_matrix)
    h = ham.generate_one_body_integral(basis='spinorbital basis', dense=True)
    v = ham.generate_two_body_integral(sym=4,
                                       basis='spinorbital basis',
                                       dense=True)

    h = ham.to_spatial(sym=1, dense=True, nbody=1)
    v = ham.to_spatial(sym=4, dense=True, nbody=2)

    h1 = np.array([[0., beta, 0., beta],
                   [beta, 0., beta, 0.],
                   [0., beta, 0., beta],
                   [beta, 0., beta, 0.]])
    h2 = np.zeros((4, 4, 4, 4))
    for i in range(4):
        h2[i, i, i, i] = 1
    np.testing.assert_allclose(h, h1)
    np.testing.assert_allclose(v, h2)
