import numpy as np
from numpy.testing import assert_allclose, assert_equal
from moha import *



def test_heisenberg_0():
    r"""Test the Heisenberg model 0 electron integral."""
    n_sites = 8
    J_xy = np.random.rand()
    J_z = np.random.rand()
    connectivity = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0
        connectivity[i, j] = 1
        connectivity[j, i] = 1

    ham = HamHeisenberg(J_eq=J_xy, J_ax=J_z, mu=0, connectivity=connectivity)
    e0 = ham.generate_zero_body_integral()
    assert_allclose(e0, J_z/4*n_sites)


def test_heisenberg_1():
    r"""Test the Heisenberg model 1 electron integral."""
    n_sites = 8
    J_xy = np.random.rand()
    J_z = np.random.rand()
    connectivity = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0
        connectivity[i, j] = 1
        connectivity[j, i] = 1

    h_exact = np.zeros((n_sites, n_sites))

    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0

        h_exact[i, i] = -J_z/2

    ham = HamHeisenberg(J_eq=J_xy, J_ax=J_z, mu=0, connectivity=connectivity)
    h = ham.generate_one_body_integral(basis='spatial basis', dense=True)
    assert_allclose(h, h_exact)


def test_heisenberg_2():
    r"""Test the Heisenberg model 2 electron integral."""
    n_sites = 8
    J_xy = np.random.rand()
    J_z = np.random.rand()
    connectivity = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0
        connectivity[i, j] = 1
        connectivity[j, i] = 1

    v_exact = np.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0

        v_exact[j, j, i, i] = J_z/4
        v_exact[i, i, j, j] = J_z/4

        v_exact[j, i, j, i] = J_xy/2
        v_exact[i, j, i, j] = J_xy/2

    ham = HamHeisenberg(J_eq=J_xy, J_ax=J_z, mu=0, connectivity=connectivity)
    v = ham.generate_two_body_integral(basis='spatial basis',
                                       dense=True,
                                       sym=4)
    # convert to chemists notation
    v = np.transpose(v, (0, 2, 1, 3))
    assert_allclose(v, v_exact)


def test_heisenberg_2_spin():
    r"""Test the Heisenberg model 2 electron integral."""
    n_sites = 8
    J_xy = np.random.rand()
    J_z = np.random.rand()
    connectivity = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0
        connectivity[i, j] = 1
        connectivity[j, i] = 1

    v_exact = np.zeros((2*n_sites, 2*n_sites, 2*n_sites, 2*n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0

        v_exact[j, j, i, i] = J_z/4
        v_exact[i, i, j, j] = J_z/4

        v_exact[j, j, i+n_sites, i+n_sites] = J_z/4
        v_exact[i, i, j+n_sites, j+n_sites] = J_z/4

        v_exact[j+n_sites, j+n_sites, i, i] = J_z/4
        v_exact[i+n_sites, i+n_sites, j, j] = J_z/4

        v_exact[j+n_sites, j+n_sites, i+n_sites, i+n_sites] = J_z/4
        v_exact[i+n_sites, i+n_sites, j+n_sites, j+n_sites] = J_z/4

        v_exact[j, i, j+n_sites, i+n_sites] = J_xy
        v_exact[i, j, i+n_sites, j+n_sites] = J_xy
        v_exact[j+n_sites, i+n_sites, j, i] = J_xy
        v_exact[i+n_sites, j+n_sites, i, j] = J_xy

    # converting to physics notation
    v_exact = np.transpose(v_exact, (0, 2, 1, 3))

    ham = HamHeisenberg(J_eq=J_xy,
                        J_ax=J_z,
                        mu=0,
                        connectivity=connectivity)
    v = ham.generate_two_body_integral(basis='spinorbital basis',
                                       dense=True,
                                       sym=4)

    inds = np.nonzero(v)
    for i, j, k, l in zip(*inds):
        print(i, j, k, l, v[i, j, k, l], v_exact[i, j, k, l])
    print("DEBUG: v type in test_heisenberg_2_spin:", type(v))
    
    assert_allclose(v, v_exact)


def test_Ising():
    n_sites = 8
    J_xy = 0
    J_z = np.random.rand()
    connectivity = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0
        connectivity[i, j] = 1
        connectivity[j, i] = 1

    v_exact = np.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0

        v_exact[j, j, i, i] = J_z/4
        v_exact[i, i, j, j] = J_z/4

        v_exact[j, i, j, i] = J_xy/2
        v_exact[i, j, i, j] = J_xy/2

    ham = HamIsing(J_ax=J_z, mu=0, connectivity=connectivity)
    v = ham.generate_two_body_integral(basis='spatial basis',
                                       dense=True,
                                       sym=4)
    # convert to chemists notation
    v = np.transpose(v, (0, 2, 1, 3))
    assert_allclose(v, v_exact)


def test_RG():
    n_sites = 8
    J_xy = np.random.rand()
    J_z = 0
    connectivity = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0
        connectivity[i, j] = 1
        connectivity[j, i] = 1

    v_exact = np.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites):
        j = i+1
        if j == n_sites:
            j = 0

        v_exact[j, j, i, i] = J_z/4
        v_exact[i, i, j, j] = J_z/4

        v_exact[j, i, j, i] = J_xy/2
        v_exact[i, j, i, j] = J_xy/2

    ham = HamRG(J_eq=J_xy, mu=0, connectivity=connectivity)
    v = ham.generate_two_body_integral(basis='spatial basis',
                                       dense=True,
                                       sym=4)
    # convert to chemists notation
    v = np.transpose(v, (0, 2, 1, 3))
    assert_allclose(v, v_exact)
