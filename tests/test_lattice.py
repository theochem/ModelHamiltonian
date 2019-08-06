"""Test lattice module."""

import numpy as np
import pytest

from moha.lattice import Lattice, LatticeSite


def test_lattice_site():
    test = LatticeSite(1, np.array([0., 0., 0.]))
    assert test.number == 1
    assert np.allclose(test.coords, np.array([0., 0., 0.]))
    for i in [2, 4, 5]:
        test.neighbours.append(i)
    assert len(test.neighbours) == 3
    assert test.neighbours == [2, 4, 5]

    with pytest.raises(TypeError):
        bad_index = LatticeSite(1.1, np.array([0., 0., 0.]))
    with pytest.raises(TypeError):
        bad_coords = LatticeSite(1, [3, 4, 5])
    with pytest.raises(ValueError):
        too_many_coords = LatticeSite(1, np.array([0., 0., 0., 0.]))


def test_linear():
    test = Lattice.linear(5)
    assert len(test.sites) == 5
    assert test.n_sites == 5
    assert np.allclose(test.adjacency_matrix, np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]]))
    print([site.coords for site in test.sites])


def test_rectangular():
    test = Lattice.rectangular(n_sites=(3, 4), dist=(1., 1.), axis=(0, 1))
    print(test.adjacency_matrix)
    print(test.adjacency_matrix.sum(1))


def test_oblique():
    test = Lattice.oblique(n_sites=(3, 4), dist=(1., np.sqrt(2)), axis=(0, 1), angle=(45 * np.pi / 180))
    print([site.coords for site in test.sites])


def test_orthorhombic():
    test = Lattice.orthorhombic(n_sites=(2, 4, 3), dist=(1., 1., 1.,))
    print(test.adjacency_matrix.sum(1))

