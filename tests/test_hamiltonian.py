"""Tests for Hamiltonians."""

import numpy as np

from moha.hamiltonian import PPPHamiltonian
from moha.lattice import Lattice


def test_ppp():
    beta = np.array([[0,1,1,1,1],[1,0,1,1,0],[1,1,0,0,1],[1,1,0,0,1],[1,0,1,1,0]])
    test = PPPHamiltonian(Lattice.linear(5), 2, np.random.random((5,)), beta, np.arange(5)+1, np.random.random((5,5)))
    # Spot checks for proper conversion
    assert np.allclose(test.n_matrix[26:30, :, 0], [[1, 0, 1, 0, 0],[0,1,1,0,0],[0,0,0,1,0],[0,0,0,1,0]])
    # Check that 10 [5 choose 2] on_down vectors are empty
    assert len([x for x in test.on_down if x == 0]) == 10
    assert len([x for x in test.on_up if x == 0]) == 10
    # Check that every state has 2 electrons
    assert np.array_equal(test.n_matrix.sum(2).sum(1), np.array([2 for _ in range(45)]))
    # Check that every element of u_matrix only appears once
    # FIXME: there is a better way to do this
    assert np.sum(test.H_interaction) == np.sum(test.u_matrix)


def test_short():
    test = PPPHamiltonian(Lattice.linear(3), 1, np.arange(3) + 1.1, np.arange(9).reshape(3, 3) + 1.1,
                          np.arange(3) + 1.1, np.arange(9).reshape(3, 3) + 1.1)
    print(test.big_n)


def test_hopping():
    test = PPPHamiltonian(Lattice.linear(3), 1, np.arange(3)+1.1, np.arange(9).reshape(3,3)+1.1, np.arange(3)+1.1, np.arange(9).reshape(3,3)+1.1)
    # Check to make sure there aren't spin flips
    assert np.sum(test.H_hopping[:3, 3:]) == 0
    assert np.sum(test.H_hopping[3:, :3]) == 0
    # Check to make sure symmetry holds
    assert np.allclose(test.H_hopping[:3, :3], test.H_hopping[3:, 3:])


def test_memory():
    n = 15
    test = PPPHamiltonian(Lattice.linear(n), 2, np.random.random((n,)), np.random.random((n, n)), np.arange(n) + 1,
                          np.random.random((n, n)))



