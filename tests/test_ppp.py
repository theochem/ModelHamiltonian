"""Test PPP Hamiltonian and subclasses."""

import numpy as np
from scipy.sparse import dok_matrix

from moha.ppp import HuckelHamiltonian


def test_huckel():
    test = HuckelHamiltonian(np.array([1, 2, 3]), np.arange(9).reshape(3,3)+0.1-np.diag([0.1, 4.1, 8.1]))
    # Test generation of compact matrices
    assert np.allclose(test.one_electron_matrix, np.array([[1, 1.1, 2.1],[3.1, 2, 5.1],[6.1, 7.1, 3]]))
    assert not test.two_electron_matrix.keys()
    # Test sparse outputs
    sparse_one, sparse_two = test.return_compact()
    assert isinstance(sparse_one, np.ndarray)
    assert isinstance(sparse_two, dok_matrix)
    assert np.allclose(sparse_one, np.array([[1, 1.1, 2.1], [3.1, 2, 5.1], [6.1, 7.1, 3]]))
    assert sparse_two.count_nonzero() == 0
    # Test dense outputs
    dense_one, dense_two = test.return_full()
    assert isinstance(dense_one, np.ndarray)
    assert isinstance(dense_two, np.ndarray)
    assert np.allclose(dense_one, np.array([[1, 1.1, 2.1], [3.1, 2, 5.1], [6.1, 7.1, 3]]))
    assert np.allclose(dense_two, np.zeros((3,3)))



