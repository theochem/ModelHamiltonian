"""Testing the MolHam class."""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from moha.molkit import utils
from moha.molkit.hamiltonians import MolHam


def test_antisymmetrize():
    """Test antisymmetrization of two-electron integrals."""
    one_body = np.eye(2)
    two_body = np.zeros((2, 2, 2, 2))
    two_body[0, 0, 1, 1] = 1.0
    two_body[1, 1, 0, 0] = 1.0

    mol_ham = MolHam(one_body=one_body, two_body=two_body)
    mol_ham.spinize_H()
    mol_ham.antisymmetrize()

    expected_two_body_aa = np.zeros((2, 2, 2, 2))
    expected_two_body_aa[0, 0, 1, 1] = 1
    expected_two_body_aa[1, 1, 0, 0] = 1
    expected_two_body_aa[0, 1, 0, 1] = 0
    expected_two_body_aa[1, 0, 1, 0] = 0

    assert_allclose(mol_ham.two_body[:2, :2, :2, :2], expected_two_body_aa)

def test_to_geminal():
    """Test conversion to geminal basis."""
    n_orb = 4
    two_body = np.zeros((2, 2, 2, 2))
    two_body[0, 0, 1, 1] = 1.0
    two_body[1, 1, 0, 0] = 1.0
    two_body[1, 0, 0, 1] = 1.0
    two_body[0, 1, 1, 0] = 1.0

    geminal_true = np.array([
        [-2.0]
    ])
    geminal = utils.tools.to_geminal(two_body, type='h2')
    assert_equal(geminal.shape, (1, 1))
    assert_allclose(geminal, geminal_true)


def test_to_reduced_ham():
    """Test conversion to reduced Hamiltonian."""
    n_orb = 2
    one_body = np.eye(n_orb)
    two_body = np.zeros((n_orb, n_orb, n_orb, n_orb))
    two_body[0, 0, 1, 1] = 1.0
    two_body[1, 1, 0, 0] = 1.0

    mol_ham = MolHam(one_body=one_body, two_body=two_body)
    reduced_ham = mol_ham.to_reduced(n_elec=2)


    # sum over the spin-orbital indices
    reduced_ham = reduced_ham[:2, :2, :2, :2] +\
                  reduced_ham[2:, 2:, 2:, 2:] +\
                  reduced_ham[:2, 2:, :2, 2:] +\
                  reduced_ham[2:, :2, 2:, :2]
    reduced_ham *= 0.25

    reduced_ham_true = 0.5 * two_body
    reduced_ham_true[0, 0, 0, 0] = 1
    reduced_ham_true[1, 1, 1, 1] = 1
    reduced_ham_true[0, 1, 0, 1] = 1
    reduced_ham_true[1, 0, 1, 0] = 1

    assert_allclose(reduced_ham, reduced_ham_true)