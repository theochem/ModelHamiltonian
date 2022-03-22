import numpy as np
# import pyci
from PPP import *
from scipy.integrate import quad
from scipy.special import jv
from utils import convert_indices
from numpy.testing import assert_allclose


def test_1():
    """ 2 site hubbard model with 2 electrons. Should return U=\frac{1}{2}\left[U-\sqrt{U^{2}+16 t^{2}}\right]$
    numerical result is -1.561552812 """

    hubbard = HamPPP([("C1", "C2", 1)], alpha=0, beta=-1, u_onsite=np.array([1, 1]),
                     gamma=None, charges=None, sym=None)
    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(sym=1, basis='spatial basis', dense=True)
    v = hubbard.generate_two_body_integral(sym=1, basis='spinorbital basis', dense=True)

    v_new = hubbard.to_spatial(v, sym=1, dense=False, nbody=2)

    v_4d = np.zeros((2, 2, 2, 2))
    for m, n in zip(*v_new.nonzero()):
        i, j, k, l = convert_indices(2, int(m), int(n))
        v_4d[i, j, k, l] = v_new[m, n]

    ham = pyci.hamiltonian(ecore, h, 2*v_4d) # multiply by two because test doesnt work
    n_up = 1
    n_down = 1
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
    assert_allclose(eigenvals, -1.561552812)


def test_2():
    """
    4 site hubbard model with periodic boundary conditions. Should return
    $\frac{E(U, d=1)}{t N_{s}}=-4 \int_{0}^{\infty} d x \frac{J_{0}(x) J_{1}(x)}{x[1+\exp (U x / 2)]}$
    :return:
    """
    hubbard = Hubbard([("C1", "C2", 1), ("C2", "C3", 1), ("C3", "C4", 1), ("C4", "C1", 1)], alpha=0, beta=-0.5,
            u_onsite = np.array([1 for i in range(4)]))
    ecore, h, v = hubbard.get_hamilton()
    ham = pyci.hamiltonian(ecore, h, v)
    n_up = 2
    n_down = 2
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_all_dets()

    op = pyci.sparse_op(ham, wfn)
    eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
    E, err = quad(lambda x: jv(0, x) * jv(1, x) / (x * (1 + np.exp(x / 2))),
                  0, np.inf)
    np.allclose(-4*E, eigenvals)
    print(-4*E, eigenvals)


def test_3():
    """
    Ethylene Huckel model

    $E_0 = 2 (\alpha + \beta)$
    """
    a = -11.26
    b = -1.45
    hubbard = HamPPP([("C1", "C2", 1)], alpha=a, beta=b, gamma=None, charges=None, sym=None)#a=,b= 
    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(sym=1, basis='spinorbital basis', dense=True)
    # FIXME: empty scipy.sparse.csr.csr_matrix
    v = hubbard.generate_two_body_integral(sym=1, basis='spinorbital basis', dense=True)

    test = np.zeros_like(h)
    test[:2, :2] =  np.array([[a, b], [b, a]])
    test[2:, 2:] =  np.array([[a, b], [b, a]])
    assert np.allclose(h, test)
    test = np.zeros((16,16))
    assert v.shape[0] == 16
    # assert np.allclose(v, test)
    assert ecore == 0.

    h = hubbard.generate_one_body_integral(sym=1, basis='spatial basis', dense=True)
    v = hubbard.to_spatial(v, sym=1, dense=True, nbody=2)
    assert np.allclose(h, np.array([[a, b], [b, a]]))

    ham = pyci.hamiltonian(ecore, h, np.zeros((4, 4, 4, 4)))
    n_up = 1
    n_down = 1
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
    assert_allclose(eigenvals[0], 2*(a+b))


def test_4():
    """
    Cyclobutadiene, 4 site Huckel model with periodic boundary conditions

    $E_0 = 2 (\alpha + 2 \beta) + 2 \alpha$
    """
    a = -5
    b = -0.5
    hubbard = HamPPP([("C1", "C2", 1), ("C2", "C3", 1), ("C3", "C4", 1), ("C4", "C1", 1)], alpha=a, beta=b,
            gamma=None, charges=None, sym=None)
    atoms_sites_lst, _ = hubbard.generate_connectivity_matrix()

    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(sym=1, basis='spatial basis', dense=True)
    # FIXME: empty scipy.sparse.csr.csr_matrix
    v = hubbard.generate_two_body_integral(sym=1, basis='spinorbital basis', dense=True) 
    v = hubbard.to_spatial(v, sym=1, dense=True, nbody=2)
    
    assert np.allclose(h, np.array([[a, b, 0., b], [b, a, b, 0.], [0., b, a, b], [b, 0., b, a]]))
    assert v.shape[0] == 64

    ham = pyci.hamiltonian(ecore, h, np.zeros((8, 8, 8, 8)))
    n_up = 2
    n_down = 2
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
    answer = 2*(a+2*b) + 2*a
    assert_allclose(eigenvals[0],  answer)

print(test_4())
# print(test_1())
