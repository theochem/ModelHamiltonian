import numpy as np
import pyci
from PPP import *
from scipy.integrate import quad
from scipy.special import jv
from numpy.testing import assert_allclose


def test_1():
    """ 2 site hubbard model with 2 electrons. Should return U=\frac{1}{2}\left[U-\sqrt{U^{2}+16 t^{2}}\right]$
    numerical result is -1.561552812 """

    hubbard = HamPPP([("C1", "C2", 1)], alpha=0, beta=-1, u_onsite=np.array([1, 1]), sym=1)
    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(sym=1, basis='spatial basis', dense=True)
    v = hubbard.generate_two_body_integral(sym=1, basis='spatial basis', dense=True)

    assert v.shape[0] == 2

    ham = pyci.hamiltonian(ecore, h, 2*v)  # multiply by two because test doesn't work
    n_up = 1
    n_down = 1
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
    assert_allclose(eigenvals, -1.561552812)


def test_2():
    """
    4 site hubbard model with periodic boundary conditions. The exact energy is Lieb-Wu equation:
    $\frac{E(U, d=1)}{t N_{s}}=-4 \int_{0}^{\infty} d x \frac{J_{0}(x) J_{1}(x)}{x[1+\exp (U x / 2)]}$
    """
    nsites = np.linspace(2, 8, 4).astype(int)
    for nsite in nsites:
        nelec = nsite//2
        hubbard = HamPPP([(f"C{i}", f"C{i+1}", 1) for i in range(1, nsite)] + [(f"C{nsite}", f"C{1}", 1)],
                         alpha=0,  beta=-1,
                         u_onsite=np.array([1 for i in range(nsite+1)]))
        ecore = hubbard.generate_zero_body_integral()
        h = hubbard.generate_one_body_integral(sym=1, basis='spatial basis', dense=True)
        v = hubbard.generate_two_body_integral(sym=1, basis='spatial basis', dense=True)

        ham = pyci.hamiltonian(ecore, h, 2 * v)  # multiply by two because test doesn't work
        n_up = nelec
        n_down = nelec
        wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
        wfn.add_all_dets()
        op = pyci.sparse_op(ham, wfn)
        eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
        print(eigenvals)



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
    v = hubbard.generate_two_body_integral(sym=1, basis='spinorbital basis', dense=True)

    test = np.zeros_like(h)
    test[:2, :2] = np.array([[a, b], [b, a]])
    test[2:, 2:] = np.array([[a, b], [b, a]])
    assert np.allclose(h, test)
    test = np.zeros((16, 16))

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
    hubbard = HamPPP([("C1", "C2", 1), ("C2", "C3", 1), ("C3", "C4", 1), ("C4", "C1", 1)], alpha=a, beta=b)
    atoms_sites_lst, _ = hubbard.generate_connectivity_matrix()

    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(sym=1, basis='spatial basis', dense=True)
    v = hubbard.generate_two_body_integral(sym=1, basis='spinorbital basis', dense=True)

    assert v.shape[0] == 8
    assert np.allclose(h, np.array([[a, b, 0., b], [b, a, b, 0.], [0., b, a, b], [b, 0., b, a]]))
    v = hubbard.to_spatial(v, sym=1, dense=True, nbody=2)

    assert v.shape[0] == 4

    ham = pyci.hamiltonian(ecore, h, np.zeros((8, 8, 8, 8)))
    n_up = 2
    n_down = 2
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
    answer = 2*(a+2*b) + 2*a
    assert_allclose(eigenvals[0],  answer)

# print("running test 4")
# print(test_4())
# print("running test 1")
# print(test_1())
print("running test 2")
print(test_2())
