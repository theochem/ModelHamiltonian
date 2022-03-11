import numpy as np
import pyci
from PPP import *
from scipy.integrate import quad
from scipy.special import jv



def test_1():
    """ 2 site hubbard model with 2 electrons. Should return U=\frac{1}{2}\left[U-\sqrt{U^{2}+16 t^{2}}\right]$
    numerical result is -1.561552812 """

    hubbard = HamPPP([("C1", "C2", 1)], alpha=0, beta=-1, u_onsite=np.array([1, 1]),
                     gamma=None, charges=None, sym=None)
    ecore = hubbard.generate_zero_body_integral()
    h = hubbard.generate_one_body_integral(sym=None, basis='spin orbital', dense=True)
    v = hubbard.generate_two_body_integral(sym=None, basis='spin orbital', dense=True)

    return None
    # v_new = to_spatial(v)

    # ham = pyci.hamiltonian(ecore, h, v_new)
    # n_up = 1
    # n_down = 1
    # wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    # wfn.add_excited_dets(0)
    # wfn.add_excited_dets(1)
    # wfn.add_excited_dets(2)
    #
    # op = pyci.sparse_op(ham, wfn)
    # eigenvals, eigenvecs = op.solve(n=1, tol=1.0e-9)
    # np.allclose(eigenvals, -1.561552812)


def test_2():
    """
    4 site hubbard model with periodic boundary conditions. Should return
    $\frac{E(U, d=1)}{t N_{s}}=-4 \int_{0}^{\infty} d x \frac{J_{0}(x) J_{1}(x)}{x[1+\exp (U x / 2)]}$
    :return:
    """
    hubbard = Hubbard([("C1", "C2", 1), ("C2", "C3", 1), ("C3", "C4", 1), ("C4", "C1", 1)], alpha=0, beta=-0.5,
            u_onsite = np.array([1 for i in range(4)]))
    ecore, h, v = hubbard.get_hamilton()
    v_new = to_spatial(v)
    ham = pyci.hamiltonian(ecore, h, v_new)
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

test_2()
