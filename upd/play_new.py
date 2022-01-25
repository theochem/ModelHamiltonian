import numpy as np
import pyci
from Huckel_Hamiltonian import *
from scipy.integrate import quad
from scipy.special import jv

def to_spatial(v):
    new_shape = (np.array(v.shape)/2).astype(int)
    all_p, all_q, all_r, all_s = new_shape
    v_new = np.zeros(tuple(new_shape))
    for p in range(all_p):
        for q in range(all_q):
            for r in range(all_r):
                for s in range(all_s):
                    elem = 0
                    for sigma_1 in [0, 2]:
                        for sigma_2 in [0, 2]:
                            elem += v[p+sigma_1, q+sigma_2, r+sigma_1, s+sigma_2]
                    v_new[p, q, r, s] = elem
    return v_new/2

def test_1():
    """system of 3 electrons on two site hubbard model. Should return 0 energy ground state"""
    EPS = 1.0e-4
    hubbard = Hubbard([(f"C{i}",f"C{i+1}", 1) for i in range(1,2)], alpha=0, beta=-1, u_onsite=np.array([1 for i in range(2)]))
    ecore, h, v = hubbard.get_hamilton()

    v_new = to_spatial(v)

    ham = pyci.hamiltonian(ecore, h, v_new)
    n_up = 2
    n_down = 1 
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_excited_dets(0)
    wfn.add_excited_dets(1)
    wfn.add_excited_dets(2)

    eigenvals, eigenvecs = pyci.solve(ham, wfn, n=1, tol=1.0e-9)
    return eigenvals

def test_2():
    """ 2 site hubbard model with 2 electrons. Should return U=\frac{1}{2}\left[U-\sqrt{U^{2}+16 t^{2}}\right]$
    numerical result is -1.561552812 """
    
    EPS = 1.0e-4
    hubbard = Hubbard([(f"C{i}",f"C{i+1}", 1) for i in range(1,2)], alpha=0, beta=-1, u_onsite=np.array([1 for i in range(2)]))
    ecore, h, v = hubbard.get_hamilton()

    v_new = to_spatial(v)

    ham = pyci.hamiltonian(ecore, h, v_new)
    n_up = 1
    n_down = 1
    wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
    wfn.add_excited_dets(0)
    wfn.add_excited_dets(1)
    wfn.add_excited_dets(2)
    eigenvals, eigenvecs = pyci.solve(ham, wfn, n=1, tol=1.0e-9)
    return eigenvals


EPS = 1.0e-4
hubbard = Hubbard([("C1", "C2", 1), ("C2", "C3", 1), ("C3", "C4", 1), ("C4", "C1", 1)], alpha=0, beta=-0.5, 
        u_onsite = np.array([1 for i in range(4)]))
ecore, h, v = hubbard.get_hamilton()
v_new = to_spatial(v)
ham = pyci.hamiltonian(ecore, h, v_new)
n_up = 2
n_down = 2
wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_down)
wfn.add_excited_dets(0)
wfn.add_excited_dets(1)
wfn.add_excited_dets(2)
wfn.add_excited_dets(3)
wfn.add_excited_dets(4)


eigenvals, eigenvecs = pyci.solve(ham, wfn, n=1, tol=1.0e-9)

print(eigenvals)
print(eigenvecs)
