import numpy as np

# import pyci
from .Huckel_Hamiltonian import *
from .Hamiltonian import *
from scipy.sparse import csr_matrix

EPS = 1.0e-4

hubbard = Hubbard([("C3", "C2", 1)], alpha=0, beta=-1, u_onsite=np.array([1, 1]))
ecore, h, v = hubbard.get_hamilton()

# for p in range(4):
#     for q in range(4):
#         for r in range(4):
#             for s in range(4):
#                 if v[p, q, r, s] != 0:
#                     print(v[p, q, r ,s], p, q, r, s)
# h_new = np.zeros((4, 4))
# h_new[:2, :2] = h
# h_new[2:, 2:] = h
#
# v = np.zeros((2, 2, 2, 2))
# v[0,0,0,0] = 1
# v[1, 1, 1, 1] = 1

# ham = pyci.hamiltonian(ecore, h, v)
# n_up = 1
# n_dn = 1
#
# wfn = pyci.fullci_wfn(ham.nbasis, n_up, n_dn)
# wfn.add_excited_dets(0)
# wfn.add_excited_dets(1)
# wfn.add_excited_dets(2)
# eigenvals, eigenvecs = pyci.solve(ham, wfn, n=1, tol=1.0e-9)

# for p in range(4):
#    for q in range(4):
#        for r in range(4):
#            for s in range(4):
#                if v[p, q, r, s] != 0:
#                    print(p, q, r, s, v[p, q, r, s])
# print(ecore, h_new, sep='\n')
# print(eigenvals)
# print(0.5*(1 - np.sqrt(1+16)))
# print('success')
#
h = csr_matrix(h)
inds = np.where(v != 0)
inds_lst = list(zip(*inds))
v_sparse = {ind: value for ind, value in zip(inds_lst, v[inds])}
print(v_sparse)
# print(v[:2, :2, :2, :2])
# np.save('ecore', ecore)
# np.save('h', h)
# np.save('v', v)


class Play(HamiltonianAPI):
    def generate_zero_body_integral(self):
        pass

    def generate_one_body_integral(self, sym: int, basis: str, dense: bool):
        pass

    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        pass

    def to_spatial(self, integral: np.ndarray, sym: int, dense: bool):
        pass

    def to_spinorbital(self, integral: np.ndarray, sym: int, dense: bool):
        pass

    def save_triqs(self, fname: str, integral):
        pass

    def save(self, fname: str, integral, basis):
        pass


play = Play()
play.one_mo = h
play.two_mo = v_sparse
play.core_energy = ecore
play.save_fcidump(open("test_fci.fcidump", "w"), nelec=2)
# play.
