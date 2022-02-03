from ..Huckel_Hamiltonian import *
from ..Hamiltonian import *
from scipy.sparse import csr_matrix


class Play(HamiltonianAPI):
    def generate_zero_body_integral(self): pass

    def generate_one_body_integral(self, sym: int, basis: str, dense: bool): pass

    def generate_two_body_integral(self, sym: int, basis: str, dense: bool): pass

    def to_spatial(self, integral: np.ndarray, sym: int, dense: bool): pass

    def to_spinorbital(self, integral: np.ndarray, sym: int, dense: bool): pass

    def save_triqs(self, fname: str, integral): pass

    def save(self, fname: str, integral, basis): pass


EPS = 1.0e-4

hubbard = Hubbard([('C3', 'C2', 1)], alpha=0, beta=-1, u_onsite=np.array([1, 1]))  # Should be re-written using HamAPI
ecore, h, v = hubbard.get_hamilton()

h = csr_matrix(h)
inds = np.where(v != 0)
inds_lst = list(zip(*inds))
v_sparse = {ind: value for ind, value in zip(inds_lst, v[inds])}
print(v_sparse)

play = Play()
play.one_ints = h
play.two_ints = v_sparse
play.core_energy = ecore
play.save_fcidump(open("test_fci.fcidump", 'w'), nelec=2)
