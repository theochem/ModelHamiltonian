import numpy as np
from PPP import *

ham = HamPPP(connectivity=[('C1', 'C2', 1)], u_onsite=np.array([1, 1]), sym=None, gamma=np.zeros((2, 2)),
             charges=np.zeros(2))
ham.generate_connectivity_matrix()
print(ham.n_sites, ham.connectivity_matrix)
print(ham.generate_one_body_integral(sym=None, basis='spinorbital basis', dense=True))