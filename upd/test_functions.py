import numpy as np
from PPP import *

ham = HamPPP(connectivity=[('C1', 'C2', 1)], u_onsite=np.array([1, 1]), sym=None, gamma=None, charges=None)
ham.generate_connectivity_matrix()
print(ham.n_sites, ham.connectivity_matrix)
print(ham.generate_one_body_integral(sym=None, basis='spatial basis', dense=True))