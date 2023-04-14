#%%
import sys  
import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix, hstack, vstack

sys.path.insert(0, '../')
def print_hamiltonian(h1):
    np.set_printoptions(precision=3)      
    print('h1=\n',h1,'\n')

#atom_dictionary={'B':2.4, 'Cl':4, 'F':5, 'Si':3 }

from moha import HamHuck
norb = 6
#polyene = HamHuck([(f"C{i}", f"C{i + 1}", 1) for i in range(1, norb)] + [(f"C{norb}", f"C{1}", 1)])
C_Cl_F_Si =  HamHuck([('B1','Cl2',1),('Cl2','F3',1),('F3','Si4',1),('Si4','B1',1)],atom_dictionary=atom_dictionary)
#h1 = polyene.generate_one_body_integral(dense=True, basis='spatial basis')
h1 = C_Cl_F_Si.generate_one_body_integral(dense=True, basis='spatial basis')
print_hamiltonian(h1)

# %%
def print_hamiltonian(h0, h1, h2):
    np.set_printoptions(precision=3)
    print('h0=\n',"%0.3f" % h0,'\n')        
    print('h1=\n',h1,'\n')
    print('h2=\n',h2,'\n')
norb = 6
b = 1.4
gamma0 = 10.84
beta = -2.5
def generate_gamma(norb, b, gamma0): 
    ang_to_bohr = 1.889726
    har_to_ev = 27.211396
    b_ieV = b*ang_to_bohr/har_to_ev

    gamma = np.zeros((norb,norb))
    for u in range(norb):
        for v in range(norb):
            duv = b*(np.sin(abs(u-v)*np.pi/norb)/np.sin(np.pi/norb))
            gamma[(u,v)] = 1/(1/gamma0 + duv)

    return gamma 

gamma = generate_gamma(norb, b, gamma0)
import sys  
sys.path.insert(0, '../')

from moha import HamPPP

polyene = HamPPP([(f"C{i}", f"C{i + 1}", 1) for i in range(1, norb)] + [(f"C{norb}", f"C{1}", 1)],
                      alpha=0, beta=beta, gamma=gamma, charges=np.ones(6),
                      u_onsite=np.array([0.5*gamma0 for i in range(norb + 1)]))

h0 = polyene.generate_zero_body_integral()
h1 = polyene.generate_one_body_integral(dense=True, basis='spatial basis')
h2 = polyene.generate_two_body_integral(dense=False, basis='spatial basis')

print_hamiltonian(h0, h1, h2)

# %%
