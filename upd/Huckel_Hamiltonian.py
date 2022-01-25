import numpy as np
from scipy.sparse import dok_matrix
from itertools import permutations

def distance(seq_1, seq_2):
    return sum([1 for i in range(len(seq_1)) if seq_1[i] != seq_2[i]])

def sign_parity(ref, seq, count=0):
    if ref == seq:
        return (-1)**count, count

    n_elems = len(ref)
    tmp_seqs = []
    distances = []
    settled = [i for i in range(n_elems) if ref[i] == seq[i]]
    for i in range(n_elems):
        for j in range(i+1, n_elems):
            if seq[i] != seq[j] and not (i in settled or j in settled):
                seq_tmp = seq.copy()
                seq_tmp[i], seq_tmp[j] = seq_tmp[j], seq_tmp[i]

                tmp_seqs.append(seq_tmp)
                distances.append(distance(ref, seq_tmp))
    if distances == []:
        print(seq, ref)
    best_seq = tmp_seqs[np.argmin(distances)]
    return sign_parity(ref, best_seq, count+1)


def fill_with_parity(V, ref_set):
    permutations_1 = list(permutations(ref_set[: 2], 2))
    permutations_2 = list(permutations(ref_set[2:], 2))
    permutation_indices = [list(ind_1)+list(ind_2) for ind_1 in permutations_1 for ind_2 in permutations_2]
    ref_value = 0
    for ind in permutation_indices:
        p, q, r, s = ind
        if V[p, q, r, s] != 0:
            ref_value = V[p, q, r, s].copy()
            p_, q_, r_, s_ = p, q, r, s
            break
    if ref_value == 0:
        return V

    for ind in permutation_indices:
        p, q, r, s = ind
        sign, _ = sign_parity([p_, q_, r_, s_], list(ind))
        
        V[p, q, r, s] = sign * ref_value
    return V

class PPP():
    def __init__(self, bond_types, alpha=-0.414, beta=-0.0533, u_onsite=None, gamma=0.0784, charges=0.417, g_pair=None,
                 atom_types=None, atom_dictionary=None, bond_dictionary=None, Bz=None):
        self.bond_types = bond_types
        self.alpha = alpha
        self.beta = beta
        self.u_onsite = u_onsite
        self.gamma = gamma
        self.charges = charges
        self.g_pair = g_pair
        self.atom_types = atom_types
        self.atom_dictionary = atom_dictionary
        self.bond_dictionary = bond_dictionary


    def get_connectivity_matrix(self):
        elements = set([atom for bond_type in self.bond_types for atom in bond_type[:2] ])
        n_atoms = len(elements)
        self.atoms_num = {elem: i for i, elem in enumerate(elements)}
        connectivity = np.zeros((n_atoms, n_atoms))
        for atom1, atom2, bond in self.bond_types:
            connectivity[self.atoms_num[atom1], self.atoms_num[atom2]] = bond
        return np.maximum(connectivity, connectivity.T)


    def get_hamilton(self):
        self.connectivity = self.get_connectivity_matrix()

        n_sp = self.connectivity.shape[0]
        h_huckel = np.diag([self.alpha for _ in range(n_sp)]) + self.beta*self.connectivity ## look at the Rauk's dictionary
        h_hubbard = np.zeros((2*n_sp, 2*n_sp))
        for p in range(n_sp):
            h_hubbard[p, p+n_sp] = self.u_onsite[p]

        h_ppp = np.zeros((2*n_sp, 2*n_sp))
        for p in range(n_sp):
            for q in range(n_sp):
                if p != q:
                    h_ppp[p, q] += self.gamma[p, q]
                    h_ppp[p, q+n_sp] += self.gamma[p, q]
                    h_ppp[p+n_sp, q] += self.gamma[p, q]
                    h_ppp[p+n_sp, q+n_sp] += self.gamma[p, q]
                    h_huckel[p, p] -= 2*self.gamma[p, q]*self.charges[q]
                    h_huckel[p, p] -= 2*self.gamma[p, q] * self.charges[q]
                    h_huckel[q, q] -= 2*self.gamma[p, q]*self.charges[p]
                    h_huckel[q, q] -= 2*self.gamma[p, q] * self.charges[p]

        h_ppp *= 0.5
        h_zero = np.sum(np.outer(self.charges, self.charges)) - np.dot(self.charges, self.charges)
        h_pair = self.g_pair

        v = np.zeros((2*n_sp, 2*n_sp, 2*n_sp, 2*n_sp))
        for p in range(n_sp):
            for q in range(n_sp):
                v[p, q, q, p] = h_ppp[p, q]
                v[p, q+n_sp, q+n_sp, p] = h_ppp[p, q+n_sp]
                v[p+n_sp, q, q, p+n_sp] = h_ppp[p+n_sp, q]
                v[p+n_sp, q+n_sp, q+n_sp, p+n_sp] = h_ppp[p+n_sp, q+n_sp]
                v[p, p+n_sp, q, q+n_sp] = -1*h_pair[p, q]
                # v[p, p+n_sp, p+n_sp, p] += h_hubbard[p, p+n_sp]
                v[p, p + n_sp, p, p + n_sp] = h_hubbard[p, p + n_sp]

        for p in range(n_sp):
            for q in range(n_sp):
                for ref_set in [[p, q, q, p],
                                [p, q+n_sp, q+n_sp, p],
                                [p+n_sp, q, q, p+n_sp],
                                [p+n_sp, q+n_sp, q+n_sp, p+n_sp],
                                [p, p+n_sp, q, q+n_sp],
                                [p, p+n_sp, p, p+n_sp]]:##!

                    v = fill_with_parity(v, ref_set)



        return h_zero, h_huckel, v


class Huckel(PPP):
    def __init__(self, bond_types, alpha=-0.414, beta=-0.0533):
        elements = set([atom for bond_type in bond_types for atom in bond_type[:2]])
        n_atoms = len(elements)
        super().__init__(bond_types=bond_types, alpha=alpha, beta=beta, u_onsite=np.zeros(n_atoms),
                         gamma=np.zeros((n_atoms, n_atoms)), charges=np.zeros(n_atoms), g_pair=np.zeros((n_atoms, n_atoms)),
                         atom_types=None, atom_dictionary=None, bond_dictionary=None, Bz=None)

class Hubbard(PPP):
    def __init__(self, bond_types, u_onsite, alpha=-0.414, beta=-0.0533):
        elements = set([atom for bond_type in bond_types for atom in bond_type[:2]])
        n_atoms = len(elements)
        super().__init__(bond_types=bond_types, alpha=alpha, beta=beta, u_onsite=u_onsite,
                         gamma=np.zeros((n_atoms, n_atoms)), charges=np.zeros(n_atoms),
                         g_pair=np.zeros((n_atoms, n_atoms)), atom_types=None, atom_dictionary=None,
                         bond_dictionary=None, Bz=None)




# ref = [0, 2, 0, 2]
# print(sign_parity(ref, [0, 2, 2, 0]))
# for p in permutations(ref, 4):
#     print(sign_parity(ref, list(p)), p)

###figure out why
# print(sign_parity([0, 1, 2, 3], [0, 3, 1, 2]))
#
# print(np.array(list(permutations(range(4), 4))).reshape(-1, 4))

# test = Hubbard([('C2', 'C1', 1)], alpha=0, beta=-1, u_onsite=np.array([1, 1]))
# test = Huckel([('C2', 'C1', 1), ('C2', 'C3', 1), ('C3', 'C4', 1), ('C4', 'C5', 1),
#                ('C5', 'C6', 1), ('C6', 'C1', 1)])
#
# huckel = test.get_hamilton()[1]
# hubbard = test.get_hamilton()

# print(hubbard[0], hubbard[1], hubbard[2])
# print(0.5+np.sqrt(0.25+4*test.beta**2), 1+test.beta)
# print(np.linalg.eig(hubbard)[0], test.alpha-2*test.beta, test.alpha-test.beta, test.alpha+test.beta, test.alpha+2*test.beta)
