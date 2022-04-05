from itertools import permutations

import numpy as np

from .hamiltonian import HamiltonianAPI


def to_spatial(v):
    """
    Converts matrix in spinorbital basis to spatial basis
    :param v: two body integral in spinorbital basis
    :return: np.array: two body integral in spatial basis
    """
    new_shape = (np.array(v.shape) / 2).astype(int)
    all_p, all_q, all_r, all_s = new_shape
    v_new = np.zeros(tuple(new_shape))
    for p in range(all_p):
        for q in range(all_q):
            for r in range(all_r):
                for s in range(all_s):
                    elem = 0
                    for sigma_1 in [0, 2]:
                        for sigma_2 in [0, 2]:
                            elem += v[p + sigma_1, q + sigma_2, r + sigma_1, s + sigma_2]
                    v_new[p, q, r, s] = elem
    return v_new / 2


def distance(seq_1, seq_2):
    """
    Calculate number of different elements in two sequences
    :param seq_1: first sequence (set of indices)
    :param seq_2: second sequence (set of distance)
    :return: number of different elements in sequences pairwise: int
    """
    return sum([1 for i in range(len(seq_1)) if seq_1[i] != seq_2[i]])


def sign_parity(ref, seq, count=0):
    """
    Calculate sign of permutations for sequence in relation to reference sequence
    :param ref: refernce sequence
    :param seq: sequence one wants to know the sign of permutations
    :param count: number of counts
    :return: sign of permutations, amount of the necessary permutations to make from sequence reference sequence: tuple
    """
    if ref == seq:
        return (-1) ** count, count

    n_elems = len(ref)
    tmp_seqs = []
    distances = []
    settled = [i for i in range(n_elems) if ref[i] == seq[i]]
    for i in range(n_elems):
        for j in range(i + 1, n_elems):
            if seq[i] != seq[j] and not (i in settled or j in settled):
                seq_tmp = seq.copy()
                seq_tmp[i], seq_tmp[j] = seq_tmp[j], seq_tmp[i]

                tmp_seqs.append(seq_tmp)
                distances.append(distance(ref, seq_tmp))
    if distances == []:
        print(seq, ref)
    best_seq = tmp_seqs[np.argmin(distances)]
    return sign_parity(ref, best_seq, count + 1)


def fill_with_parity(V, ref_set):
    """
    Fill the gaps in V matrix based on the existing elements, taking into account their indices and parity
    :param V: Two electron integral matrix with filled elements indices of wich are p<q<r<s
    :param ref_set: set of indices based of which wil be filled other elements of the V matrix taking into account
                    their parity
    :return: filled two electron integral matrix: np.array
    """
    permutations_1 = list(permutations(ref_set[:2], 2))
    permutations_2 = list(permutations(ref_set[2:], 2))
    permutation_indices = [
        list(ind_1) + list(ind_2) for ind_1 in permutations_1 for ind_2 in permutations_2
    ]
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


class PPP(HamiltonianAPI):
    def __init__(
        self,
        bond_types,
        alpha=-0.414,
        beta=-0.0533,
        u_onsite=None,
        gamma=0.0784,
        charges=0.417,
        g_pair=None,
        atom_types=None,
        atom_dictionary=None,
        bond_dictionary=None,
        Bz=None,
    ):
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
        """
        Builds the connectivity matrix based on the bond_types
        :return: connectivuty matrix :np.array
        """
        elements = set([atom for bond_type in self.bond_types for atom in bond_type[:2]])
        n_atoms = len(elements)
        self.atoms_num = {elem: i for i, elem in enumerate(elements)}
        connectivity = np.zeros((n_atoms, n_atoms))
        for atom1, atom2, bond in self.bond_types:
            connectivity[self.atoms_num[atom1], self.atoms_num[atom2]] = bond
        return np.maximum(connectivity, connectivity.T)

    def generate_zero_body_integral(self):
        """Generates zero body integral"""
        pass

    def generate_one_body_integral(self, sym: int, basis: str, dense: bool):
        """
        Generates one body integral in spatial or spin orbital basis
        :param sym: symmetry -- [2, 4, 8] default is None
        :param basis: basis -- ['spatial', 'spin orbital']
        :param dense: dense or sparse matrix; default dense
        :return: numpy.ndarray or scipy.sparse.csc_matrix
        """
        pass

    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        """
        Generates two body integral in spatial or spinorbital basis
        :param sym: symmetry -- [2, 4, 8] default is None
        :param basis: basis -- ['spatial', 'spin orbital']
        :param dense: dense or sparse matrix; default dense
        :return: numpy.ndarray or sparse
        """
        pass

    def to_spatial(self, integral: np.ndarray, sym: int, dense: bool):
        """
        Converts one-/two- integral matrix from spin-orbital to spatial basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default is None
        :param dense: dense or sparse matrix; default sparse
        :return:
        """
        pass

    def to_spinorbital(self, integral: np.ndarray, sym: int, dense: bool):
        """
        Converts one-/two- integral matrix from spatial to spin-orbital basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default None
        :param dense: dense or sparse matrix; default sparse
        :return:
        """
        pass

    def get_hamilton(self):
        """
        Build zero, one and two body integrals
        :return: tuple: zero, one and two body integrals as numpy arrays
        """
        self.connectivity = self.get_connectivity_matrix()
        self.cm_len = self.connectivity.shape[0]
        n_sp = self.cm_len

        h_huckel = (
            np.diag([self.alpha for _ in range(n_sp)]) + self.beta * self.connectivity
        )  ## look at the Rauk's dictionary
        h_hubbard = np.zeros((2 * n_sp, 2 * n_sp))
        for p in range(n_sp):
            h_hubbard[p, p + n_sp] = self.u_onsite[p]

        h_ppp = np.zeros((2 * n_sp, 2 * n_sp))
        for p in range(n_sp):
            for q in range(n_sp):
                if p != q:
                    h_ppp[p, q] += self.gamma[p, q]
                    h_ppp[p, q + n_sp] += self.gamma[p, q]
                    h_ppp[p + n_sp, q] += self.gamma[p, q]
                    h_ppp[p + n_sp, q + n_sp] += self.gamma[p, q]
                    h_huckel[p, p] -= 2 * self.gamma[p, q] * self.charges[q]
                    h_huckel[p, p] -= 2 * self.gamma[p, q] * self.charges[q]
                    h_huckel[q, q] -= 2 * self.gamma[p, q] * self.charges[p]
                    h_huckel[q, q] -= 2 * self.gamma[p, q] * self.charges[p]

        h_ppp *= 0.5
        h_zero = np.sum(np.outer(self.charges, self.charges)) - np.dot(self.charges, self.charges)
        h_pair = self.g_pair

        v = np.zeros((2 * n_sp, 2 * n_sp, 2 * n_sp, 2 * n_sp))
        for p in range(n_sp):
            for q in range(n_sp):
                v[p, q, q, p] = h_ppp[p, q]
                v[p, q + n_sp, q + n_sp, p] = h_ppp[p, q + n_sp]
                v[p + n_sp, q, q, p + n_sp] = h_ppp[p + n_sp, q]
                v[p + n_sp, q + n_sp, q + n_sp, p + n_sp] = h_ppp[p + n_sp, q + n_sp]
                v[p, p + n_sp, q, q + n_sp] = -1 * h_pair[p, q]
                # v[p, p+n_sp, p+n_sp, p] += h_hubbard[p, p+n_sp]
                v[p, p + n_sp, p, p + n_sp] = h_hubbard[p, p + n_sp]

        for p in range(n_sp):
            for q in range(n_sp):
                for ref_set in [
                    [p, q, q, p],
                    [p, q + n_sp, q + n_sp, p],
                    [p + n_sp, q, q, p + n_sp],
                    [p + n_sp, q + n_sp, q + n_sp, p + n_sp],
                    [p, p + n_sp, q, q + n_sp],
                    [p, p + n_sp, p, p + n_sp],
                ]:  ##!

                    v = fill_with_parity(v, ref_set)

        return h_zero, h_huckel, v


class Huckel(PPP):
    def __init__(self, bond_types, alpha=-0.414, beta=-0.0533):
        elements = set([atom for bond_type in bond_types for atom in bond_type[:2]])
        n_atoms = len(elements)
        super().__init__(
            bond_types=bond_types,
            alpha=alpha,
            beta=beta,
            u_onsite=np.zeros(n_atoms),
            gamma=np.zeros((n_atoms, n_atoms)),
            charges=np.zeros(n_atoms),
            g_pair=np.zeros((n_atoms, n_atoms)),
            atom_types=None,
            atom_dictionary=None,
            bond_dictionary=None,
            Bz=None,
        )


class Hubbard(PPP):
    def __init__(self, bond_types, u_onsite, alpha=-0.414, beta=-0.0533):
        elements = set([atom for bond_type in bond_types for atom in bond_type[:2]])
        n_atoms = len(elements)
        super().__init__(
            bond_types=bond_types,
            alpha=alpha,
            beta=beta,
            u_onsite=u_onsite,
            gamma=np.zeros((n_atoms, n_atoms)),
            charges=np.zeros(n_atoms),
            g_pair=np.zeros((n_atoms, n_atoms)),
            atom_types=None,
            atom_dictionary=None,
            bond_dictionary=None,
            Bz=None,
        )


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
