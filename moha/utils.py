r"""MoHa utilities submodule."""

import numpy as np

from scipy.sparse import diags


__all__ = [
    "convert_indices",
    "get_atom_type",
    "expand_sym",
]


def convert_indices(N, *args):
    r"""
    Convert indices from 4d array to 2d numpy array and vice-versa.

    :param N: size of corresponding _4d_ matrix: int
    :param args: indices i, j, k, l or p,q: int
    :return: list of converted indices: list
    """
    if len(args) == 4:
        # Check if indices are right
        for elem in args:
            if not isinstance(elem, int):
                raise TypeError("Wrong indices")
            if elem >= N:
                raise TypeError("index is greater than size of the matrix")

        # converting indices
        i, j, k, l_ = args
        p = int(i * N + j)
        q = int(k * N + l_)
        return [p, q]

    elif len(args) == 2:
        # check indices
        for elem in args:
            if not isinstance(elem, int):
                raise TypeError("Wrong indices")
            if elem >= N**2:
                raise TypeError("index is greater than size of the matrix")

        # converting indices
        p, q = args
        i, k = p // N, q // N
        j, l_ = p % N, q % N
        return [i, j, k, l_]
    else:
        raise TypeError("Wrong indices")


def get_atom_type(atom):
    r"""
    Return atom type, site index and coordination if given; 
    "C23" -> "C", 23. "C5_4" -> "C", 5, 4. 

    :param atom: str
    :return: tuple
    """
    c = None
    if "_" in atom:
        atom, c = atom.split("_")
        i = 1
        while atom[-i:].isdigit():
            i += 1
        i -= 1
    else:
        i = 1
        while atom[-i:].isdigit():
            i += 1
        i -= 1
        c= None
    return atom[:-i], int(atom[-i:]), c


def expand_sym(sym, integral, nbody):
    r"""
    Restore permutational symmetry of one- and two-body terms.

    Parameters
    ----------
    sym: int
        integral symmetry, one of 1 (no symmetry), 2, 4 or 8.
    integral: scipy.sparce.csr_matrix
        2-D sparse array, the {one,two}-body integrals
    nbody: int
        number of particle variables in the integral,
        one of 1 (one-body) or 2 (two-body)

    Returns
    -------
    integral: scipy.sparse.csr_matrix
        2d array with the symmetry 1

    Notes
    -----
        Given the one- or two-body Hamiltonian matrix terms,
        :math:`h_{i,j}` and :math:`g_{ij,kl}` respectively,
        the supported permutational symmetries are:
        sym = 2:
        :math:`h_{i,j} = h_{j,i}`
        :math:`g_{ij,kl} = g_{kl,ij}`
        sym = 4:
        :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji}`
        sym = 8:
        :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji} =
         g_{kj,il} = g_(il,kj) = g_(li,jk) = g_(jk,li)`
        sym = 1 corresponds to no-symmetry
        where it is assumed the integrals are over real orbitals.

        The input Hamiltonian terms are expected to be sparse
        arrays of dimensions :math:`(N,N)` or
        :math:`(N^2, N^2)` for the one- and two-body integrals respectively.
        :math:`N` represents the number of basis functions,
        which may be either of spatial or spin-orbital type.
        This function applies to the input array the
        permutations indicated by the symmetry parameter `sym`
        to adds the missing terms.
        Phicisist notation is used for the two-body integrals:
         :math:`<pq|rs>` and further details of the
        permutations considered can be
        found in [this site]
        (http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.html).
    """
    if sym not in [1, 2, 4, 8]:
        raise ValueError("Wrong input symmetry")
    if nbody not in [1, 2]:
        raise ValueError(f"`nbody` must be an integer, "
                         f"either 1 or 2, but {nbody} given")
    if sym == 1:
        return integral

    # Expanding Symmetries
    if nbody == 1:
        if not sym == 2:
            raise ValueError("Wrong 1-body term symmetry")
        h_ii = diags(integral.diagonal()).copy()
        integral = integral + integral.T - h_ii
    else:
        # getting nonzero elements from the 2d _sparse_ array
        pq_array, rs_array = integral.nonzero()
        n = int(np.sqrt(integral.shape[0]))

        for pq, rs in zip(pq_array, rs_array):
            p, q, r, s = convert_indices(n, int(pq), int(rs))
            if sym >= 2:
                # 1. Transpose: <pq|rs>=<rs|pq>
                rs, pq = convert_indices(n, r, s, p, q)
                integral[rs, pq] = integral[pq, rs]
            if sym >= 4:
                # 2. Permute dummy indices
                # (swap variables of particles 1 and 2):
                # <p_1 q_2|r_1 s_2> = <q_1 p_2|s_1 r_2>
                qp, sr = convert_indices(n, q, p, s, r)
                integral[qp, sr] = integral[pq, rs]
                integral[sr, qp] = integral[rs, pq]
            if sym == 8:
                # 3. Permute orbitals of the same variable,
                # e.g. <p_1 q_2|r_1 s_2> = <r_1 q_2|p_1 s_2>
                rq, ps = convert_indices(n, r, q, p, s)
                sp, qr = convert_indices(n, s, p, q, r)
                integral[rq, ps] = integral[pq, rs]
                integral[ps, rq] = integral[rs, pq]
                integral[sp, qr] = integral[qp, sr]
                integral[qr, sp] = integral[sr, qp]
    return integral

ionization = {'H':  13.5984, 'He': 24.5874, 'Li': 5.3917, \
              'Be':  9.3227, 'B':    8.298, 'C': 11.2603, \
              'N':  14.5341, 'O':  13.6181, 'F': 17.4228, \
              'Ne': 21.5645, 'Na':  5.1391, 'Mg': 7.6462, \
              'Al':  5.9858, 'Si':  8.1517, 'P': 10.4867, \
              'S':    10.36, 'Cl': 12.9676, 'Ar':15.7596, \
              'K':   4.3407, 'Ca':  6.1132, 'Sc': 6.5615, \
              'Ti':  6.8281, 'V':   6.7462, 'Cr': 6.7665, \
              'Mn':   7.434, 'Fe':  7.9024, 'Co':  7.881, \
              'Ni':  7.6398, 'Cu':  7.7264, 'Zn': 9.3942, \
               'Ga': 5.9993, 'Ge':  7.8994, 'As': 9.7886, \
               'Se': 9.7524, 'Br': 11.8138, 'Kr':13.9996, \
               'Rb': 4.1771, 'Sr':  5.6949, 'Y':  6.2173, \
               'Zr': 6.6339, 'Nb':  6.7589, 'Mo': 7.0924, \
                 'Tc': 7.28, 'Ru':  7.3605, 'Rh': 7.4589, \
               'Pd': 8.3369, 'Ag':  7.5762, 'Cd': 8.9938, \
               'In': 5.7864, 'Sn':  7.3439, 'Sb': 8.6084, \
               'Te': 9.0096, 'I':  10.4513, 'Xe':12.1298, \
               'Cs': 3.8939, 'Ba':  5.2117, 'La': 5.5769, \
               'Ce': 5.5387, 'Pr':   5.473, 'Nd':  5.525, \
                'Pm': 5.582, 'Sm':  5.6437, 'Eu': 5.6704, \
               'Gd': 6.1501, 'Tb':  5.8638, 'Dy': 5.9389, \
               'Ho': 6.0215, 'Er':  6.1077, 'Tm': 6.1843, \
               'Yb': 6.2542, 'Lu':  5.4259, 'Hf': 6.8251, \
               'Ta': 7.5496, 'W':    7.864, 'Re': 7.8335, \
               'Os': 8.4382, 'Ir':   8.967, 'Pt': 8.9587, \
               'Au': 9.2255, 'Hg': 10.4375, 'Tl': 6.1082, \
               'Pb': 7.4167, 'Bi':  7.2856, 'Po':  8.417, \
               'At':    9.3, 'Rn': 10.7485, 'Fr': 4.0727, \
               'Ra': 5.2784, 'Ac':    5.17, 'Th': 6.3067, \
                 'Pa': 5.89, 'U':   6.1941, 'Np': 6.2657, \
               'Pu': 6.0262, 'Am':  5.9738, 'Cm': 5.9915, \
               'Bk': 6.1979, 'Cf':  6.2817, 'Es':   6.42, \
               'Fm':    6.5, 'Md':    6.58, 'No':   6.65}