from abc import ABC, abstractmethod
from typing import TextIO
import numpy as np
from scipy.sparse import csr_matrix, diags, triu


class HamiltonianAPI(ABC):
    @abstractmethod
    def generate_zero_body_integral(self):
        """Generates zero body integral"""
        pass

    @abstractmethod
    def generate_one_body_integral(self, sym: int, basis: str, dense: bool):
        """
        Generates one body integral in spatial or spin orbital basis
        :param sym: symmetry -- [2, 4, 8] default is None
        :param basis: basis -- ['spatial', 'spin orbital']
        :param dense: dense or sparse matrix; default dense
        :return: numpy.ndarray or scipy.sparse.csc_matrix
        """
        pass

    @abstractmethod
    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        """
        Generates two body integral in spatial or spinorbital basis
        :param sym: symmetry -- [2, 4, 8] default is None
        :param basis: basis -- ['spatial', 'spin orbital']
        :param dense: dense or sparse matrix; default dense
        :return: numpy.ndarray or sparse
        """
        pass

    def to_dense(self, integral):
        """
        Generate a dense array from the sparse matrix
        :param integral: sparse
        :return: np.ndarray
        """

    def to_spatial(self, integral: np.ndarray, sym: int, dense: bool):
        """
        Converts one-/two- integral matrix from spin-orbital to spatial basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default is None
        :param dense: dense or sparse matrix; default sparse
        :return:
        """
        # FIXME: Eliminate hard-coded number of sites (k)
        # make it a class attribute: self._k
        k = 2
        if not sym in [1, 2, 4, 8]:
            raise ValueError('Wrong inpput symmetry')
        
        #
        # Assumption: spatial components of alpha and beta spin-orbitals are equivalent
        #
        if isinstance(integral, csr_matrix):
            spatial_int = csr_matrix((k, k))
            spatial_int = integral[:k, :k]
        elif isinstance(integral, dict):
            spatial_int = {}
            for p in range(k):
                # U_pppp
                spatial_int[(p,p,p,p)] = integral[(p, p + k, p, p + k)]
                for q in range(p, k):
                    # Gamma_pqpq, Pairing_ppqq
                    spatial_int[(p,q,p,q)] = integral[(p, q, p, q)]
                    spatial_int[(p,p,q,q)] = integral[(p, p + k, q, q + k)]
        else:
            raise ValueError('Wrong integral input.')
        
        spatial_int = expand_sym(sym, spatial_int, k)
        
        if dense:
            if isinstance(spatial_int, csr_matrix):
                spatial_int.toarray()
            else:
                spatial_int = self.to_dense(spatial_int)
        return spatial_int

    @abstractmethod
    def to_spinorbital(self, integral: np.ndarray, sym: int, dense: bool):
        """
        Converts one-/two- integral matrix from spatial to spin-orbital basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default None
        :param dense: dense or sparse matrix; default sparse
        :return:
        """
        pass

    def save_fcidump(self, f: TextIO, nelec=0, spinpol=0):
        """
        Save all parts of hamiltonian in fcidump format
        Adapted from https://github.com/theochem/iodata/blob/master/iodata/formats/fcidump.py
        :param f: TextIO file
        :param nelec: The number of electrons in the system
        :param spinpol: The spin polarization. By default, its value is derived from the
                        molecular orbitals (mo attribute), as abs(nalpha - nbeta). In this case,
                        spinpol cannot be set. When no molecular orbitals are present, this
                        attribute can be set.
        :return: None
        """
        one_ints = self.one_ints

        # Write header
        nactive = one_ints.shape[0]
        print(f' &FCI NORB={nactive:d},NELEC={nelec:d},MS2={spinpol:d},', file=f)
        print(f"  ORBSYM= {','.join('1' for v in range(nactive))},", file=f)
        print('  ISYM=1', file=f)
        print(' &END', file=f)

        # Write integrals and core energy
        two_ints = self.two_ints
        for i in range(nactive):  # pylint: disable=too-many-nested-blocks
            for j in range(i + 1):
                for k in range(nactive):
                    for l in range(k + 1):
                        if (i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l:
                            if (i, k, j, l) in two_ints:
                                value = two_ints[(i, k, j, l)]
                                print(f'{value:23.16e} {i + 1:4d} {j + 1:4d} {k + 1:4d} {l + 1:4d}', file=f)
        for i in range(nactive):
            for j in range(i + 1):
                value = one_ints[i, j]
                if value != 0.0:
                    print(f'{value:23.16e} {i + 1:4d} {j + 1:4d} {0:4d} {0:4d}', file=f)

        core_energy = self.core_energy
        if core_energy is not None:
            print(f'{core_energy:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}', file=f)


    @abstractmethod
    def save_triqs(self, fname:str, integral):
        """
        Save matrix in triqc format
        :param fname: filename
        :param integral: matrix to be saved
        :return: None
        """
        pass

    @abstractmethod
    def save(self, fname: str, integral, basis):
        """Save file as regular numpy array"""
        pass


def expand_sym(sym, integral, n):
    """
    Restore permutational symmetry of one- and two-body terms
    :param integral: input {one,two}-body integrals
    :param sym: int, current symmetry
    :param n: int, number of basis functions
    :return: integral with a symmetry of 1

    Notes
    -----
    The input integrals are expected to be sparse arrays with dimensions :math:`(N,N)` and 
    :math:`(N^2, N^2)` for the one- and two-body terms respectively. :math:`N` represents 
    the number of basis functions, which may be either of spatial or spin-orbital type. This
    function adds missing terms in the input array related to the given ones by the indicated 
    symmetry parameter `sym`. Allowed permutational symmetries are 1 (no symmetry), 2, 4 and 8. 
    Phicisist notation is used for the two-body integrals: :math:`<pq|rs>` and details of the 
    permutations considered can be found in [this site](http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.html). 
    """    
    if not sym in [1, 2, 4, 8]:
        raise ValueError('Wrong inpput symmetry')
    if sym == 1:
        return integral
    #
    # Expanding Symmetries
    #
    if integral.shape[0] == n:
        if not sym == 2:
            raise ValueError('Wrong 1-body term symmetry')
        h_ii = diags(integral.diagonal()).copy()
        integral = integral + integral.T - h_ii
    elif integral.shape[0] == n**2:
        # getting nonzero elements from the 2d _sparse_ array
        pq_array, rs_array = integral.nonzero()

        for pq, rs in zip(pq_array, rs_array):
            p, q, r, s = convert_indices(n, pq, rs)
            if sym == 2: 
                # 1. Transpose: <pq|rs>=<rs|pq>
                rs, pq = convert_indices(n, r,s,p,q)
                integral[rs, pq] = integral[pq, rs]
            elif sym == 4:
                # Add terms from fourfold permutational symmetry
                # 1. Transpose: <pq|rs>=<rs|pq>
                # 2. Permute dummy indices (swap variables of particles 1 and 2): 
                # <p_1 q_2|r_1 s_2> = <q_1 p_2|s_1 r_2>
                rs, pq = convert_indices(n, r,s,p,q)
                qp, sr = convert_indices(n, q,p,s,r)
                integral[rs, pq] = integral[pq, rs]
                integral[qp, sr] = integral[pq, rs]
                integral[sr, qp] = integral[rs, pq]
            else:
                # Add terms from eightfold permutational symmetry
                # 1. Transpose: <pq|rs>=<rs|pq>
                # 2. Permute dummy indices (swap variables of particles 1 and 2): 
                # <p_1 q_2|r_1 s_2> = <q_1 p_2|s_1 r_2>
                # 3. Permute orbitals of the same variable, e.g. <p_1 q_2|r_1 s_2> = <r_1 q_2|p_1 s_2>
                rs, pq = convert_indices(n, r,s,p,q)
                qp, sr = convert_indices(n, q,p,s,r)
                rq, ps = convert_indices(n, r,q,p,s)
                sp, qr = convert_indices(n, s,p,q,r)
                integral[rs, pq] = integral[pq, rs]
                integral[qp, sr] = integral[pq, rs]
                integral[sr, qp] = integral[rs, pq]                
                integral[rq, ps] = integral[pq, rs]
                integral[ps, rq] = integral[rs, pq]
                integral[sp, qr] = integral[qp, sr]
                integral[qr, sp] = integral[sr, qp]
    else:
        raise ValueError('Wrong input integral')
    return integral