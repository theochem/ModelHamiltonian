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

    @abstractmethod
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
        
        # FIXME: Eliminate hard-coded {one,two}-body Hamiltonian operator symmetries
        # make it a class attribute: self._sym_{1b,2b}
        sym_1b, sym_2b = (2, 4)
        reduce_sym(spatial_int, k, sym_1b, sym_2b, sym)
        
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


def reduce_sym(integral, k, sym_1b, sym_2b, to_sym):
    """
    Reduce permutational symmetry of one- and two-body terms
    :param integral: input {one,two}-body integrals
    :param k: number of basis functions
    :param sym_1b: current symmetry of one-body integrals
    :param sym_2b: current symmetry of two-body integrals
    :param to_sym: target symmetry
    :return: integral with target symmetry
    """
    # FIXME: Only works for integrals in spatial-orbital basis
    #
    # Symmetries
    #
    if isinstance(integral, csr_matrix):
        if not to_sym in [1,2]:
            raise ValueError('Wrong 1-body term symmetry')
        if to_sym == sym_1b:
            pass
        elif to_sym > sym_1b:
            raise ValueError(f'Wrong symmetry requested, desired symmetry must be <= {sym_1b}')
        else:
            integral = integral + integral.T - diags(integral.diagonal())
    else:
        if to_sym > sym_2b:
            raise ValueError(f'Wrong symmetry requested, desired symmetry must be <= {sym_2b}')
        elif to_sym == sym_2b:
            pass 
        elif sym_2b == 4:
            if to_sym == 1:
                # Add terms from fourfold permutational symmetry
                for i in range(k):
                    for j in range(i,k):
                        integral[(j, i, j, i)] = integral[(i, j, i, j)]
                        integral[(j,j,i,i)] = integral[(i,i,j,j)]
            else:
                # leave out <ij|kl>=<kl|ij> terms
                for i in range(k):
                    for j in range(i,k):
                        integral[(j, i, j, i)] = integral[(i, j, i, j)]
        elif sym_2b == 8:
            if to_sym == 1:
                # Add all terms from eightfold permutational symmetry
                for i in range(k):
                    for j in range(i,k):
                        integral[(j, i, j, i)] = integral[(i, j, i, j)]
                        integral[(j,j,i,i)] = integral[(i,i,j,j)]
                        integral[(j,i,i,j)] = integral[(i,i,j,j)]
                        integral[(i,j,j,i)] = integral[(i,i,j,j)]
            elif to_sym == 2:
                # leave out <ij|kl>=<kl|ij> terms
                for i in range(k):
                    for j in range(i,k):
                        integral[(j,i,i,j)] = integral[(i,i,j,j)]
                        integral[(i,j,j,i)] = integral[(i,i,j,j)] 
            else:
                # leave out <ij|kl>=<kl|ij>=<ji|lk>=<lk|ji> terms
                for i in range(k):
                    for j in range(i,k):
                        integral[(j,i,i,j)] = integral[(i,i,j,j)]
                        integral[(i,j,j,i)] = integral[(i,i,j,j)]
        else: # sym_2b = 2
            for i in range(k):
                for j in range(i,k):
                    integral[(j,j,i,i)] = integral[(i,i,j,j)]
        return integral


def restore_sym(sym, integral, k, sym_2b):
    """
    Restore permutational symmetry of one- and two-body terms
    :param integral: input {one,two}-body integrals
    :param k: number of sites
    :param sym_1b: int, current  one-body integrals symmetry
    :param sym_2b: int, current two-body integrals symmetry
    :param sym: int, target symmetry
    :return: integral with target symmetry
    """
    #
    # Symmetries
    #
    if isinstance(integral, csr_matrix):
        if not sym == 2:
            raise ValueError('Wrong 1-body term symmetry')
        integral_sym = integral + integral.T - diags(integral.diagonal())
    else:
        if sym > sym_2b:
            raise ValueError(f'Wrong symmetry requested, desired symmetry must be <= {sym_2b}')
        integral_sym = integral.copy()
        if sym == 1:
            pass 
        elif sym == 2: 
            # Add <ij|kl>=<kl|ij> terms
            for (p, q, r, s) in integral.keys():
                if p != r and (q%k == p) and (s%k == r):
                    integral_sym[(r,s,p,q)] = integral[(p,q,r,s)]
        elif sym == 4:
            # Add terms from fourfold permutational symmetry
            for (p, q, r, s) in integral.keys():
                if p != r and (q%k == p) and (s%k == r):
                    integral_sym[(r,s,p,q)] = integral[(p,q,r,s)]
                    integral_sym[(q,p,s,r)] = integral[(p,q,r,s)]
                    integral_sym[(s,r,q,p)] = integral[(p,q,r,s)]
                elif (r%k == p) and (s%k == q):
                    integral_sym[(q,p,q,p)] = integral[(p,q,p,q)]
        else:
            # Add all terms from eightfold permutational symmetry
            for (p, q, r, s) in integral.keys():
                if p != r and (q%k == p) and (s%k == r):
                    integral_sym[(r,s,p,q)] = integral[(p,q,r,s)]
                    integral_sym[(q,p,s,r)] = integral[(p,q,r,s)]
                    integral_sym[(s,r,q,p)] = integral[(p,q,r,s)]
                    integral_sym[(r,q,p,s)] = integral[(p,q,r,s)]
                    integral_sym[(s,p,q,r)] = integral_sym[(q,p,s,r)]
                    integral_sym[(p,s,r,q)] = integral_sym[(r,s,p,q)]
                    integral_sym[(q,r,s,p)] = integral_sym[(s,r,q,p)]
                elif (r%k == p) and (s%k == q):
                    integral_sym[(q,p,q,p)] = integral[(p,q,p,q)]
        return integral_sym