from abc import ABC, abstractmethod
from typing import TextIO
from utils import convert_indices
import numpy as np
from scipy.sparse import csr_matrix
from utils import convert_indices


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

    def to_sparse(self, Md):
        """
        Converts dense array of integrals to sparse array in scipy csr format.
        :param Md: 2 or 4 dimensional numpy.array
        :return scipy.sparse.csr_matrix
        """
        # Finding indices for non-zero elements and shape of Md.
        indices = np.array(np.where(Md != 0)).astype(int).T
        N = Md.shape[0]
        
        # Converting 2D array to csr_matrix
        if np.ndim(Md) == 2:
            return csr_matrix(Md)
        
        # Converting 4D array to csr_matrix using convert_indices from util.py.
        elif np.ndim(Md) == 4:
            row = np.array([]); col = np.array([]); data = np.array([]);
            for ind in indices:
                p,q = convert_indices(N, int(ind[0]), int(ind[1]), int(ind[2]), int(ind[3]));
                row = np.append(row,p)
                col = np.append(col,q)
                data = np.append(data,Md[tuple(ind)])
            return csr_matrix((data, (row, col)), shape=(N*N, N*N))
        
        # Return if array dimensions incompatible.
        else:
            print("Incompatible dense array dimension. Must be either 2 or 4 dimensions.")
            return

    def to_dense(self, Ms, dim=2):
        """
        Converts sparse arry of integrals in scipy csr format to dense numpy array.
        :param Ms: scipy.sparse.csr_matrix
        :param dim: target dimension of output array (either 2 or 4)
        :return: numpy.array
        """
        # return dense 2D array (default).
        if dim == 2:
            return Ms.todense()

        # Optionally, return dense 4D array for two-particle integrals.
        elif dim == 4:
            N = int(np.sqrt(Ms.shape[0]))
            vd = np.zeros([N,N,N,N])
            for p,q in np.array(Ms.nonzero()).T:
                i,j,k,l = convert_indices(N, int(p),int(q))
                vd[(i,j,k,l)] = Ms[p,q]
            return vd

        # Return if target dim is not 2 or 4.
        else:
            print("Target output dimension must be either 2 or 4.")
            return

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
        # Reduce symmetry of integral
        one_ints = self.reduse_sym(self.one_ints)

        # Write header
        nactive = one_ints.shape[0]
        print(f' &FCI NORB={nactive:d},NELEC={nelec:d},MS2={spinpol:d},', file=f)
        print(f"  ORBSYM= {','.join('1' for v in range(nactive))},", file=f)
        print('  ISYM=1', file=f)
        print(' &END', file=f)

        # Reduce symmetry of integrals
        two_ints = self.reduce_sym(self.two_ints)

        # getting nonzero elements from the 2d _sparse_ array
        p_array, q_array = two_ints.nonzero()

        # converting 2d indices to 4d indices
        N = int(np.sqrt(two_ints.shape[0]))
        for p, q in zip(p_array, q_array):
            i, j, k, l = convert_indices(N, p, q)
            j, k = k, j  # changing indexing from physical to chemical notation
            if j > i and l > k and (i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l:
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


    def save_triqs(self, fname:str, integral):
        """
        Save matrix in triqc format
        :param fname: filename
        :param integral: matrix to be saved
        :return: None
        """
        pass

    def save(self, fname: str, integral, basis):
        """Save file as regular numpy array"""
        pass
