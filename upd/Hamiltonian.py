from abc import ABC, abstractmethod
from typing import TextIO
import numpy as np


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
        pass

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