from abc import ABC, abstractmethod

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
        :param sym: symmetry -- [2, 4, 8]
        :param basis: basis -- ['spatial', 'spin orbital']
        ç
        :return: numpy.ndarray or scipy.sparse.csc_matrix
        """
        pass

    @abstractmethod
    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        """
        Generates two body integral in spatial or spinorbital basis
        :param sym: symmetry -- [2, 4, 8] default None
        :param basis: basis -- ['spatial', 'spin orbital']
        :param dense: dense or sparse matrix; default dense
        :return: numpy.ndarray or scipy.sparse.csc_matrix
        """
        pass

    @abstractmethod
    def to_spatial(self, integral: np.ndarray, sym:int, dense: bool):
        """
        Converts one-/two- integral matrix from spin-orbital to spatial basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default None
        :param dense: dense or sparse matrix; default dense
        :return:
        """
        pass

    @abstractmethod
    def to_spinorbital(self, integral: np.ndarray, sym: int, dense: bool):
        """
        Converts one-/two- integral matrix from spatial to spin-orbital basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default None
        :param dense: dense or sparse matrix; default dense
        :return:
        """
        pass

    @abstractmethod
    def save_fcidump(self, fname:str, integral):
        """
        Save matrix in fcidump format
        :param fname: filename
        :param integral: matrix to be saved
        :return: None
        """
        pass

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
