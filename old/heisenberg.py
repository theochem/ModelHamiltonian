"""Heisenberg model Hamiltonian Transformation Tools."""

from collections import defaultdict
import numpy as np
from scipy.sparse import dok_matrix


class HeisenbergHamiltonian:
    r"""Heisenberg model Hamiltonian.

    Attributes
    ----------
    zero_energy : float

    one_electron_matrix : np.ndarray(K, K)

    two_electron_matrix : dict
        (p, q, r, s) -> float

    Methods
    -------


    Notes
    -----


    """
    def __init__(self, e_matrix, g_matrix, h_matrix):
        """Generate the transformation from Heisenberg Hamiltonian to general Hamiltonian.

        Parameters
        ----------
        e_matrix : np.ndarray(K,)
            ...
        g_matrix : np.ndarray(K,K)
            ...
        h_matrix : np.ndarray(K, K)
            ...

        """
        if isinstance(e_matrix, np.ndarray):
            self.k = e_matrix.shape[0]
        elif isinstance(g_matrix, np.ndarray):
            self.k = g_matrix.shape[0]
        elif isinstance(h_matrix, np.ndarray):
            self.k = h_matrix.shape[0]
        else:
            raise ValueError("No Heisenberg parameters were defined")
        self.zero_energy = self.generate_zero_energy(e_matrix, h_matrix)
        self.one_electron_matrix = self.generate_one_elec_matrix(e_matrix, h_matrix)
        self.two_electron_matrix = self.generate_two_elec_matrix(g_matrix, h_matrix)

    def generate_zero_energy(self, e_matrix, h_matrix):
        """."""
        return np.sum(e_matrix * -1/2) + np.sum(h_matrix * 1/4)

    def generate_one_elec_matrix(self, e_matrix, h_matrix):
        """Generate the compact one-electron matrix for the general Hamiltonian.

        Parameters
        ----------
        e_matrix : np.ndarray(K,)
            ...
        h_matrix : np.ndarray(K, K)
            ...

        """
        return np.diag(e_matrix/2 + (h_matrix.sum(0)+h_matrix.sum(1))/4)

    # FIXME: this outputs a dict, while ppp uses a DOK.
    def generate_two_elec_matrix(self, g_matrix, h_matrix):
        """Generate the compact two-electron matrix for the general Hamiltonian.


        """
        two_electron_matrix = defaultdict(float)
        for i in range(self.k):
            for j in range(i, self.k):
                h_ij = h_matrix[i, j]/4
                two_electron_matrix[i, j, i, j] += h_ij
                two_electron_matrix[j, i, j, i] += h_ij

                two_electron_matrix[i, j+self.k, i, j+self.k] += h_ij
                two_electron_matrix[j+self.k, i, j+self.k, i] += h_ij

                two_electron_matrix[i+self.k, j, i+self.k, j] += h_ij
                two_electron_matrix[j, i+self.k, j, i+self.k] += h_ij

                two_electron_matrix[i+self.k, j+self.k, i+self.k, j+self.k] += h_ij
                two_electron_matrix[j+self.k, i+self.k, j+self.k, i+self.k] += h_ij

                g_ij = g_matrix[i, j]
                two_electron_matrix[i, i + self.k, j, j+self.k] += g_ij
                two_electron_matrix[j, j+self.k, i, i+self.k] += g_ij

                two_electron_matrix[i, i+self.k, j+self.k, j] += g_ij
                two_electron_matrix[j + self.k, j, i, i + self.k] += g_ij

                two_electron_matrix[i + self.k, i, j, j+self.k] += g_ij
                two_electron_matrix[j, j + self.k, i + self.k, i] += g_ij

                two_electron_matrix[i + self.k, i, j + self.k, j] += g_ij
                two_electron_matrix[j + self.k, j, i + self.k, i] += g_ij
        return dict(two_electron_matrix)

    def return_compact(self):
        """Return the compact one- and two-electron matrices of the Heisenberg Hamiltonian.

        Returns
        -------
        # TODO: fix docstring
        compact_matrices : tuple of dict, np.ndarray
            The compact one- and two-electron matrices :math:`h_{pq}` and :math:`V_{pqrs}`,
            respectively.

        """
        return self.one_electron_matrix, self.two_electron_matrix

    def return_full(self):
        r"""Return the full one- and two-electron matrices of the PPP Hamiltonian.

        Returns
        -------
        full_matrices : tuple of np.ndarray
            The one- and two-electron matrices :math:`h_{pq}` and :math:`V_{pqrs}`, respectively.

        Notes
        -----
        For a lattice of K sites, there are 2K spin-orbitals. By convention, the first k sites will
        be spin-up and the next k sites will be spin down; thus, if :math:`i \alpha` corresponds to
        index :math:`p = i`, then :math:`i \beta`, corresponds to index :math:`\bar{p} = i + s`.

        Warnings
        --------
        This method will attempt to allocate the full two-electron matrix (2K, 2K, 2K, 2K).
        Due to Python's issues with memory-related error handling, this method will fail without
        warning if the array is too large. To check if this is a concern, a quick back-of-the-
        envelope calculation for total array size is 8 bytes * (2 * K) ** 4.
        75 sites (150 total orbitals) is over 4 GB of memory.
        """
        one_electron_matrix = np.zeros((2*self.k, 2*self.k))
        one_electron_matrix[:self.k, :self.k] = self.one_electron_matrix
        one_electron_matrix[self.k:, self.k:] = self.one_electron_matrix
        two_electron_matrix = np.zeros((2 * self.k, 2 * self.k, 2 * self.k, 2 * self.k))
        for key in self.two_electron_matrix.keys():
            two_electron_matrix[key] = self.two_electron_matrix[key]
        return one_electron_matrix, two_electron_matrix


class RichardsonHamiltonian(HeisenbergHamiltonian):
    """."""
    def __init__(self, e_matrix, g):
        self.k = e_matrix.shape[0]
        super().__init__(e_matrix, np.ones((self.k, self.k)) * g, np.zeros((self.k, self.k)))




