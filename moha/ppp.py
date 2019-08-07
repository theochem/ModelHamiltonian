"""Pariser-Parr-Pople Hamiltonian Transformation tools."""

import numpy as np
from scipy.sparse import dok_matrix


class PPPHamiltonian:
    r"""Pariser-Parr-Pople model Hamiltonian.

    ..math::

        \hat{H}_{\text{PPP}} = H_{\text{energy}} + H_{\text{hopping}} + H_{\text{interaction}} +
        H_{\text{off-site repulsion}},

    where :math:`H_{energy} = \sum_{i, \sigma} \alpha_{i} a_{i \sigma}^{\dagger} a_{i \sigma}`,
    :math:`H_{hopping} =
    \sum_{\langle i, j \rangle, \sigma} \beta_{ij} a_{i \sigma}^{\dagger} a_{i \sigma}`,
    :math:`H_{interaction} = \sum_{i} U_{i} a_{i \alpha}^{\dagger} a_{i \alpha}
    a_{i \beta}^{\dagger} a_{i \beta}`, and
    :math:`H_{off-site repulsion} = \sum_{\langle i, j \rangle} V_{ij} (a_{i \alpha}^{\dagger}
    a_{i \alpha} + a_{i \beta}^{\dagger} a_{i \beta})(a_{j \alpha}^{\dagger} a_{j \alpha} +
    a_{j \beta}^{\dagger} a_{j \beta})`.

    Attributes
    ----------
    one_electron_matrix : np.ndarray(K, K)
        The one-electron component :math:`h_{pq}` of the generalized Hamiltonian, where K is the
        number of sites in the lattice structure. This matrix is the spin-independent one-electron
        matrix, to be transformed into a 2K by 2K matrix to account for both spin-orbitals on each
        site.
    two_electron_matrix : np.sparse.dok_matrix(K, K)
        The two-electron component :math:`V_{pqrs}` of the generalized Hamiltonian, where K is the
        number of sites in the lattice structure. In this representation, only two indices are
        stored, as only the cases where :math:`p = r` and :math:`q = s` are nonzero.

    Notes
    -----
    For a lattice of K sites, there are 2K spin-orbitals. By convention, the first k sites will be
    spin-up and the next k sites will be spin down; thus, if :math:`i \alpha` corresponds to index
    :math:`p = i`, then :math:`i \beta`, corresponds to index :math:`\bar{p} = i + s`.

    """

    def __init__(self, alpha, beta, u_matrix, v_matrix):
        # TODO: Update docstring
        """Generate the transformation from PPP Hamiltonian to general Hamiltonian.

        Parameters
        ----------
        alpha : np.ndarray(K,)
            The core energy at each site k.
        beta : np.ndarray(K, K)

        u_matrix : np.ndarray(K,)

        v_matrix : np.ndarray(K, K)

        """
        if isinstance(alpha, np.ndarray):
            self.k = alpha.shape[0]
        elif isinstance(beta, np.ndarray):
            self.k = beta.shape[0]
        elif isinstance(u_matrix, np.ndarray):
            self.k = u_matrix.shape[0]
        elif isinstance(v_matrix, np.ndarray):
            self.k = v_matrix.shape[0]
        else:
            raise ValueError("No PPP parameters were defined")
        self.one_electron_matrix = self.generate_one_elec_matrix(alpha, beta)
        self.two_electron_matrix = self.generate_two_elec_matrix(u_matrix, v_matrix)

    def generate_one_elec_matrix(self, alpha, beta):
        """Generate the compact one-electron matrix for the general Hamiltonian.

        Parameters
        ----------
        alpha : np.ndarray(K,)
            The core energy at each site k.
        beta : np.ndarray(K, K)

        Returns
        -------
        one_electron_matrix : np.ndarray(K, K)
            The compact one-electron matrix of the PPP Hamiltonian.

        """
        return np.diag(alpha) + beta

    def generate_two_elec_matrix(self, u_matrix, v_matrix):
        """Generate the compact two-electron matrix for the general Hamiltonian.

        Parameters
        ----------
        u_matrix : np.ndarray(K,)

        v_matrix : np.ndarray(K, K)

        Returns
        -------
        two_electron_matrix : np.sparse.dok_matrix(K, K)
            The sparse two-electron matrix of the PPP Hamiltonian, :math:`V_{p < q, r < s}`. In
            this representation, only two indices are stored, as only the cases where :math:`p = r`
            and :math:`q = s` are nonzero.

        Notes
        -----
        For a lattice of K sites, there are 2K spin-orbitals. By convention, the first k sites will be
        spin-up and the next k sites will be spin down; thus, if :math:`i \alpha` corresponds to index
        :math:`p = i`, then :math:`i \beta`, corresponds to index :math:`\bar{p} = i + s`.
        75 sites (150 total indices) requires more than 4 GB of memory in order to instantiate the
        dense numpy array.

        """
        two_electron_matrix = dok_matrix((2 * self.k, 2 * self.k))
        for i in range(self.k):
            two_electron_matrix[i, i+self.k] = u_matrix[i]
            two_electron_matrix[i + self.k, i] = u_matrix[i]
        for i in range(self.k):
            for j in range(i, self.k):
                v_ij = v_matrix[i, j]
                two_electron_matrix[i, j] = v_ij
                two_electron_matrix[j, i] = v_ij

                two_electron_matrix[i, j+self.k] = v_ij
                two_electron_matrix[j+self.k, i] = v_ij

                two_electron_matrix[i+self.k, j] = v_ij
                two_electron_matrix[j, i+self.k] = v_ij

                two_electron_matrix[i+self.k, j+self.k] = v_ij
                two_electron_matrix[j+self.k, i+self.k] = v_ij
        return two_electron_matrix

    def return_compact(self):
        """Return the compact one- and two-electron matrices of the PPP Hamiltonian.

        Returns
        -------
        # TODO: fix docstring
        compact_matrices : tuple of np.ndarray
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
            two_electron_matrix[key + key] = self.two_electron_matrix[key]
        return one_electron_matrix, two_electron_matrix


class HubbardHamiltonian(PPPHamiltonian):
    """Hubbard model Hamiltonian.

    Attributes
    ----------

    """
    def __init__(self, alpha, beta, u_matrix):
        # TODO: Update docstring
        """Generate the transformation from Hubbard Hamiltonian to general Hamiltonian.

        Parameters
        ----------
        alpha : np.ndarray(K,)
            The core energy at each site k.
        beta : np.ndarray(K, K)

        u_matrix : np.ndarray(K,)

        """
        super().__init__(alpha, beta, u_matrix, None)

    def generate_two_elec_matrix(self, u_matrix, v_matrix):
        """Generate the compact two-electron matrix for the general Hamiltonian.

        Parameters
        ----------
        u_matrix : np.ndarray(K,)

        v_matrix : None

        Returns
        -------
        two_electron_matrix : np.sparse.dok_matrix(K, K)
            The sparse two-electron matrix of the Hubbard Hamiltonian, :math:`V_{p < q, r < s}`. In
            this representation, only two indices are stored, as only the cases where :math:`p = r`
            and :math:`q = s` are nonzero.
            For the Huckel Hamiltonian, the two-electron integrals are all zero.

        Notes
        -----
        For a lattice of K sites, there are 2K spin-orbitals. By convention, the first k sites will be
        spin-up and the next k sites will be spin down; thus, if :math:`i \alpha` corresponds to index
        :math:`p = i`, then :math:`i \beta`, corresponds to index :math:`\bar{p} = i + s`.
        75 sites (150 total indices) requires more than 4 GB of memory in order to instantiate the
        dense numpy array.

        """
        two_electron_matrix = dok_matrix((2*self.k, 2*self.k))
        for i in range(self.k):
            two_electron_matrix[i, i + self.k] = u_matrix[i]
            two_electron_matrix[i + self.k, i] = u_matrix[i]
        return two_electron_matrix


class HuckelHamiltonian(PPPHamiltonian):
    """Huckel model Hamiltonian.

    Attributes
    ----------


    """
    def __init__(self, alpha, beta):
        # TODO: Update docstring
        """Generate the transformation from Huckel Hamiltonian to general Hamiltonian.

        Parameters
        ----------
        alpha : np.ndarray(K,)
            The core energy at each site k.
        beta : np.ndarray(K, K)

        """
        super().__init__(alpha, beta, None, None)

    def generate_two_elec_matrix(self, u_matrix, v_matrix):
        """Generate the compact two-electron matrix for the general Hamiltonian.

        Parameters
        ----------
        u_matrix : None

        v_matrix : None

        Returns
        -------
        two_electron_matrix : np.sparse.dok_matrix(K, K)
            The sparse two-electron matrix of the Hubbard Hamiltonian, :math:`V_{p < q, r < s}`. In
            this representation, only two indices are stored, as only the cases where :math:`p = r`
            and :math:`q = s` are nonzero.

        Notes
        -----
        For a lattice of K sites, there are 2K spin-orbitals. By convention, the first k sites will be
        spin-up and the next k sites will be spin down; thus, if :math:`i \alpha` corresponds to index
        :math:`p = i`, then :math:`i \beta`, corresponds to index :math:`\bar{p} = i + s`.
        75 sites (150 total indices) requires more than 4 GB of memory in order to instantiate the
        dense numpy array.

        """
        return dok_matrix((2*self.k, 2*self.k))


class IsingHamiltonian(PPPHamiltonian):
    """Ising model Hamiltonian

    Parameters
    ----------

    """
    def __init__(self, alpha, v_matrix):
        super().__init__(alpha, None, None, v_matrix)

    def generate_one_elec_matrix(self, alpha, beta):
        """Generate the compact one-electron matrix for the general Hamiltonian.

        Parameters
        ----------
        alpha : np.ndarray(K,)
            The core energy at each site k.
        beta : None

        Returns
        -------
        one_electron_matrix : np.ndarray(K, K)
            The compact one-electron matrix of the PPP Hamiltonian.

        """
        return np.diag(alpha)

    def generate_two_elec_matrix(self, u_matrix, v_matrix):
        """Generate the compact two-electron matrix for the general Hamiltonian.

        Parameters
        ----------
        u_matrix : None

        v_matrix : np.ndarray(K, K)

        Returns
        -------
        two_electron_matrix : np.sparse.dok_matrix(K, K)
            The sparse two-electron matrix of the PPP Hamiltonian, :math:`V_{p < q, r < s}`. In
            this representation, only two indices are stored, as only the cases where :math:`p = r`
            and :math:`q = s` are nonzero.

        Notes
        -----
        For a lattice of K sites, there are 2K spin-orbitals. By convention, the first k sites will be
        spin-up and the next k sites will be spin down; thus, if :math:`i \alpha` corresponds to index
        :math:`p = i`, then :math:`i \beta`, corresponds to index :math:`\bar{p} = i + s`.
        75 sites (150 total indices) requires more than 4 GB of memory in order to instantiate the
        dense numpy array.

        """
        two_electron_matrix = dok_matrix((2*self.k, 2*self.k))
        for i in range(self.k):
            for j in range(i, self.k):
                v_ij = v_matrix[i, j]
                two_electron_matrix[i, j] = v_ij
                two_electron_matrix[j, i] = v_ij

                two_electron_matrix[i, j+self.k] = v_ij
                two_electron_matrix[j+self.k, i] = v_ij

                two_electron_matrix[i+self.k, j] = v_ij
                two_electron_matrix[j, i+self.k] = v_ij

                two_electron_matrix[i+self.k, j+self.k] = v_ij
                two_electron_matrix[j+self.k, i+self.k] = v_ij
        return two_electron_matrix
