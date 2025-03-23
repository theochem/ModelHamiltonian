r"""Model Hamiltonian classes."""

import numpy as np
import warnings
import pytest


from scipy.sparse import csr_matrix, diags, lil_matrix, hstack, vstack, \
SparseEfficiencyWarning

from .api import HamiltonianAPI

from .utils import convert_indices, expand_sym, antisymmetrize_two_electron_integrals

from typing import Union

from moha.rauk.rauk import assign_rauk_parameters
from moha.rauk.PariserParr import compute_overlap
import warnings

warnings.simplefilter('always',
                      SparseEfficiencyWarning)

__all__ = [
    "HamPPP",
    "HamHuck",
    "HamHub",
    "HamHeisenberg"
]


class HamPPP(HamiltonianAPI):
    r"""Pariser-Parr-Pople Hamiltonian."""

    def __init__(
            self,
            connectivity: Union[list, np.ndarray],
            alpha=-0.414,
            beta=-0.0533,
            u_onsite=None,
            gamma=None,
            charges=None,
            sym=1,
            atom_dictionary=None,
            bond_dictionary=None,
            orbital_overlap=None,
            enforce_pg_symmetry = False,
            enforce_trs = False,
            enforce_spin_symmetry = False,
            enforce_permutational_symmetry = False,

    ):
        r"""
        Initialize Pariser-Parr-Pople Hamiltonian.

        Parameters
        ----------
        connectivity: list, np.ndarray
            list of tuples that specifies sites and bonds between them
            or symmetric np.ndarray of shape (n_sites, n_sites) that specifies
            the connectivity between sites.
            For example, for a linear chain of 4 sites, the connectivity
            can be specified as [(C1, C2, 1), (C2, C3, 1), (C3, C4, 1)]
        alpha: float
            specifies the site energy if all sites are equivalent.
            Default value is the 2p-pi orbital of Carbon
        beta: float
            specifies the resonance energy, hopping term,
            if all bonds are equivalent.
            The default value is appropriate for a pi-bond between Carbon atoms
        u_onsite: np.ndarray
            on-site Coulomb interaction; 1d np.ndarray
        gamma: np.ndarray
            parameter that specifies long-range Coulomb interaction; 2d
        charges: np.ndarray
            Charges on sites; 1d np.ndarray
        sym: int
             symmetry of the Hamiltonian: int [1, 2, 4, 8]. Default is 1

        Notes
        -----
        The Hamiltonian is given by:

        .. math::
            \begin{align}
            \hat{H}_{\mathrm{PPP}\mathrm{P}} &=
            \sum_{pq}h_{pq} a_{p}^{\dagger}a_{q} \\
            &+ \sum_{p} U_{p} \hat{n}_{p \alpha}
            \hat{n}_{p\beta} \\
            &+ \frac{1}{2}\sum_{p\neq q}\gamma_{pq}\left(\hat{n}_{p\alpha}
             + \hat{n}_{p \beta}-Q_{p}\right)
            \left(\hat{n}_{q \alpha}+\hat{n}_{q\beta}-Q_{q}\right)
            \end{align}

        """
        self._sym = sym
        self.n_sites = None
        self.connectivity = connectivity
        self.alpha = alpha
        self.beta = beta
        self.u_onsite = u_onsite
        self.gamma = gamma
        self.charges = charges
        self.atom_types = None
        self.atoms_dist = None
        self.atoms_num, self.connectivity_matrix = \
            self.generate_connectivity_matrix()
        self.zero_energy = None
        self.one_body = None
        self.two_body = None
        self.bond_dictionary = bond_dictionary
        self.atom_dictionary = atom_dictionary
        self.orbital_overlap = orbital_overlap
        self.two_body = self.generate_two_body_integral(basis="spatial basis", dense=False, sym=self._sym)
        if self.two_body is not None:
            self.two_body = antisymmetrize_two_electron_integrals(
                self.two_body,
                enforce_pg_symmetry=self.enforce_pg_symmetry,
                enforce_trs=self.enforce_trs,
                enforce_spin_symmetry=self.enforce_spin_symmetry,
                enforce_permutational_symmetry=self.enforce_permutational_symmetry
            )


    def generate_zero_body_integral(self):
        r"""Generate zero body integral.

        Returns
        -------
        float
        """
        if self.charges is None or self.gamma is None:
            self.zero_energy = 0
            return 0
        else:
            self.zero_energy = 0.5 * self.charges @ self.gamma @ self.charges
        return self.zero_energy

    def generate_one_body_integral(self, basis: str, dense: bool):
        r"""
        Generate one body integral in spatial or spin orbital basis.

        Parameters
        ----------
        basis: str
            ['spatial', 'spin orbital']
        dense: bool
            dense or sparse matrix; default False

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
        """
        # check if connectivity matrix is adjacency
        if isinstance(self.connectivity, np.ndarray):
            one_body_term = (
                diags([self.alpha for _ in range(self.n_sites)], format="csr")
                + self.beta * self.connectivity_matrix
            )
        # check if alpha and beta are different from the default or
        # all atom types are the same
        elif (
            self.alpha != -0.414 and self.beta != -0.0533
        ) or len(np.unique(self.atom_types)) == 1:
            one_body_term = (
                diags([self.alpha for _ in range(self.n_sites)])
                + self.beta * self.connectivity_matrix
            )
        # check if elements in connectivity matrix are integer
        elif np.all([isinstance(elem[2], int) for elem in self.connectivity]):
            one_body_term = assign_rauk_parameters(
                self.connectivity,
                self.atom_dictionary,
                self.bond_dictionary
            )
        # check if elements in connectivity matrix are float
        elif np.all([isinstance(elem[2], float)
                    for elem in self.connectivity]):
            one_body_term = compute_overlap(
                self.connectivity,
                self.atom_dictionary,
                self.bond_dictionary,
                self.orbital_overlap
            )
        else:
            raise TypeError("Connectivity matrix has wrong type")

        one_body_term = one_body_term.tolil()
        if (self.gamma is not None) and (self.charges is not None):
            for p in range(self.n_sites):
                for q in range(self.n_sites):
                    if p != q:
                        mult = 0.5 * self.gamma[p, q]
                        one_body_term[p, p] -= mult * self.charges[p]
                        one_body_term[q, q] -= mult * self.charges[q]

        if basis == "spatial basis":
            self.one_body = one_body_term.tocsr()
        elif basis == "spinorbital basis":
            one_body_term_spin = hstack(
                [one_body_term, csr_matrix(one_body_term.shape)], format="csr"
            )
            one_body_term_spin = vstack(
                [
                    one_body_term_spin,
                    hstack([csr_matrix(one_body_term.shape),
                            one_body_term], format="csr"),
                ],
                format="csr",
            )
            self.one_body = one_body_term_spin
        else:
            raise TypeError("Wrong basis")

        return self.one_body.todense() if dense else self.one_body

    def generate_two_body_integral(self, basis: str, dense: bool, sym=1):
        r"""
        Generate two body integral in spatial or spinorbital basis.

        Parameters
        ----------
        basis: str
            ['spatial', 'spin orbital']
        dense: bool
            dense or sparse matrix; default False
        sym: int
            symmetry -- [2, 4, 8] default is 1

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
        """
        n_sp = self.n_sites
        Nv = 2 * n_sp
        v = lil_matrix((Nv * Nv, Nv * Nv))

        if self.u_onsite is not None:
            for p in range(n_sp):
                i, j = convert_indices(Nv, p, p + n_sp, p, p + n_sp)
                v[i, j] = self.u_onsite[p]

        if self.gamma is not None:
            if basis == "spinorbital basis" and \
                    self.gamma.shape != (n_sp, n_sp):
                raise TypeError("Gamma matrix has wrong shape")

            for p in range(n_sp):
                for q in range(n_sp):
                    if p != q:
                        i, j = convert_indices(Nv, p, q, p, q)
                        v[i, j] = 0.5 * self.gamma[p, q]

                        i, j = convert_indices(Nv, p, q + n_sp, p, q + n_sp)
                        v[i, j] = 0.5 * self.gamma[p, q]

                        i, j = convert_indices(Nv, p + n_sp, q, p + n_sp, q)
                        v[i, j] = 0.5 * self.gamma[p, q]

                        i, j = convert_indices(Nv,
                                               p + n_sp,
                                               q + n_sp,
                                               p + n_sp,
                                               q + n_sp)
                        v[i, j] = 0.5 * self.gamma[p, q]

        v = v.tocsr()
        self.two_body = expand_sym(sym, v, 2)
        # converting basis if necessary
        if basis == "spatial basis":
            v = self.to_spatial(sym=sym, dense=False, nbody=2)
        elif basis == "spinorbital basis":
            pass
        else:
            raise TypeError("Wrong basis")

        self.two_body = v
        # return either sparse csr array (default) or dense N^2*N^2 array
        return self.to_dense(v, dim=4) if dense else v


class HamHub(HamPPP):
    r"""
    Hubbard Hamiltonian.

    The Hubbard model corresponds to choosing $\gamma_{pq} = 0$
    It can be invoked by choosing gamma = 0 from PPP hamiltonian.

    """

    def __init__(
            self,
            connectivity: Union[list, np.ndarray],
            alpha=-0.414,
            beta=-0.0533,
            u_onsite=None,
            sym=1,
            atom_types=None,
            atom_dictionary=None,
            bond_dictionary=None,
            orbital_overlap=None,
            Bz=None,
            gamma=None,
            enforce_pg_symmetry=False,
            enforce_trs=False,
            enforce_spin_symmetry=False,
            enforce_permutational_symmetry=False,
    ):
        r"""
        Hubbard Hamiltonian.

        Parameters
        ----------
        connectivity: list, np.ndarray
            List of tuples that specify sites and bonds between them
            or symmetric np.ndarray of shape (n_sites, n_sites) that specifies
            the connectivity between sites.
            For example, for a linear chain of 4 sites, the connectivity
            can be specified as [(C1, C2, 1), (C2, C3, 1), (C3, C4, 1)]
        alpha: float
            Specifies the site energy if all sites are equivalent.
            Default value is the 2p-pi orbital of Carbon
        beta: float
            Specifies the resonance energy, hopping term,
            if all bonds are equivalent.
            The default value is appropriate for a pi-bond between Carbon atoms
        u_onsite: np.ndarray
            On-site Coulomb interaction; 1D np.ndarray
        sym: int
            Symmetry of the Hamiltonian: int [1, 2, 4, 8]. Default is 1
        enforce_pg_symmetry: bool
            Whether to enforce point-group symmetry during antisymmetrization
        enforce_trs: bool
            Whether to enforce time-reversal symmetry
        enforce_spin_symmetry: bool
            Whether to enforce spin symmetry
        enforce_permutational_symmetry: bool
            Whether to enforce permutational symmetry

        Notes
        -----
        The Hamiltonian is given by:

        .. math::
            \hat{H}_{\mathrm{PPP}\mathrm{P}} =
            \sum_{pq}h_{pq} a_{p}^{\dagger}a_{q}
            + \sum_{p} U_{p} \hat{n}_{p \alpha} \hat{n}_{p\beta}

        """
        super().__init__(
            connectivity=connectivity,
            alpha=alpha,
            beta=beta,
            u_onsite=u_onsite,
            gamma=None,  # Set gamma to None for Hubbard Model
            atom_dictionary=atom_dictionary,
            bond_dictionary=bond_dictionary,
            orbital_overlap=orbital_overlap,
            charges=np.array(0),
            sym=sym
        )

        self.charges = np.zeros(self.n_sites)

        # Compute the two-body integrals
        self.two_body = self.generate_two_body_integral(basis="spatial", dense=False, sym=sym)

        # Apply antisymmetrization if two-body integrals exist
        if self.two_body is not None:
            self.two_body = antisymmetrize_two_electron_integrals(
                self.two_body,
                enforce_pg_symmetry=enforce_pg_symmetry,
                enforce_trs=enforce_trs,
                enforce_spin_symmetry=enforce_spin_symmetry,
                enforce_permutational_symmetry=enforce_permutational_symmetry
            )
class HamHuck(HamHub):
    r"""
    Huckel Hamiltonian.

    It can be invoked by choosing u_onsite = None from Hubbard hamiltonian.
    """

    def __init__(
            self,
            connectivity: Union[list, np.ndarray],
            alpha=-0.414,
            beta=-0.0533,
            sym=1,
            atom_dictionary=None,
            bond_dictionary=None,
            enforce_pg_symmetry=False,
            enforce_trs=False,
            enforce_spin_symmetry=False,
            enforce_permutational_symmetry=False,
    ):
        r"""
        Hückel Hamiltonian.
         Notes
        -----
        The Hamiltonian is given by:
        .. math::
            \hat{H}_{\mathrm{PPP}\mathrm{P}} =
            \sum_{pq}h_{pq} a_{p}^{\dagger}a_{q}

        Parameters
        ----------
        connectivity: list, np.ndarray
            List of tuples that specify sites and bonds between them
            or symmetric np.ndarray of shape (n_sites, n_sites) that specifies
            the connectivity between sites.
            Example: [(C1, C2, 1), (C2, C3, 1), (C3, C4, 1)]
        alpha: float
            Specifies the site energy if all sites are equivalent.
            Default value is the 2p-pi orbital of Carbon
        beta: float
            Specifies the resonance energy, hopping term,
            if all bonds are equivalent.
            The default value is appropriate for a pi-bond between Carbon atoms
        sym: int
            Symmetry of the Hamiltonian: int [1, 2, 4, 8]. Default is 1
        enforce_pg_symmetry: bool
            Whether to enforce point-group symmetry during antisymmetrization
        enforce_trs: bool
            Whether to enforce time-reversal symmetry
        enforce_spin_symmetry: bool
            Whether to enforce spin symmetry
        enforce_permutational_symmetry: bool
            Whether to enforce permutational symmetry
        """
        super().__init__(
            connectivity=connectivity,
            alpha=alpha,
            beta=beta,
            u_onsite=None,
            gamma=None,
            sym=sym,
            atom_dictionary=atom_dictionary,
            bond_dictionary=bond_dictionary,
        )

        self.charges = np.zeros(self.n_sites)

        # Compute two-body integrals using the existing method
        self.two_body = self.generate_two_body_integral(basis="spatial", dense=False, sym=sym)

        # Apply antisymmetrization if two-body integrals exist
        if self.two_body is not None:
            self.two_body = antisymmetrize_two_electron_integrals(
                self.two_body,
                enforce_pg_symmetry=enforce_pg_symmetry,
                enforce_trs=enforce_trs,
                enforce_spin_symmetry=enforce_spin_symmetry,
                enforce_permutational_symmetry=enforce_permutational_symmetry
            )
class HamHeisenberg(HamiltonianAPI):
    r"""XXZ Heisenberg Hamiltonian."""

    def __init__(self,
                 mu: np.ndarray,
                 J_eq: np.ndarray,
                 J_ax: np.ndarray,
                 connectivity: np.ndarray = None,
                 enforce_pg_symmetry=False,
                 enforce_trs=False,
                 enforce_spin_symmetry=False,
                 enforce_permutational_symmetry=False
                 ):
        r"""
        Initialize XXZ Heisenberg Hamiltonian.
         Notes
        -----
        The form of the Hamiltonian is given by:
        .. math::
            \hat{H}_{X X Z}=\sum_p\left(\mu_p^Z-J_{p p}^{\mathrm{eq}}\right)
            S_p^Z+\sum_{p q} J_{p q}^{\mathrm{ax}} S_p^Z S_q^Z+\sum_{p q}
            J_{p q}^{\mathrm{eq}} S_p^{+} S_q^{-}


        Parameters
        ----------
        mu: np.ndarray
            Zeeman term
        J_eq: np.ndarray
            J equatorial term
        J_ax: np.ndarray
            J axial term
        connectivity: np.ndarray
            Symmetric numpy array that specifies the connectivity between sites
        enforce_pg_symmetry: bool
            Whether to enforce point-group symmetry during antisymmetrization
        enforce_trs: bool
            Whether to enforce time-reversal symmetry
        enforce_spin_symmetry: bool
            Whether to enforce spin symmetry
        enforce_permutational_symmetry: bool
            Whether to enforce permutational symmetry
        """
        if connectivity is not None:
            self.connectivity = connectivity
            self.n_sites = connectivity.shape[0]
            if isinstance(J_eq, (int, float)):
                self.J_eq = J_eq * connectivity
                self.J_ax = J_ax * connectivity
                self.mu = mu * np.ones(self.n_sites)
            else:
                raise TypeError("Connectivity matrix is provided, "
                                "J_eq, J_ax, and mu should be floats")
        else:
            if isinstance(J_eq, np.ndarray) and \
               isinstance(J_ax, np.ndarray) and \
               isinstance(mu, np.ndarray) and \
               J_eq.shape == J_ax.shape and \
               mu.shape[0] == J_eq.shape[0]:
                self.n_sites = J_eq.shape[0]
                self.J_eq = J_eq
                self.J_ax = J_ax
            else:
                raise TypeError("J_eq and J_ax should be numpy arrays of the same shape")

        self.mu = np.array(mu)
        self.atom_types = None
        self.zero_energy = None
        self.one_body = None

        # Compute two-body integrals if applicable
        self.two_body = self.generate_two_body_integral(basis="spinorbital basis", dense=False, sym=1)

        # Apply antisymmetrization if two-body integrals exist
        if self.two_body is not None:
            self.two_body = antisymmetrize_two_electron_integrals(
                self.two_body,
                enforce_pg_symmetry=enforce_pg_symmetry,
                enforce_trs=enforce_trs,
                enforce_spin_symmetry_func=None,
                enforce_permutational_symmetry=enforce_permutational_symmetry
            )

    def generate_zero_body_integral(self):
        """
        Generate zero body term.

        Returns
        -------
        zero_energy: float
        """
        zero_energy = -0.5 * np.sum(self.mu - np.diag(self.J_eq)) \
            + 0.25 * np.sum(self.J_ax) / 2  # divide by 2 to avoid double counting # noqa: E501
        self.zero_energy = zero_energy
        return zero_energy

    def generate_one_body_integral(self,
                                   dense: bool,
                                   basis='spinorbital basis'):
        r"""Generate one body integral.

        Parameters
        ----------
        dense: bool
        basis: str

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
        """

        if basis == 'spatial basis':
            if self.J_ax.shape != (self.n_sites, self.n_sites):
                raise TypeError("J_ax matrix has wrong basis")
            if self.J_eq.shape != (self.n_sites, self.n_sites):
                raise TypeError("J_eq matrix has wrong basis")

            J_ax = self.J_ax
            J_eq = self.J_eq
            mu = self.mu

        elif basis == "spinorbital basis":
            if self.J_ax.shape != (2 * self.n_sites, 2 * self.n_sites) and \
                    self.J_ax.shape == (self.n_sites, self.n_sites):

                J_ax = np.hstack([np.vstack([self.J_ax,
                                             np.zeros((self.n_sites,
                                                       self.n_sites))]),
                                  np.vstack([np.zeros((self.n_sites,
                                                       self.n_sites)),
                                             self.J_ax])])
            else:
                raise TypeError("J_ax matrix has wrong basis")
            if self.J_eq.shape != (2 * self.n_sites, 2 * self.n_sites) and \
                    self.J_eq.shape == (self.n_sites, self.n_sites):

                J_eq = np.hstack([np.vstack([self.J_eq,
                                             np.zeros((self.n_sites,
                                                       self.n_sites))]),
                                  np.vstack([np.zeros((self.n_sites,
                                                       self.n_sites)),
                                             self.J_eq])])
            else:
                raise TypeError("J_eq matrix has wrong basis")

            if self.mu.shape != (2 * self.n_sites,) and\
               self.mu.shape == (self.n_sites,):
                mu = np.hstack([self.mu, self.mu])
            else:
                raise TypeError("mu array has wrong basis")

        # sum of J_ax_ij for j>i
        sum_J_ax_ij = (np.sum(J_ax, axis=1) - np.diag(J_ax)) / 2
        # if only two sites are present,
        # then no need to devide by 2 two avoid double counting
        if sum_J_ax_ij.shape[0] == 2:
            sum_J_ax_ij *= 2

        one_body_term = 0.5 * diags(mu - np.diag(J_eq) -
                                    sum_J_ax_ij,
                                    format="csr")

        self.one_body = one_body_term
        return self.one_body.todense() if dense else self.one_body

    def generate_two_body_integral(self,
                                   sym=1,
                                   dense=False,
                                   basis='spinorbital basis'):
        r"""Generate two body integral in spatial or spinorbital basis.

        Parameters
        ----------
        basis: str
            ['spinorbital basis', 'spatial basis']
        dense: bool
            dense or sparse matrix; default False
        sym: int
            symmetry -- [2, 4, 8] default is 1

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
        """

        n_sp = self.n_sites
        Nv = 2 * n_sp
        v = lil_matrix((Nv * Nv, Nv * Nv))

        if self.J_eq is not None and self.J_ax is not None:
            J_eq = self.J_eq
            J_ax = self.J_ax
            for p in range(n_sp):
                for q in range(p + 1, n_sp):
                    i, j = convert_indices(Nv, p, q, p, q)
                    v[i, j] = 0.25 * J_ax[p, q]

                    i, j = convert_indices(Nv, p, q + n_sp, p, q + n_sp)
                    v[i, j] = 0.25 * J_ax[p, q]

                    i, j = convert_indices(Nv, p + n_sp, q, p + n_sp, q)
                    v[i, j] = 0.25 * J_ax[p, q]

                    i, j = convert_indices(Nv,
                                           p + n_sp,
                                           q + n_sp,
                                           p + n_sp,
                                           q + n_sp)
                    v[i, j] = 0.25 * J_ax[p, q]

                    i, j = convert_indices(Nv, p, p + n_sp, q, q + n_sp)
                    v[i, j] = J_eq[p, q]

        v = v.tocsr()
        print(f"Debug: generate_two_body_integral called with basis='{basis}'")

        # expanding symmetry
        v = expand_sym(sym, v, 2)
        self.two_body = v
        if basis == "spatial basis":
            v = v.tolil()
            v = self.to_spatial(sym=sym, dense=False, nbody=2)
            v=v.tocsr()
        elif basis == "spinorbital basis":
            pass
        else:
            raise TypeError("Wrong basis")

        self.two_body = v
        # return either sparse csr array (default) or dense N^2*N^2 array
        return self.to_dense(v, dim=4) if dense else v


class HamIsing(HamHeisenberg):
    r"""Ising Hamiltonian."""

    def __init__(self,
                 mu: np.ndarray,
                 J_ax: np.ndarray,
                 connectivity: np.ndarray = None
                 ):
        r"""Initialize XXZ Heisenberg Hamiltonian.

        Parameters
        ----------
        mu: np.ndarray
            Zeeman term
        J_ax: np.ndarray
            J axial term
        connectivity: np.ndarray
            symmetric numpy array that specifies the connectivity between sites

        Notes
        -----
        The form of the Hamiltonian is given by:

        .. math::
            \hat{H}_{Ising}=\sum_p\mu_p^Z
            S_p^Z+\sum_{p q} J_{p q}^{\mathrm{ax}} S_p^Z S_q^Z
        """
        if isinstance(connectivity, csr_matrix):  #
            connectivity = connectivity.tolil()
        if isinstance(J_ax, float):
            J_eq = 0
        elif isinstance(J_ax, np.ndarray):
            J_eq = np.zeros(J_ax.shape)
        else:
            raise TypeError("J_ax should be a float or a numpy array")

        super().__init__(
            mu=mu,
            J_eq=J_eq,
            J_ax=J_ax,
            connectivity=connectivity
        )


class HamRG(HamHeisenberg):
    r"""Richardson-Gaudin Hamiltonian."""

    def __init__(self,
                 mu: np.ndarray,
                 J_eq: np.ndarray,
                 connectivity: np.ndarray = None
                 ):
        r"""Initialize XXZ Heisenberg Hamiltonian.

        Parameters
        ----------
        mu: np.ndarray
            Zeeman term
        J_eq: np.ndarray
            J equatorial term
        connectivity: np.ndarray

        Notes
        -----
        The form of the Hamiltonian is given by:

        .. math::
            \hat{H}_{RG}=\sum_p\left(\mu_p^Z-J_{p p}^{\mathrm{eq}}\right)
            S_p^Z+\sum_{p q} J_{p q}^{\mathrm{eq}} S_p^{+} S_q^{-}

        """
        # if J_eq is a float or numpy float
        if isinstance(J_eq, float):
            J_ax = 0
        elif isinstance(J_eq, np.ndarray):
            J_ax = np.zeros(J_eq.shape)
        else:
            raise TypeError("J_ax should be a float or a numpy array")

        super().__init__(
            mu=mu,
            J_eq=J_eq,
            J_ax=J_ax,
            connectivity=connectivity
        )
