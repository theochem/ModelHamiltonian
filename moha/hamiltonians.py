r"""Model Hamiltonian classes."""

import numpy as np

from scipy.sparse import csr_matrix, diags, lil_matrix, hstack, vstack

from .api import HamiltonianAPI

from .utils import get_atom_type, convert_indices, expand_sym

__all__ = [
    "HamPPP",
    "HamHuck",
    "HamHub",
]


class HamPPP(HamiltonianAPI):
    r"""Pariser-Parr-Pople Hamiltonian."""

    def __init__(
            self,
            connectivity: list,
            alpha=-0.414,
            beta=-0.0533,
            u_onsite=None,
            gamma=None,
            charges=None,
            sym=1,
            g_pair=None,
            atom_types=None,
            atom_dictionary=None,
            bond_dictionary=None,
            Bz=None,
    ):
        r"""
        Initialize Pariser-Parr-Pople Hamiltonian.

        The form:
        :math:`\hat{H}_{\mathrm{PPP}+\mathrm{P}}=\sum_{p q}
        h_{p q} a_{p}^{\dagger}
        a_{q}+\sum_{p} U_{p} \hat{n}_{p \alpha} \hat{n}{p\beta}+\frac{1}{2}
        \sum{p\neq q}\gamma{pq}\left(\hat{n}_{p\alpha}+
        hat{n}_{p \beta}-Q_{p}\right)\left(\hat{n}_{q \alpha}+\hat{n}_{q
        \beta}-Q_{q}\right)+
        \sum_{p \neq q} g_{p q} a_{p \alpha}^{\dagger}
        a_{p \beta}^{\dagger} a_{q \beta} a_{q \alpha}`
        Parameters
        ----------
        connectivity: list
            list of tuples that specifies sites and bonds between them
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
        g_pair: float
            g_pq term that captures interaction between electron pairs
        atom_types: list
            A list of dimension equal to the number of sites,
            specifying the atom type of each site
            If a list of atom types is specified,
            the values of alpha and beta are ignored.
        atom_dictionary: dict
            Contains information about alpha and U values for each atom type
        bond_dictionary: dict
            Contains information about beta values for each bond type
        Bz: np.ndarray
            external magnetic field
        """
        self._sym = sym
        self.n_sites = None
        self.connectivity = connectivity
        self.alpha = alpha
        self.beta = beta
        self.u_onsite = u_onsite
        self.gamma = gamma
        self.charges = charges
        self.g_pair = g_pair
        self.atom_types = atom_types
        self.atom_dictionary = atom_dictionary
        self.bond_dictionary = bond_dictionary
        self.atoms_num, self.connectivity_matrix = \
            self.generate_connectivity_matrix()
        self.zero_energy = None
        self.one_body = None
        self.two_body = None

    def generate_connectivity_matrix(self):
        r"""
        Generate connectivity matrix.

        Returns
        -------
        tuple
            (dictionary, np.ndarray)
        """
        max_site = 0
        atoms_sites_lst = []
        for atom1, atom2, bond in self.connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)
            for pair in [(atom1_name, site1), (atom2_name, site2)]:
                if pair not in atoms_sites_lst:
                    atoms_sites_lst.append(pair)
            if max_site < max(site1, site2):  # finding the max index of site
                max_site = max(site1, site2)
        self.n_sites = len(atoms_sites_lst)

        if self.atom_types is None:
            atom_types = [None for i in range(max_site + 1)]
            for atom, site in atoms_sites_lst:
                atom_types[site] = atom
            self.atom_types = atom_types
        connectivity_mtrx = np.zeros((max_site, max_site))

        for atom1, atom2, bond in self.connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)
            connectivity_mtrx[site1 - 1, site2 - 1] = bond
            # numbering of sites starts from 1

        connectivity_mtrx = np.maximum(connectivity_mtrx, connectivity_mtrx.T)
        self.connectivity_matrix = csr_matrix(connectivity_mtrx)
        return atoms_sites_lst, self.connectivity_matrix

    def generate_zero_body_integral(self):
        r"""Generate zero body integral.

        Returns
        -------
        float
        """
        if self.charges is None or self.gamma is None:
            return 0
        else:
            self.zero_energy = 0.5*self.charges@self.gamma@self.charges
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
        one_body_term = (
                diags([self.alpha for _ in range(self.n_sites)], format="csr")
                + self.beta * self.connectivity_matrix
        )

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
                    self.gamma.shape != (2 * n_sp, 2 * n_sp):
                raise TypeError("Gamma matrix has wrong basis")

            if basis == "spatial basis" and \
                    self.gamma.shape == (n_sp, n_sp):
                zeros_block = np.zeros((n_sp, n_sp))
                gamma = np.vstack(
                    [np.hstack([self.gamma, zeros_block]),
                     np.hstack([zeros_block, self.gamma])]
                )
            for p in range(n_sp):
                for q in range(n_sp):
                    if p != q:
                        i, j = convert_indices(Nv, p, q, p, q)
                        v[i, j] = 0.5*gamma[p, q]

                        i, j = convert_indices(Nv, p, q + n_sp, p, q + n_sp)
                        v[i, j] = 0.5*gamma[p, q + n_sp]

                        i, j = convert_indices(Nv, p + n_sp, q, p + n_sp, q)
                        v[i, j] = 0.5*gamma[p + n_sp, q]

                        i, j = convert_indices(Nv,
                                               p + n_sp,
                                               q + n_sp,
                                               p + n_sp,
                                               q + n_sp)
                        v[i, j] = 0.5*gamma[p + n_sp, q + n_sp]

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
            connectivity: list,
            alpha=-0.414,
            beta=-0.0533,
            u_onsite=None,
            sym=1,
            atom_types=None,
            atom_dictionary=None,
            bond_dictionary=None,
            Bz=None,
    ):
        r"""
        Hubbard Hamiltonian.

        Parameters
        ----------
        connectivity: list
            list of tuples that specifies sites and bonds between them
        alpha: float
            specifies the site energy if all sites are equivalent.
            Default value is the 2p-pi orbital of Carbon
        beta: float
            specifies the resonance energy, hopping term,
            if all bonds are equivalent.
            The default value is appropriate for a pi-bond between Carbon atoms
        u_onsite: np.ndarray
            on-site Coulomb interaction; 1d np.ndarray
        sym: int
             symmetry of the Hamiltonian: int [1, 2, 4, 8]. Default is 1
        atom_types: list
            A list of dimension equal to the number of sites
            specifying the atom type of each site
            If a list of atom types is specified,
            the values of alpha and beta are ignored.
        atom_dictionary: dict
            Contains information about alpha and U values for each atom type
        bond_dictionary: dict
            Contains information about beta values for each bond type
        Bz: np.ndarray
            external magnetic field
        """
        super().__init__(
            connectivity=connectivity,
            alpha=alpha,
            beta=beta,
            u_onsite=u_onsite,
            gamma=None,
            charges=0,
            sym=sym,
            atom_types=atom_types,
            atom_dictionary=atom_dictionary,
            bond_dictionary=bond_dictionary,
            Bz=Bz,
        )
        self.charges = np.zeros(self.n_sites)


class HamHuck(HamHub):
    r"""
    Huckle Hamiltonian.

    It can be invoked by choosing u_onsite = 0 from Hubbard hamiltonian.
    """

    def __init__(
            self,
            connectivity: list,
            alpha=-0.414,
            beta=-0.0533,
            sym=1,
            atom_types=None,
            atom_dictionary=None,
            bond_dictionary=None,
            Bz=None,
    ):
        r"""
        Huckle hamiltonian.

        Parameters
        ----------
        connectivity: list
            list of tuples that specifies sites and bonds between them
        alpha: float
            specifies the site energy if all sites are equivalent.
            Default value is the 2p-pi orbital of Carbon
        beta: float
            specifies the resonance energy, hopping term,
            if all bonds are equivalent.
            The default value is appropriate for a pi-bond between Carbon atoms
        sym: int
             symmetry of the Hamiltonian: int [1, 2, 4, 8]. Default is 1
        atom_types: list
            A list of dimension equal to the number of sites
            specifying the atom type of each site
            If a list of atom types is specified,
            the values of alpha and beta are ignored.
        atom_dictionary: dict
            Contains information about alpha and U values for each atom type
        bond_dictionary: dict
            Contains information about beta values for each bond type
        Bz: np.ndarray
            external magnetic field
        """
        super().__init__(
            connectivity=connectivity,
            alpha=alpha,
            beta=beta,
            u_onsite=0,
            gamma=None,
            charges=charges,
            sym=sym,
            atom_types=atom_types,
            atom_dictionary=atom_dictionary,
            bond_dictionary=bond_dictionary,
            Bz=Bz,
        )
        self.charges = np.zeros(self.n_sites)
