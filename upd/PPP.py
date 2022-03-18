from Hamiltonian import HamiltonianAPI
from utils import get_atom_type, convert_indices
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, diags


class HamPPP(HamiltonianAPI):
    def __init__(self, connectivity: list, alpha=-0.414, beta=-0.0533, u_onsite=None, gamma=0.0784, charges=0.417,
                 sym=None, g_pair=None, atom_types=None, atom_dictionary=None, bond_dictionary=None, Bz=None):
        """
        Initialize Pariser-Parr-Pople Hamiltonian in the form:
        $\hat{H}_{\mathrm{PPP}+\mathrm{P}}=\sum_{p q} h_{p q} a_{p}^{\dagger} a_{q}+\
         \sum_{p} U_{p} \hat{n}_{p \alpha} \hat{n}{p\beta}+\frac{1}{2} \sum{p\neq q}\gamma{pq}\left(\hat{n}_{p\alpha}+
         \hat{n}_{p \beta}-Q_{p}\right)\left(\hat{n}_{q \alpha}+\hat{n}_{q \beta}-Q_{q}\right)+
         \sum_{p \neq q} g_{p q} a_{p \alpha}^{\dagger} a_{p \beta}^{\dagger} a_{q \beta} a_{q \alpha}$

        :param connectivity: list of tuples that specifies sites and bonds between them: list
        :param alpha: specifies the site energy if all sites are equivalent. Default value is the 2p-pi orbital of
                      Carbon: float
        :param beta: specifies the resonance energy, hopping term, if all bonds are equivalent.
                     The default value is appropriate for a pi-bond between Carbon atoms: float
        :param u_onsite: on-site Coulomb interaction: 1d np.ndarray
        :param gamma: parameter that specifies long-range Coulomb interaction: 2d np.ndarray
        :param charges: Charges on sites: 1d np.ndarray
        :param sym: symmetry of the Hamiltonian: int [2, 4, 8] or None. Default is None
        :param g_pair:  g_pq term that captures interaction between electron pairs
        :param atom_types: A list of dimension equal to the number of sites specifying the atom type of each site
                           If a list of atom types is specified, the values of alpha and beta are ignored.
        :param atom_dictionary: Contains information about alpha and U values for each atom type: dict
        :param bond_dictionary: Contains information about beta values for each bond type: dict
        :param Bz:
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
        self.atoms_num, self.connectivity_matrix = self.generate_connectivity_matrix()
        self.zero_energy = None
        self.one_body = None
        self.two_body = None

    def generate_connectivity_matrix(self):
        """
        Generates connectivity matrix
        :return: dictionary in which np.ndarray
        """
        max_site = 0
        atoms_sites_lst = []
        for atom1, atom2, bond in self.connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)

            atoms_sites_lst.append((atom1_name, site1))
            atoms_sites_lst.append((atom2_name, site2))
            if max_site < max(site1, site2):  # finding the maximum index of site
                max_site = max(site1, site2)
        self.n_sites = len(atoms_sites_lst)

        if self.atom_types is None:
            atom_types = [None for i in range(max_site+1)]
            for atom, site in atoms_sites_lst:
                atom_types[site] = atom
            self.atom_types = atom_types
        connectivity_mtrx = np.zeros((max_site, max_site))

        for atom1, atom2, bond in self.connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)
            connectivity_mtrx[site1-1, site2-1] = bond ##numbering of sites should start from 1

        connectivity_mtrx = np.maximum(connectivity_mtrx, connectivity_mtrx.T)
        self.connectivity_matrix = csr_matrix(connectivity_mtrx)
        return atoms_sites_lst, self.connectivity_matrix

    def generate_zero_body_integral(self):
        if self.charges is None:
            return 0
        self.zero_energy = np.sum(np.outer(self.charges, self.charges)) - np.dot(self.charges, self.charges)
        return self.zero_energy

    def generate_one_body_integral(self, sym: int, basis: str, dense: bool):
        one_body_term = diags(
            [self.alpha for _ in range(self.n_sites)],
            format="csr") + self.beta * self.connectivity_matrix

        if (self.gamma is not None) and (self.charges is not None):
            for p in range(self.n_sites):
                for q in range(self.n_sites):
                    if p != q:
                        one_body_term[p, p] -= 2 * self.gamma[p, q] * self.charges[p]
                        one_body_term[q, q] -= 2 * self.gamma[p, q] * self.charges[q]
        if basis == 'spatial basis':
            self.one_body = one_body_term
        elif basis == 'spinorbital basis':
            one_body_term_spin = scipy.sparse.hstack([one_body_term, csr_matrix(one_body_term.shape)], format='csr')
            one_body_term_spin = scipy.sparse.vstack([one_body_term_spin,
                                                     scipy.sparse.hstack([csr_matrix(one_body_term.shape), one_body_term],
                                                                         format='csr')],
                                                     format='csr')
            self.one_body = one_body_term_spin
        else:
            raise TypeError("Wrong basis")

        return self.one_body.todense() if dense else self.one_body

    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        n_sp = self.n_sites
        Nv = 2*n_sp
        v = csr_matrix((Nv*Nv, Nv*Nv))

        if self.u_onsite is not None:
            for p in range(n_sp):
                i,j = convert_indices(Nv, p, p+n_sp, p+n_sp, p) 
                v[i,j] = self.u_onsite[p]

        if self.gamma is not None:
            for p in range(n_sp):
                for q in range(n_sp):
                    if p != q:
                        i,j = convert_indices(Nv, p, q, p, q)
                        v[i,j] = self.gamma[p, q]

                        i,j = convert_indices(Nv, p, q+n_sp, p, q+n_sp)
                        v[i,j] = self.gamma[p, q+n_sp]
   
                        i,j = convert_indices(Nv, p+n_sp, q, p+n_sp, q)
                        v[i,j] = self.gamma[p+n_sp, q]
   
                        i,j = convert_indices(Nv, p+n_sp, q+n_sp, p+n_sp, q+n_sp)
                        v[i,j] = self.gamma[p+n_sp, q+n_sp]
        return 0.5*v