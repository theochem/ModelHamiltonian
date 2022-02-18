from Hamiltonian import HamiltonianAPI
import numpy as np


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

    def generate_zero_body_integral(self):
        pass

    def generate_one_body_integral(self, sym: int, basis: str, dense: bool):
        pass

    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        pass
