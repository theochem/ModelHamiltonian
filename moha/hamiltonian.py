""" Model Hamiltonian Operator classes."""
import numpy as np
from itertools import combinations


class Hamiltonian:
    """Hamiltonian super class."""

    def __init__(self):
        """Hamiltonian.

        Parameters
        ----------

        """
        pass


class HubbardHamiltonian(Hamiltonian):
    r"""Hubbard model Hamiltonian.

    ..math::

        H = H_{energy} + H_{hopping} + H_{interaction}

    H_energy : zero electron term scaled by :math:`\epsilon_0`
    H_hopping : one electron term between nearest-neighbour sites scaled by :math:`t`
    H_interaction : two electron term for electrons on same site scaled by :math:`U`

    """

    def __init__(self, lattice, n_electrons, t, u):
        """Hubbard Hamiltonian:

        Parameters
        ----------
        lattice : Lattice
            Lattice class with an orbital on each site

        n_electrons : int
            The number operator N, corresponding to the number of electrons in the state

        t : float
            The hopping amplitude for electrons hopping to nearest-neighbour sites

        u : float
            The repulsion energy between two electrons on the same site (clearly of opposite spins)
        """
        super().__init__()
        self.lattice = lattice
        self.N = n_electrons
        self.t = t
        self.U = u

        # Construct the Hilbert space for the Hamiltonian
        self.space = HilbertSpace(self.N, self.lattice.Nsites)
        self.basis = self.space.states
        self.n_basis_states = self.basis.shape[0]
        self.int_to_state = {basis: i for i, basis in enumerate(self.basis)}

        # Initialize Hamiltonian components
        self.H_interaction = np.zeros((self.n_basis_states, self.n_basis_states))
        self.H_hopping = np.zeros((self.n_basis_states, self.n_basis_states))
        self.H_energy = np.zeros((self.n_basis_states, self.n_basis_states))
        self.H_total = np.zeros((self.n_basis_states, self.n_basis_states))

    def n_down(self, on_vector):
        """n_down operator: number of spin down e-."""
        k = self.lattice.Nsites
        return bin(on_vector)[-k:].count("1")

    @staticmethod
    def x_operator(state, i, j, l, spin):
        if spin == "down":
            return state + 2 ** l * (2 ** j - 2 ** i)
        else:
            return state + 2 ** j - 2 ** i

    def calc(self):
        # Interaction term

        # Determine the ON vectors for each basis state
        on_down = self.basis >> self.lattice.Nsites  # The first half of I is I(spin down)
        on_up = self.basis % 2 ** self.lattice.Nsites  # The second half of I is I(spin up)

        # Count the number of sites with 2 electrons
        interactions = list(
            map(lambda x: bin(x).count("1"), on_down & on_up)
        )  # Count the ones in the bin string repr.
        self.H_interaction = self.U * np.diag(interactions)

        # Hopping term

        # Determine which basis states have an allowed i -> n hop
        hops = []
        for i in range(self.lattice.Nsites):
            for n in self.lattice.sites[
                i
            ].neighbours:  # Should be a way to avoid redundancy here...
                hop_spin_down = (((2 ** n ^ on_down) & (~on_down)) >> n) & ((2 ** i & on_down) >> i)
                hop_spin_up = (((2 ** n ^ on_up) & (~on_up)) >> n) & ((2 ** i & on_up) >> i)
                hop_up = list(
                    map(lambda x: self.x_operator(x, i, n, self.lattice.Nsites, "up"), self.basis)
                )
                hop_down = list(
                    map(lambda x: self.x_operator(x, i, n, self.lattice.Nsites, "down"), self.basis)
                )
                hop_up *= hop_spin_up
                hop_down *= hop_spin_down
                [hops.append((y, z)) for y, z in zip(hop_down, self.basis) if y != 0 and z != 0]
                [hops.append((y, z)) for y, z in zip(hop_up, self.basis) if y != 0 and z != 0]
        print(hops)

        for r, c in hops:
            self.H_hopping[self.int_to_state[r], self.int_to_state[c]] = self.t

        # Energy term

        self.H_total = self.H_interaction + self.H_hopping + self.H_energy


class PPPHamiltonian(Hamiltonian):
    r"""Pariser-Parr-Pople model Hamiltonian.

    ..math::

        \hat{H}_{\text{PPP}} = H_{\text{energy}} + H_{\text{hopping}} + H_{\text{interaction}} +
        H_{\text{off-site repulsion}},

    where :math:`H_{energy} = \sum_{i, \sigma} \alpha_{i} a_{i \sigma}^{\dagger} a_{i \sigma}`,
    :math:`H_{hopping} =
    \sum_{\langle i, j \rangle, \sigma} \beta_{ij} a_{i \sigma}^{\dagger} a_{i \sigma}`,
    :math:`H_{interaction} = \sum_{i} U_{i} a_{i \alpha}^{\dagger} a_{i \alpha}
    a_{i \beta}^{\dagger} a_{i \beta}`, and
    :math:`H_{off-site repulsion} = \sum_{\langle i, j \rangle} U_{ij} (a_{i \alpha}^{\dagger}
    a_{i \alpha} + a_{i \beta}^{\dagger} a_{i \beta} - 1)(a_{j \alpha}^{\dagger} a_{j \alpha} +
    a_{j \beta}^{\dagger} a_{j \beta} - 1)`.

    Attributes
    ----------

    """

    # FIXME: change alpha, beta, u_matrix to useful names
    # FIXME: using binary limits this usage to 15 lattice sites, need to modify indexing
    # TODO: utilize the fact that H_PPP is invariant under electron-hole transformation
    def __init__(self, lattice, n_electrons, alpha, beta, u_matrix, off_site):
        """Initialize.

        Parameters
        ----------
        lattice : Lattice
        n_electrons : int
        alpha : np.ndarray(L,)
            The alpha matrix, corresponding to the ground state energy?? for each lattice site.
        beta : np.ndarray(L, L)
            The beta matrix, corresponding to the hopping energy in some way.
        u_matrix : np.ndarray(L,)
            The electron-electron repulsion integral approximation.
        off_site : np.ndarray(L, L)
            The off-site repulsion parameters.

        """
        super().__init__()
        self.lattice = lattice
        # FIXME: awful var name
        self.N = n_electrons
        # Reshape to align alpha matrix with Dimension 1
        self.alpha = alpha[None, :, None]
        self.beta = beta
        self.u_matrix = u_matrix
        self.u_ij = off_site[None, :, :]

        # Construct the Hilbert space for the Hamiltonian
        self.space = HilbertSpace(self.N, self.lattice.n_sites)
        self.basis = self.space.states
        self.n_basis_states = self.basis.shape[0]
        self.int_to_state = {basis: i for i, basis in enumerate(self.basis)}

        # Initialize Hamiltonian components
        # TODO: make Hamiltonian components' wording consistent
        self.H_interaction = np.zeros((self.n_basis_states, self.n_basis_states))
        self.H_hopping = np.zeros((self.n_basis_states, self.n_basis_states))
        self.H_energy = np.zeros((self.n_basis_states, self.n_basis_states))
        self.H_off_site_repulsion = np.zeros((self.n_basis_states, self.n_basis_states))
        self.H_total = np.zeros((self.n_basis_states, self.n_basis_states))

        # Build a matrix of n all operator results
        # Dimension 0: Basis state (position in self.basis) (size: L choose N)
        # Dimension 1: Lattice site number (position in lattice) (size: L)
        # Dimension 2: Spin state (0: spin down, 1: spin up) (size: 2)
        self.n_matrix = np.zeros((self.n_basis_states, self.lattice.n_sites, 2), dtype=bool)
        self.on_down = [state >> self.lattice.n_sites for state in self.basis]
        # Read binary right-to-left so that site 0 is index 0
        # FIXME: clean up format strings with f"{x:0{lattice}}"
        self.n_matrix[:, :, 0] = np.array(
            [
                list(map(int, list(format(x, "#0{}b".format(self.lattice.n_sites + 2))[-1:1:-1])))
                for x in self.on_down
            ]
        )
        self.on_up = [state % 2 ** self.lattice.n_sites for state in self.basis]
        self.n_matrix[:, :, 1] = np.array(
            [
                list(map(int, list(format(x, "#0{}b".format(self.lattice.n_sites + 2))[-1:1:-1])))
                for x in self.on_up
            ]
        )

        # Calculate H_energy
        self.H_energy = np.diag((self.alpha * self.n_matrix).sum(2).sum(1))

        # Calculate H_interaction
        # Compute N operator for each state and lattice site
        # TODO: Keep big_n assigned??
        self.big_n = np.logical_and(self.n_matrix[:, :, 0], self.n_matrix[:, :, 1])
        # Reshape to align u matrix with Dimension 1
        # FIXME: move this up
        self.u_matrix = self.u_matrix[None, :]
        self.H_interaction = np.diag((self.u_matrix * self.big_n).sum(1))

        # Calculate H_hopping
        # Dimension 0: Basis state (position in self.basis) (size: L choose N)
        # Dimension 1: Lattice site to hop to (position in lattice) (size: L)
        # Dimension 2: Lattice site to hop from (position in lattice) (size: L)
        # Dimension 3: Spin state (0: spin down, 1: spin up) (size: 2)
        # FIXME: move this up
        self.beta = self.beta[None, :, :, None]
        hopping = np.logical_and(
            self.n_matrix[:, None, :, :], np.logical_not(self.n_matrix[:, :, None, :])
        )
        # FIXME: remove assignment after testing
        self.hopping = hopping
        up_transitions = (
            2 ** np.arange(self.lattice.n_sites)[:, None]
            - 2 ** np.arange(self.lattice.n_sites)[None, :]
        )
        down_transitions = (
            2 ** np.arange(self.lattice.n_sites, 2 * self.lattice.n_sites)[:, None]
            - 2 ** np.arange(self.lattice.n_sites, 2 * self.lattice.n_sites)[None, :]
        )
        self.transitions = np.stack((down_transitions, up_transitions), axis=-1)
        # Avoid vectorizing here due to memory concerns
        # TODO: consider attempting vectorization unless it crashes when allocating array
        for i, state in enumerate(hopping):
            state_transitions = self.basis[i] + self.transitions
            allowed_hops = np.nonzero(hopping[i])
            components = np.dstack(
                (state_transitions[allowed_hops], (state * self.beta[0, :, :, 0:1])[allowed_hops])
            )
            for new_state, amplitude in components[0]:
                self.H_hopping[i, self.int_to_state[int(new_state)]] = amplitude

        not_singly_occ = np.logical_not(np.logical_xor(self.n_matrix[:, None, :, 0], self.n_matrix[:, :, None, 1]))
        self.H_off_site_repulsion = 0


# FIXME: This needs to be reworked to avoid overflow on K > 15
class HilbertSpace:
    """Hilbert space for N electrons on K sites.

    Hilbert space for lattice sites, with each site containing 1 spin-up orbital and 1 spin-down
    orbital.

    """

    def __init__(self, n_electrons=0, n_sites=0):
        """Hilbert space: 2K orbitals filled by N electrons

        Parameters
        ----------
        n_electrons : int
            The number of electrons in the system
        n_sites : int
            The number of sites, with each containing a pair of spin-orbitals

        """
        self.n_electrons = n_electrons
        self.n_sites = n_sites
        self.states = self.construct_states()

    def construct_states(self):
        """Construct the basis states for the Hilbert space.

        Each basis state is represented as a binary integer I, and ordered such that
        I = 2**L * I(spin down) + I(spin up).

        Each integer I(spin ...) represents an occupation number (ON) vector, where
        | k_m , k_m-1 , ... , k_0 > = I = 2**(k_m) + 2**(k_m-1) + ... + 2**(0),
        so | 0_2 , 1_1 , 1_0 > = 011

        """
        states = []
        for i in combinations(
            reversed(range(2 * self.K)), self.N
        ):  # Reversing ensures correct order of final states
            states.append(sum([2 ** j for j in i]))
        return np.array(list(reversed(states)))
