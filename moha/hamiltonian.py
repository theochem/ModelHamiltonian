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
    """Hubbard model Hamiltonian.

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
        self.space.construct()
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
            return state + 2**l * (2**j - 2**i)
        else:
            return state + 2**j - 2**i

    def calc(self):
        # Interaction term

        # Determine the ON vectors for each basis state
        on_down = self.basis >> self.lattice.Nsites     # The first half of I is I(spin down)
        on_up = self.basis % 2**self.lattice.Nsites     # The second half of I is I(spin up)

        # Count the number of sites with 2 electrons
        interactions = list(map(lambda x: bin(x).count("1"), on_down & on_up))  # Count the ones in the bin string repr.
        self.H_interaction = self.U*np.diag(interactions)

        # Hopping term

        # Determine which basis states have an allowed i -> n hop
        hops = []
        for i in range(self.lattice.Nsites):
            for n in self.lattice.sites[i].neighbours:   # Should be a way to avoid redundancy here...
                hop_spin_down = (((2**n ^ on_down) & (~on_down)) >> n) & ((2**i & on_down) >> i)
                hop_spin_up = (((2 ** n ^ on_up) & (~on_up)) >> n) & ((2 ** i & on_up) >> i)
                hop_up = list(map(lambda x: self.x_operator(x, i, n, self.lattice.Nsites, "up"), self.basis))
                hop_down = list(map(lambda x: self.x_operator(x, i, n, self.lattice.Nsites, "down"), self.basis))
                hop_up *= hop_spin_up
                hop_down *= hop_spin_down
                [hops.append((y, z)) for y, z in zip(hop_down, self.basis) if y != 0 and z != 0]
                [hops.append((y, z)) for y, z in zip(hop_up, self.basis) if y != 0 and z != 0]
        print(hops)

        for r, c in hops:
            self.H_hopping[self.int_to_state[r], self.int_to_state[c]] = self.t

        # Energy term

        self.H_total = self.H_interaction + self.H_hopping + self.H_energy


class HilbertSpace:
    """Hilbert space for N electrons on K sites.

    Hilbert space for lattice sites, with each site containing 1 spin up orbital and 1 spin down
    orbital.

    """

    def __init__(self, n, k):
        """Hilbert space: 2K orbitals filled by N electrons

        Parameters
        ----------
        n : int
            The number of electrons

        k : int
            The number of sites

        """
        self.N = n
        self.K = k
        self.states = None

    def construct(self):
        """Construct the basis states for the Hilbert space.

        Each basis state is represented as a binary integer I, and ordered such that
        I = 2**L * I(spin down) + I(spin up).

        Each integer I(spin ...) represents an occupation number (ON) vector, where
        | k_m , k_m-1 , ... , k_0 > = I = 2**(k_m) + 2**(k_m-1) + ... + 2**(0),
        so | 0_2 , 1_1 , 1_0 > = 011

        """
        states = []
        for i in combinations(reversed(range(2*self.K)), self.N):     # Reversing ensures correct order of final states
            states.append(sum([2**j for j in i]))
        self.states = np.array(list(reversed(states)))
