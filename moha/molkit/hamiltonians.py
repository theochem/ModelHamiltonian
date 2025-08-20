r"""Molkit Module."""

from .utils.spinops import (
    antisymmetrize_two_body,
    get_spin_blocks,
    upscale_one_body
)
from .utils.tools import load_fcidump
import numpy as np


class MolHam:
    """Hamiltonian class for molecular systems.

    Class handles handles molecular Hamiltonian
    in the form of one- and two-electron integrals.
    """

    def __init__(self, one_body=None, two_body=None, zero_body=0):
        """Initialize MolHam with one and two electron integrals.

        Parameters
        ----------
        one_body : np.ndarray
            One-electron integrals in spatial orbital basis
        two_body : np.ndarray
            Two-electron integrals in spatial orbital basis
        zero_body: float
            Core energy

        """
        self.zero_body = zero_body
        self.one_body = one_body
        self.two_body = two_body
        if self.one_body is not None:
            self.n_spatial = one_body.shape[0]
            self.n_spin = 2 * self.n_spatial
        else:
            self.n_spatial = None
            self.n_spin = None
        self.reduced_ham = None

    def antisymmetrize(self):
        """Apply proper antisymmetrization to two-electron integrals."""
        self.two_body = antisymmetrize_two_body(self.two_body, inplace=True)

    def get_spin_blocks(self):
        """Return the main spin blocks of the two-body spin-orbital tensor.

        Returns
        -------
        dict
            Dictionary with spin blocks: 'aaaa', 'bbbb', 'abab'

        """
        if not hasattr(self, "two_body_spin"):
            raise RuntimeError(
                "Call .spinize_H() first to compute spin-orbital form.")

        return get_spin_blocks(self.two_body_spin, self.n_spatial)

    def spinize_H(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Convert the one/two body terms from spatial to spin-orbital basis.

        Parameters
        ----------
        one_body : np.ndarray
            One-body term in spatial orbital basis (shape (n, n)).
        two_body : np.ndarray
            Two-body term in spatial orbital basis (shape (n, n, n, n)).

        Returns
        -------
        one_body_spin : np.ndarray
            One-body term in spin-orbital basis in physics notation
        two_body_spin : np.ndarray
            Two-body term in spin-orbital basis in physics notation

        Notes
        -----
        Rules for the conversion:
        - The one-body term is converted as follows:
            - :math:`h_{pq}^{\\alpha \\alpha}=h_{pq}^{\\beta \\beta}
            =h_{pq}^{\\text{spatial}}`
            - :math:`h_{pq}^{\\alpha \\beta}=h_{pq}^{\\beta \\alpha}=0`
        - The two-body term is converted as follows:
            - :math:`V_{pqrs}^{\\alpha \\alpha \\alpha \\alpha}=\\
                    V_{pqrs}^{\\alpha \\beta \\alpha \\beta}=\\
                    V_{pqrs}^{\\beta \\alpha \\beta \\alpha}=\\
                    V_{pqrs}^{\\text{spatial}}`

        """
        one_body, two_body = self.one_body, self.two_body
        one_body = np.asarray(one_body)
        two_body = np.asarray(two_body)

        if one_body.ndim != 2 or one_body.shape[0] != one_body.shape[1]:
            raise ValueError("one_body must be square (n, n)")
        n = one_body.shape[0]
        if two_body.shape != (n, n, n, n):
            raise ValueError(
                "two_body must have shape (n, n, n, n) with same n")

        one_body_spin = np.zeros((2 * n, 2 * n), dtype=one_body.dtype)
        one_body_spin[:n, :n] = one_body   # αα block
        one_body_spin[n:, n:] = one_body   # ββ block

        two_body_spin = np.zeros((2 * n, 2 * n, 2 * n, 2 * n),
                                 dtype=two_body.dtype)
        # αααα
        two_body_spin[:n, :n, :n, :n] = two_body
        # ββββ
        two_body_spin[n:, n:, n:, n:] = two_body
        # αβαβ
        two_body_spin[:n, n:, :n, n:] = two_body
        # βαβα
        two_body_spin[n:, :n, n:, :n] = two_body

        self.one_body_spin = one_body_spin
        self.two_body_spin = two_body_spin

        return one_body_spin, two_body_spin

    def to_reduced(self, n_elec):
        r"""Return the reduced 4-index Hamiltonian tensor.

        .. math::

        k_{pqrs}=\frac{1}{2\,(N-1)}\bigl(h_{pq}\,\delta_{rs}
        +h_{rs}\,\delta_{pq}\bigr)+\tfrac12\,V_{pqrs}.

        Parameters
        ----------
        one_body_spatial : ndarray, shape (n, n)
            One‑electron integral matrix :math:`h_{pq}` in a spatial‑orbital
            basis.
        two_body_spatial : ndarray, shape (n, n, n, n)
            Two‑electron integral tensor :math:`V_{pqrs}` (Dirac convention,
            spatial orbitals).
        n_elec : int
            Number of electrons *N* in the system.

        Returns
        -------
        ndarray, shape (2n, 2n, 2n, 2n)
            Reduced Hamiltonian tensor :math:`k_{pqrs}` in the
            spin‑orbital basis.

        Notes
        -----
        The function is stateless; it does not modify the parent
        ``MolHam`` instance.
        """
        h_spin, V_spin = self.spinize_H()

        h_upscaled = upscale_one_body(h_spin, n_elec)

        k = h_upscaled + 0.5 * V_spin
        return k

    def from_fcidump(self, path):
        """Assign Hamiltonian parameters from an FCIDUMP file.

        Parameters
        ----------
        path : str
            Path to the FCIDUMP file.

        Returns
        -------
        None
        """
        data = load_fcidump(open(path))
        self.one_body = data['one_ints']
        self.two_body = data['two_ints']
        self.zero_body = data['core_energy']
        self.n_spatial = self.one_body.shape[0]
        self.n_spin = 2 * self.n_spatial
