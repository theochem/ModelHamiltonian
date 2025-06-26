r"""Molkit Module."""

from .utils.spinops import antisymmetrize_two_body, get_spin_blocks
import numpy as np


class MolHam:
    """
    Hamiltonian class for molecular systems.

    Class handles handles molecular Hamiltonian
    in the form of one- and two-electron integrals.
    """

    def __init__(self, one_body, two_body):
        """Initialize MolHam with one and two electron integrals.

        Parameters
        ----------
        one_body : np.ndarray
            One-electron integrals in spatial orbital basis
        two_body : np.ndarray
            Two-electron integrals in spatial orbital basis

        """
        self.one_body = one_body
        self.two_body = two_body
        self.n_spatial = one_body.shape[0]
        self.n_spin = 2 * self.n_spatial
        self.reduced_ham = None

    def spinize_H(one_body: np.ndarray,
                  two_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Convert the one/two body terms from spatial to spin-orbital basis.

        Parameters
        ----------
        one_body : np.ndarray
            One-body term in spatial basis in physics notation
        two_body : np.ndarray
            Two-body term in spatial basis in physics notation

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

        return one_body_spin, two_body_spin

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
                "Call .spinize() first to compute spin-orbital form.")

        return get_spin_blocks(self.two_body_spin, self.n_spatial)

    def build_reduced(self):
        """Build the reduced hamiltonian form."""
        pass
