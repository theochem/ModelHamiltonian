r"""Molkit Module."""

from .utils.spinops import antisymmetrize_two_body, get_spin_blocks
import numpy as np


class MolHam:
    """Hamiltonian class for molecular systems.

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

    def spinize_H(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Convert the one/two body terms from spatial to spin-orbital basis.

        Parameters
        ----------
        None

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
        one_body = self.one_body
        two_body = self.two_body
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

    def to_geminal(two_body):
        r"""Convert the two-body term to the geminal basis.

        Parameters
        ----------
        two_body : np.ndarray
            Two-body term in spin-orbital basis in physics notation.

        Returns
        -------
        two_body_gem : np.ndarray
            Two-body term in the geminal basis

        Notes
        -----
        Assuming that rdm2 obbey the following permutation rules:
        - :math:`\Gamma_{p q r s}=-\Gamma_{q p r s}=-\Gamma_{p q s r}
        =\Gamma_{q p s r}`
        we can convert the two-body term to the geminal basis
        by the following formula:

        .. math::

            \Gamma_{p q}=0.5 * 4 \Gamma_{p q r s}

        where:
        - :math:`\Gamma_{p q}` is the two-body term in the geminal basis
        - :math:`\Gamma_{p q r s}` is the two-body term in the spin-orbital
        Hamiltonian in the geminal basis is obtained by the following formula:

        .. math::

        V_{A B}
        =\frac{1}{2}\left(V_{p q r s}-V_{q p r s}-V_{p q r s}+V_{qprs}\right)

        This is coming from the fact, that the Hamiltonian object
        retured from the fcidump (converted to the physics notation)
        doesn't obbey the permutation rules.
        Thus, the full formula has to be used.

        """
        n = two_body.shape[0]
        two_body_gem = []

        # i,j,k,l -> pqrs
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    for s in range(r + 1, n):
                        term = 0.5 * (
                            two_body[p, q, r, s]
                            - two_body[q, p, r, s]
                            - two_body[p, q, s, r]
                            + two_body[q, p, s, r]
                        )
                        two_body_gem.append(term)

        n_gem = n * (n - 1) // 2
        return np.array(two_body_gem).reshape(n_gem, n_gem)

    def build_reduced(self):
        """Build the reduced hamiltonian form."""
        pass
