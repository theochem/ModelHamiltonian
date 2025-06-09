r"""Molkit Module."""

from .utils.spinops import spinize_H, antisymmetrize_two_body, get_spin_blocks


class MolHam:
    def __init__(self, one_body, two_body):
        """
        Initialize MolHam with one and two electron integrals.

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

    def spinize(self) -> None:
        """
        Convert the stored spatial-orbital integrals to the spin-orbital
        basis and cache them in `self.one_body_spin` / `self.two_body_spin`.

        """
        self.one_body_spin, self.two_body_spin = spinize_H(
            self.one_body, self.two_body
        )

    def antisymmetrize(self):
        """Apply proper antisymmetrization to two-electron integrals"""
        self.two_body = antisymmetrize_two_body(self.two_body, inplace=True)

    def get_spin_blocks(self):
        """
        Return the main spin blocks of the two-body spin-orbital tensor.

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
        """Build the reduced hamiltonian form"""
        pass
