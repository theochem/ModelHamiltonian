r"""Utilities for molecular Hamiltonians."""

import numpy as np
from warnings import warn


def to_geminal(two_body=None, type='h2'):
        r"""
        Convert the two-body term to the geminal basis.

        Parameters
        ----------
        two_body : np.ndarray
            Two-body term in spin-orbital basis in physics notation.
        type : str
            ['rdm2', 'h2']. Type of the two-body term.
            - 'rdm2' : 2 body reduced density matrix
            - 'h2' : 2 body Hamiltonian

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

        """
        n = two_body.shape[0]
        two_body_gem = []

        # i,j,k,l -> pqrs
        for p in range(n):
            for q in range(p + 1, n):
                for r in range(n):
                    for s in range(r + 1, n):
                        if type == 'rdm2':
                            two_body_gem.append(
                                0.5 * 4 * two_body[p, q, r, s]
                            )
                        elif type == 'h2':
                            two_body_gem.append(
                                0.5 * (
                                    two_body[p, q, r, s]
                                    - two_body[q, p, r, s]
                                    - two_body[p, q, s, r]
                                    + two_body[q, p, s, r]
                                )
                            )

        n_gem = n * (n - 1) // 2
        return np.array(two_body_gem).reshape(n_gem, n_gem)


def from_geminal(two_body_gem, n_orb):
    """
    Inverse of MolHam.to_geminal().

    Parameters
    ----------
    two_body_gem : (n_gem, n_gem) ndarray
        Matrix in the geminal basis.
    n_orb : int
        Number of spin orbitals.

    Returns
    -------
    V : (n_orb, n_orb, n_orb, n_orb) ndarray
        Fully antisymmetrised two-electron tensor V_{ijkl}.
    """
    n_gem = n_orb * (n_orb - 1) // 2
    if two_body_gem.shape != (n_gem, n_gem):
        raise ValueError(f"Shape mismatch: got {two_body_gem.shape}")

    # Generate flattened pair list exactly like to_geminal
    pairs = [(i, j) for i in range(n_orb) for j in range(i + 1, n_orb)]
    V = np.zeros((n_orb, n_orb, n_orb, n_orb))

    for A, (i, j) in enumerate(pairs):
        for B, (k, l) in enumerate(pairs):
            val = 0.25 * two_body_gem[A, B]  # undo the factor 0.5 * 4 = 2

            # Apply antisymmetric filling
            V[i, j, k, l] = val
            V[j, i, k, l] = -val
            V[i, j, l, k] = -val
            V[j, i, l, k] = val
            V[k, l, i, j] = val
            V[l, k, i, j] = -val
            V[k, l, j, i] = -val
            V[l, k, j, i] = val

    return V

def set_four_index_element(
    four_index_object, i0, i1, i2, i3, value
):
    """
    This function is adapted from IOData library.
    https://iodata.readthedocs.io/en/latest/index.html

    Assign values to a four index object, account for 8-fold index symmetry.

    This function assumes physicists' notation.

    Parameters
    ----------
    four_index_object
        The four-index object. It will be written to.
        shape=(nbasis, nbasis, nbasis, nbasis), dtype=float
    i0, i1, i2, i3
        The indices to assign to.
    value
        The value of the matrix element to store.

    """
    four_index_object[i0, i1, i2, i3] = value
    four_index_object[i1, i0, i3, i2] = value
    four_index_object[i2, i1, i0, i3] = value
    four_index_object[i0, i3, i2, i1] = value
    four_index_object[i2, i3, i0, i1] = value
    four_index_object[i3, i2, i1, i0] = value
    four_index_object[i1, i2, i3, i0] = value
    four_index_object[i3, i0, i1, i2] = value


def load_fcidump(lit) -> dict:
    """ This function is adapted from IOData module.
    https://iodata.readthedocs.io/en/latest/index.html

    Molpro 2012 FCIDUMP file format.


    Parameters
    ----------
    lit: LineIterator

    Notes
    -----
    1. This function works only for restricted wave-functions.
    2. One- and two-electron integrals are stored in chemists' notation in an FCIDUMP file,
    while IOData and MoHa internally uses physicists' notation.
    3. Keep in mind that the FCIDUMP format changed in MOLPRO 2012, so files generated with
    older versions are not supported.
    """
    # check header
    line = next(lit)
    if not line.startswith(" &FCI NORB="):
        raise (f"Incorrect file header: {line.strip()}", lit)

    # read info from header
    words = line[5:].split(",")
    header_info = {}
    for word in words:
        if word.count("=") == 1:
            key, value = word.split("=")
            header_info[key.strip()] = value.strip()
    nbasis = int(header_info["NORB"])
    nelec = int(header_info["NELEC"])
    spinpol = int(header_info["MS2"])

    # skip rest of header
    for line in lit:
        words = line.split()
        if words[0] == "&END" or words[0] == "/END" or words[0] == "/":
            break

    # read the integrals
    one_mo = np.zeros((nbasis, nbasis))
    two_mo = np.zeros((nbasis, nbasis, nbasis, nbasis))
    core_energy = 0.0

    for line in lit:
        words = line.split()
        if len(words) != 5:
            raise (
                f"Expecting 5 fields on each data line in FCIDUMP, got {len(words)}.", lit
            )
        value = float(words[0])
        if words[3] != "0":
            ii = int(words[1]) - 1
            ij = int(words[2]) - 1
            ik = int(words[3]) - 1
            il = int(words[4]) - 1
            if two_mo[ii, ik, ij, il] != 0.0:
                warn(
                    "Duplicate entries in the FCIDUMP file are ignored",
                    stacklevel=2,
                )
            set_four_index_element(two_mo, ii, ik, ij, il, value)
        elif words[1] != "0":
            ii = int(words[1]) - 1
            ij = int(words[2]) - 1
            one_mo[ii, ij] = value
            one_mo[ij, ii] = value
        else:
            core_energy = value

    return {
        "nelec": nelec,
        "spinpol": spinpol,
        "one_ints": one_mo,
        "two_ints": two_mo,
        "core_energy": core_energy,
    }

