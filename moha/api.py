r"""Model Hamiltonian API."""

from abc import ABC, abstractmethod

from typing import TextIO

import numpy as np

from scipy.sparse import csr_matrix, lil_matrix

from .utils import convert_indices, expand_sym

__all__ = [
    "HamiltonianAPI",
]


class HamiltonianAPI(ABC):
    r"""Hamiltonian abstract base class."""

    @abstractmethod
    def generate_zero_body_integral(self):
        r"""Generate zero body integral."""
        pass

    @abstractmethod
    def generate_one_body_integral(self, sym: int, basis: str, dense: bool):
        r"""
        Generate one body integral in spatial or spin orbital basis.

        Parameters
        ----------
        sym: int
            symmetry -- [1, 2] default is 1
        basis: str
            ['spatial', 'spin orbital']
        dense: bool
            dense or sparse matrix; default False

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
        """
        pass

    @abstractmethod
    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        r"""
        Generate two body integral in spatial or spinorbital basis.

        Parameters
        ----------
        sym: int
            symmetry -- [2, 4, 8] default is 1
        basis: str
            ['spatial', 'spin orbital']
        dense: bool
            dense or sparse matrix; default False

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
        """
        pass

    def to_sparse(self, Md):
        r"""
        Convert dense array of integrals to sparse array in scipy csr format.

        Parameters
        ----------
        Md: np.ndarray
            input matrix of the shape 2d or 4d

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        # Finding indices for non-zero elements and shape of Md.
        indices = np.array(np.where(Md != 0)).astype(int).T
        N = Md.shape[0]

        # Converting 2D array to csr_matrix
        if np.ndim(Md) == 2:
            return csr_matrix(Md)

        # Converting 4D array to csr_matrix using convert_indices from util.py.
        elif np.ndim(Md) == 4:
            row = np.array([])
            col = np.array([])
            data = np.array([])
            for ind in indices:
                p, q = convert_indices(N,
                                       int(ind[0]),
                                       int(ind[1]),
                                       int(ind[2]),
                                       int(ind[3]))
                row = np.append(row, p)
                col = np.append(col, q)
                data = np.append(data, Md[tuple(ind)])
            return csr_matrix((data, (row, col)), shape=(N * N, N * N))

        # Return if array dimensions incompatible.
        else:
            print("Incompatible dense array dimension.",
                  " Must be either 2 or 4 dimensions.")
            return

    def to_dense(self, Ms, dim=2):
        r"""
        Convert to dense matrix.

        Convert sparse array of integrals
        in scipy csr format to dense numpy array.

        Parameters
        ----------
        Ms: scipy.sparse.csr_matrix
        dim: int
            target dimension of output array (either 2 or 4)

        Returns
        -------
        np.ndarray
        """
        # return dense 2D array (default).
        if dim == 2:
            return Ms.todense() if isinstance(Ms, csr_matrix) else Ms

        # Optionally, return dense 4D array for two-particle integrals.
        elif dim == 4:
            N = int(np.sqrt(Ms.shape[0]))
            vd = np.zeros([N, N, N, N])
            for p, q in np.array(Ms.nonzero()).T:
                i, j, k, l_ = convert_indices(N, int(p), int(q))
                vd[(i, j, k, l_)] = Ms[p, q]
            return vd

        # Return if target dim is not 2 or 4.
        else:
            print("Target output dimension must be either 2 or 4.")
            return

    def to_spatial(self, sym: int, dense: bool, nbody: int):
        r"""
        Convert one-/two- integral matrix from spin-orbital to spatial basis.

        Parameters
        ----------
        sym: int
            symmetry -- [2, 4, 8] default is 1
        dense: bool
            dense or sparse matrix; default False
        nbody: int
            type of integral, one of 1 (one-body) or 2 (two-body)

        Returns
        -------
        spatial_int: scipy.sparce.csr_matrix or np.ndarray
            one-/two-body integrals in spatial basis
        """
        # Assumption: spatial components of alpha and beta
        # spin-orbitals are equivalent
        integral = self.one_body if nbody == 1 else self.two_body
        n = 2 * self.n_sites
        if integral.shape[0] == 2 * self.n_sites:
            spatial_int = lil_matrix((self.n_sites, self.n_sites))
            spatial_int = integral[: self.n_sites, : self.n_sites]
        elif integral.shape[0] == 4 * self.n_sites ** 2:
            spatial_int = lil_matrix((self.n_sites ** 2, self.n_sites ** 2))
            for p in range(self.n_sites):
                # v_pppp = U_pppp_ab
                pp, pp = convert_indices(self.n_sites, p, p, p, p)
                pp_, pp_ = convert_indices(n,
                                           p, p + self.n_sites,
                                           p, p + self.n_sites)
                spatial_int[pp, pp] = integral[(pp_, pp_)]
                for q in range(p, self.n_sites):
                    # v_pqpq = 0.5 * (Gamma_pqpq_aa + Gamma_pqpq_bb)
                    pq, pq = convert_indices(self.n_sites, p, q, p, q)
                    pq_, pq_ = convert_indices(n, p, q, p, q)
                    spatial_int[pq, pq] = integral[pq_, pq_]
                    # v_pqpq += 0.5 * (Gamma_pqpq_ab + Gamma_pqpq_ba)
                    pq_, pq_ = convert_indices(n,
                                               p, q + self.n_sites,
                                               p, q + self.n_sites)
                    spatial_int[pq, pq] += integral[pq_, pq_]
                    #  v_ppqq = Pairing_ppqq_ab
                    pp, qq = convert_indices(self.n_sites, p, p, q, q)
                    pp_, qq_ = convert_indices(n,
                                               p, p + self.n_sites,
                                               q, q + self.n_sites)
                    spatial_int[pp, qq] = integral[pp_, qq_]
        else:
            raise ValueError("Wrong integral input.")
        spatial_int = expand_sym(sym, spatial_int, nbody)
        spatial_int = spatial_int.tocsr()

        if dense:
            if isinstance(
                    spatial_int, csr_matrix
            ):  # FixMe make sure that this works for every system
                spatial_int = spatial_int.toarray()
                spatial_int = np.reshape(
                    spatial_int, (self.n_sites,
                                  self.n_sites,
                                  self.n_sites,
                                  self.n_sites)
                )
            else:
                spatial_int = self.to_dense(spatial_int,
                                            dim=4 if nbody == 2 else 1)
        return spatial_int

    def to_spinorbital(self, integral: np.ndarray, sym=1, dense=False):
        r"""
        Convert one-/two- integral matrix from spatial to spin-orbital basis.

        Parameters
        ----------
        integral: scipy.sparse.csr_matrix
            type of integral, one of 1 (one-body) or 2 (two-body)
        sym: int
            symmetry -- [2, 4, 8] default is 1
        dense: bool
            dense or sparse matrix; default False

        Returns
        -------
        None
        """
        pass

    def save_fcidump(self, f: TextIO, nelec=0, spinpol=0):
        r"""
        Save all parts of hamiltonian in fcidump format.

        Parameters
        ----------
        f: TextIO file
        nelec: int
            The number of electrons in the system
        spinpol: float
            The spin polarization. By default, its value is derived from the
            molecular orbitals (mo attribute), as abs(nalpha - nbeta).
            In this case, spinpol cannot be set.
            When no molecular orbitals are present,
            this attribute can be set.

        Returns
        -------
        None
        """
        # Reduce symmetry of integral
        one_ints = expand_sym(self._sym, self.one_body, 1)

        # Write header
        nactive = one_ints.shape[0]
        print(f" &FCI NORB={nactive:d},"
              f"NELEC={nelec:d},MS2={spinpol:d},", file=f)
        print(f"  ORBSYM= {','.join('1' for v in range(nactive))},", file=f)
        print("  ISYM=1", file=f)
        print(" &END", file=f)

        # Reduce symmetry of integrals
        two_ints = expand_sym(self._sym, self.two_body, 2)

        # getting nonzero elements from the 2d _sparse_ array
        p_array, q_array = two_ints.nonzero()

        # converting 2d indices to 4d indices
        N = int(np.sqrt(two_ints.shape[0]))
        for p, q in zip(p_array, q_array):
            i, j, k, l_ = convert_indices(N, int(p), int(q))
            # changing indexing from physical to chemical notation
            j, k = k, j
            if j > i and l_ > k and\
                    (i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l_:
                value = two_ints[(i, k, j, l_)]
                print(f"{value:23.16e} "
                      f"{i + 1:4d} {j + 1:4d} {k + 1:4d} "
                      f"{l_ + 1:4d}", file=f)
        for i in range(nactive):
            for j in range(i + 1):
                value = one_ints[i, j]
                if value != 0.0:
                    print(f"{value:23.16e} {i + 1:4d}"
                          f" {j + 1:4d} {0:4d} {0:4d}", file=f)

        core_energy = self.zero_energy
        if core_energy is not None:
            print(f"{core_energy:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}", file=f)

    def save_triqs(self, fname: str, integral):
        r"""
        Save matrix in triqc format.

        Parameters
        ----------
        fname: str
            name of the file
        integral: int
            type of integral, one of 1 (one-body) or 2 (two-body)

        Returns
        -------
        None
        """
        pass

    def save(self, fname: str, integral, basis):
        r"""Save file as regular numpy array."""
        pass
