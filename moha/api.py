r"""Model Hamiltonian API."""

from abc import ABC, abstractmethod

from typing import TextIO

import numpy as np

from scipy.sparse import csr_matrix, lil_matrix, diags

from .utils import convert_indices, get_atom_type

__all__ = [
    "HamiltonianAPI",
]


class HamiltonianAPI(ABC):
    r"""Hamiltonian abstract base class."""

    def generate_connectivity_matrix(self):
        r"""
        Generate connectivity matrix.

        Returns
        -------
        tuple
            (dictionary, np.ndarray)
        """
        max_site = 0
        atoms_sites_lst = []

        # check if self.connectivity is a matrix
        # if so, put assign it to self.connectivity_matrix
        # and set the atom_types to None
        if isinstance(self.connectivity, np.ndarray):
            self.connectivity_matrix = csr_matrix(self.connectivity)
            self.atom_types = None
            self.n_sites = self.connectivity_matrix.shape[0]

            return None, self.connectivity_matrix

        for atom1, atom2, bond in self.connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)
            for pair in [(atom1_name, site1), (atom2_name, site2)]:
                if pair not in atoms_sites_lst:
                    atoms_sites_lst.append(pair)
            if max_site < max(site1, site2):  # finding the max index of site
                max_site = max(site1, site2)
        self.n_sites = len(atoms_sites_lst)

        if self.atom_types is None:
            atom_types = [None for i in range(max_site + 1)]
            for atom, site in atoms_sites_lst:
                atom_types[site] = atom
            self.atom_types = atom_types
        connectivity_mtrx = np.zeros((max_site, max_site))

        for atom1, atom2, bond in self.connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)
            connectivity_mtrx[site1 - 1, site2 - 1] = bond
            # numbering of sites starts from 1

        connectivity_mtrx = np.maximum(connectivity_mtrx, connectivity_mtrx.T)
        self.connectivity_matrix = csr_matrix(connectivity_mtrx)
        return atoms_sites_lst, self.connectivity_matrix

    @abstractmethod
    def generate_zero_body_integral(self):
        r"""Generate zero body integral."""
        pass

    @abstractmethod
    def generate_one_body_integral(self, dense: bool, basis: str):
        r"""
        Generate one body integral in spatial or spin orbital basis.

        Parameters
        ----------
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
            raise ValueError("Target output dimension must be either 2 or 4.")

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


        Notes
        -----
        Given the one- or two-body Hamiltonian matrix terms,
        :math:`h_{i,j}` and :math:`g_{ij,kl}` respectively,
        we populate the spatial integrals by calcualting __average__ over the
        spin-orbitals

        Specifically, for the one-body integrals,
        we have:
        :math:`h_{pq} = 0.25*(h_{pq}^{aa} + h_{pq}^{bb} + h_{pq}^{ab}
        + h_{pq}^{ba}) = h_{pq}^{aa} = h_{pq}^{bb}`
        Therefore, the one-body integrals in the spatial basis
        are the same as the aa part of
        one-body integrals in the spin-orbital basis.

        For the two-body integrals, we have:
        :math:`v_{pqrs} = 0.25*(v_{pqrs}^{aaaa} + v_{pqrs}^{bbbb} +
        v_{pqrs}^{abab} + v_{pqrs}^{baba})`
        Assuming that :math:`v_{pqrs}^{abab} = v_{pqrs}^{baba}` and
        :math:`v_{pqrs}^{aaaa} = v_{pqrs}^{bbbb}`
        :math:`v_{pqrs} = 0.5*(v_{pqrs}^{aaaa} + v_{pqrs}^{abab})`
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
                for q in range(p+1, self.n_sites):
                    # v_pqpq = 0.5*Gamma_pqpq_aa = 0.5*Gamma_pqpq_bb
                    pq, pq = convert_indices(self.n_sites, p, q, p, q)
                    pq_, pq_ = convert_indices(n, p, q, p, q)
                    spatial_int[pq, pq] = 0.5 * integral[pq_, pq_]
                    # v_pqpq += 0.5*Gamma_pqpq_ab
                    # assuming that Gamma_pqpq_ab = Gamma_pqpq_ba
                    pq_, pq_ = convert_indices(n,
                                               p, q + self.n_sites,
                                               p, q + self.n_sites)
                    spatial_int[pq, pq] += 0.5 * integral[pq_, pq_]
                    #  v_ppqq = Pairing_ppqq_ab
                    pp, qq = convert_indices(self.n_sites, p, p, q, q)
                    pp_, qq_ = convert_indices(n,
                                               p, p + self.n_sites,
                                               q, q + self.n_sites)
                    spatial_int[pp, qq] = 0.5 * integral[pp_, qq_]
        else:
            raise ValueError("Wrong integral input.")
        spatial_int = expand_sym(sym, spatial_int, nbody)
        spatial_int = spatial_int.tocsr()

        if dense:
            spatial_int = self.to_dense(spatial_int,
                                        dim=4 if nbody == 2 else 2)
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
            if (i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l_:
                value = two_ints[p, q]
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

    def savez(self, fname: str):
        r"""Save file as regular npz file.

        Parameters
        ----------
        fname: str
            name of the file

        Returns
        -------
        None
        """
        if self.zero_energy is not None:
            e0 = self.zero_energy
        else:
            raise ValueError("Zero energy was not calculated.")

        if self.one_body is not None:
            h = self.to_dense(self.one_body, dim=2)
        else:
            raise ValueError("One body integrals were not calculated.")

        if self.two_body is not None:
            v = self.to_dense(self.two_body, dim=4)
        else:
            raise ValueError("Two body integrals were not calculated.")

        np.savez(fname, e0=e0, h1=h, h2=v)


def expand_sym(sym, integral, nbody):
    r"""
    Restore permutational symmetry of one- and two-body terms.

    Parameters
    ----------
    sym: int
        integral symmetry, one of 1 (no symmetry), 2, 4 or 8.
    integral: scipy.sparce.csr_matrix
        2-D sparse array, the {one,two}-body integrals
    nbody: int
        number of particle variables in the integral,
        one of 1 (one-body) or 2 (two-body)

    Returns
    -------
    integral: scipy.sparse.csr_matrix
        2d array of with the symmetry 1

    Notes
    -----
        Given the one- or two-body Hamiltonian matrix terms,
        :math:`h_{i,j}` and :math:`g_{ij,kl}` respectively,
        the supported permutational symmetries are:
        sym = 2:
        :math:`h_{i,j} = h_{j,i}`
        :math:`g_{ij,kl} = g_{kl,ij}`
        sym = 4:
        :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji}`
        sym = 8:
        :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji} =
         g_{kj,il} = g_(il,kj) = g_(li,jk) = g_(jk,li)`
        sym = 1 corresponds to no-symmetry
        where it is assumed the integrals are over real orbitals.

        The input Hamiltonian terms are expected to be sparse
        arrays of dimensions :math:`(N,N)` or
        :math:`(N^2, N^2)` for the one- and two-body integrals respectively.
        :math:`N` represents the number of basis functions,
        which may be either of spatial or spin-orbital type.
        This function applies to the input array the
        permutations indicated by the symmetry parameter `sym`
        to adds the missing terms.
        Phicisist notation is used for the two-body integrals:
         :math:`<pq|rs>` and further details of the
        permutations considered can be
        found in [this site]
        (http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.html).
    """
    if sym not in [1, 2, 4, 8]:
        raise ValueError("Wrong input symmetry")
    if nbody not in [1, 2]:
        raise ValueError(f"`nbody` must be an integer, "
                         f"either 1 or 2, but {nbody} given")
    if sym == 1:
        return integral

    # Expanding Symmetries
    if nbody == 1:
        if not sym == 2:
            raise ValueError("Wrong 1-body term symmetry")
        h_ii = diags(integral.diagonal()).copy()
        integral = integral + integral.T - h_ii
    else:
        # getting nonzero elements from the 2d _sparse_ array
        pq_array, rs_array = integral.nonzero()
        # getting the size of the corresponding 4d array
        n = int(np.sqrt(integral.shape[0]))

        for pq, rs in zip(pq_array, rs_array):
            p, q, r, s = convert_indices(n, pq, rs)
            if sym >= 2:
                # 1. Transpose: <pq|rs>=<rs|pq>
                rs, pq = convert_indices(n, r, s, p, q)
                integral[rs, pq] = integral[pq, rs]
            if sym >= 4:
                # 2. Permute dummy indices
                # (swap variables of particles 1 and 2):
                # <p_1 q_2|r_1 s_2> = <q_1 p_2|s_1 r_2>
                qp, sr = convert_indices(n, q, p, s, r)
                integral[qp, sr] = integral[pq, rs]
                integral[sr, qp] = integral[rs, pq]
            if sym == 8:
                # 3. Permute orbitals of the same variable,
                # e.g. <p_1 q_2|r_1 s_2> = <r_1 q_2|p_1 s_2>
                rq, ps = convert_indices(n, r, q, p, s)
                sp, qr = convert_indices(n, s, p, q, r)
                integral[rq, ps] = integral[pq, rs]
                integral[ps, rq] = integral[rs, pq]
                integral[sp, qr] = integral[qp, sr]
                integral[qr, sp] = integral[sr, qp]
    return integral
