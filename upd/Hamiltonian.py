from abc import ABC, abstractmethod
from typing import TextIO
from utils import convert_indices
import numpy as np
from scipy.sparse import csr_matrix, diags
from utils import convert_indices


class HamiltonianAPI(ABC):
    @abstractmethod
    def generate_zero_body_integral(self):
        """Generates zero body integral"""
        pass

    @abstractmethod
    def generate_one_body_integral(self, sym: int, basis: str, dense: bool):
        """
        Generates one body integral in spatial or spin orbital basis
        :param sym: symmetry -- [2, 4, 8] default is None
        :param basis: basis -- ['spatial', 'spin orbital']
        :param dense: dense or sparse matrix; default dense
        :return: numpy.ndarray or scipy.sparse.csc_matrix
        """
        pass

    @abstractmethod
    def generate_two_body_integral(self, sym: int, basis: str, dense: bool):
        """
        Generates two body integral in spatial or spinorbital basis
        :param sym: symmetry -- [2, 4, 8] default is None
        :param basis: basis -- ['spatial', 'spin orbital']
        :param dense: dense or sparse matrix; default dense
        :return: numpy.ndarray or sparse
        """
        pass

    def to_sparse(self, Md):
        """
        Converts dense array of integrals to sparse array in scipy csr format.
        :param Md: 2 or 4 dimensional numpy.array
        :return scipy.sparse.csr_matrix
        """
        # Finding indices for non-zero elements and shape of Md.
        indices = np.array(np.where(Md != 0)).astype(int).T
        N = Md.shape[0]
        
        # Converting 2D array to csr_matrix
        if np.ndim(Md) == 2:
            return csr_matrix(Md)
        
        # Converting 4D array to csr_matrix using convert_indices from util.py.
        elif np.ndim(Md) == 4:
            row = np.array([]); col = np.array([]); data = np.array([]);
            for ind in indices:
                p,q = convert_indices(N, int(ind[0]), int(ind[1]), int(ind[2]), int(ind[3]));
                row = np.append(row,p)
                col = np.append(col,q)
                data = np.append(data,Md[tuple(ind)])
            return csr_matrix((data, (row, col)), shape=(N*N, N*N))
        
        # Return if array dimensions incompatible.
        else:
            print("Incompatible dense array dimension. Must be either 2 or 4 dimensions.")
            return

    def to_dense(self, Ms, dim=2):
        """
        Converts sparse arry of integrals in scipy csr format to dense numpy array.
        :param Ms: scipy.sparse.csr_matrix
        :param dim: target dimension of output array (either 2 or 4)
        :return: numpy.array
        """
        # return dense 2D array (default).
        if dim == 2:
            return Ms.todense()

        # Optionally, return dense 4D array for two-particle integrals.
        elif dim == 4:
            N = int(np.sqrt(Ms.shape[0]))
            vd = np.zeros([N,N,N,N])
            for p,q in np.array(Ms.nonzero()).T:
                i,j,k,l = convert_indices(N, int(p),int(q))
                vd[(i,j,k,l)] = Ms[p,q]
            return vd

        # Return if target dim is not 2 or 4.
        else:
            print("Target output dimension must be either 2 or 4.")
            return

          
    def to_spatial(self, integral: np.ndarray, sym: int, dense: bool, nbody: int):
        """
        Converts one-/two- integral matrix from spin-orbital to spatial basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default is None
        :param dense: dense or sparse matrix; default sparse
        :param nbody: int, type of integral, one of 1 (one-body) or 2 (two-body)
        :return: one-/two-body integrals in spatial basis
        """
        #
        # Assumption: spatial components of alpha and beta spin-orbitals are equivalent
        #
        n = 2 * self.n_sites
        if integral.shape[0] == 2*self.n_sites:
            spatial_int = csr_matrix((self.n_sites, self.n_sites))
            spatial_int = integral[:self.n_sites, :self.n_sites]
        elif integral.shape[0] == 4*self.n_sites**2:
            spatial_int = csr_matrix((self.n_sites**2, self.n_sites**2))
            for p in range(self.n_sites):
                # v_pppp = U_pppp_ab
                pp, pp = convert_indices(self.n_sites, p,p,p,p)
                pp_, pp_ = convert_indices(n, p, p + self.n_sites, p, p + self.n_sites)
                spatial_int[pp, pp] = integral[(pp_, pp_)]
                for q in range(p, self.n_sites):
                    # v_pqpq = 0.5 * (Gamma_pqpq_aa + Gamma_pqpq_bb)
                    pq, pq = convert_indices(self.n_sites, p, q, p, q)
                    pq_, pq_ = convert_indices(n, p, q, p, q)
                    spatial_int[pq, pq] = integral[pq_, pq_]
                    # v_pqpq += 0.5 * (Gamma_pqpq_ab + Gamma_pqpq_ba)
                    pq_, pq_ = convert_indices(n, p, q + self.n_sites, p, q + self.n_sites)
                    spatial_int[pq, pq] += integral[pq_, pq_]
                    #  v_ppqq = Pairing_ppqq_ab
                    pp, qq = convert_indices(self.n_sites, p,p,q,q)
                    pp_, qq_ = convert_indices(n, p, p + self.n_sites, q, q + self.n_sites)
                    spatial_int[pp, qq] = integral[pp_, qq_]
        else:
            raise ValueError('Wrong integral input.')

        spatial_int = expand_sym(sym, spatial_int, nbody)
        
        if dense:
            if isinstance(spatial_int, csr_matrix):
                spatial_int.toarray()
            else:
                spatial_int = self.to_dense(spatial_int)
        return spatial_int

    def to_spinorbital(self, integral: np.ndarray, sym=1, dense=False):
        """
        Converts one-/two- integral matrix from spatial to spin-orbital basis
        :param integral: input matrix
        :param sym: symmetry -- [2, 4, 8] default 1
        :param dense: dense or sparse matrix; default is sparse
        :return:
        """
        pass

    def save_fcidump(self, f: TextIO, nelec=0, spinpol=0):
        """
        Save all parts of hamiltonian in fcidump format
        Adapted from https://github.com/theochem/iodata/blob/master/iodata/formats/fcidump.py
        :param f: TextIO file
        :param nelec: The number of electrons in the system
        :param spinpol: The spin polarization. By default, its value is derived from the
                        molecular orbitals (mo attribute), as abs(nalpha - nbeta). In this case,
                        spinpol cannot be set. When no molecular orbitals are present, this
                        attribute can be set.
        :return: None
        """
        # Reduce symmetry of integral
        one_ints = self.reduse_sym(self.one_ints)

        # Write header
        nactive = one_ints.shape[0]
        print(f' &FCI NORB={nactive:d},NELEC={nelec:d},MS2={spinpol:d},', file=f)
        print(f"  ORBSYM= {','.join('1' for v in range(nactive))},", file=f)
        print('  ISYM=1', file=f)
        print(' &END', file=f)

        # Reduce symmetry of integrals
        two_ints = self.reduce_sym(self.two_ints)

        # getting nonzero elements from the 2d _sparse_ array
        p_array, q_array = two_ints.nonzero()

        # converting 2d indices to 4d indices
        N = int(np.sqrt(two_ints.shape[0]))
        for p, q in zip(p_array, q_array):
            i, j, k, l = convert_indices(N, p, q)
            j, k = k, j  # changing indexing from physical to chemical notation
            if j > i and l > k and (i * (i + 1)) / 2 + j >= (k * (k + 1)) / 2 + l:
                value = two_ints[(i, k, j, l)]
                print(f'{value:23.16e} {i + 1:4d} {j + 1:4d} {k + 1:4d} {l + 1:4d}', file=f)
        for i in range(nactive):
            for j in range(i + 1):
                value = one_ints[i, j]
                if value != 0.0:
                    print(f'{value:23.16e} {i + 1:4d} {j + 1:4d} {0:4d} {0:4d}', file=f)

        core_energy = self.core_energy
        if core_energy is not None:
            print(f'{core_energy:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}', file=f)


    def save_triqs(self, fname:str, integral):
        """
        Save matrix in triqc format
        :param fname: filename
        :param integral: matrix to be saved
        :return: None
        """
        pass

    def save(self, fname: str, integral, basis):
        """Save file as regular numpy array"""
        pass


def expand_sym(sym, integral, nbody):
    """
    Restore permutational symmetry of one- and two-body terms
    :param integral: 2-D sparse array, the {one,two}-body integrals
    :param sym: int, integral symmetry, one of 1 (no symmetry), 2, 4 or 8.
    :param nbody: int, number of particle variables in the integral, one of 1 (one-body) or 2 (two-body)
    :return: integral's matrix with a symmetry of 1

    Notes
    -----
    Given the one- or two-body Hamiltonian matrix terms, :math:`h_{i,j}` and :math:`g_{ij,kl}` respectively,
    the supported permutational symmetries are:
    sym = 2:  
    :math:`h_{i,j} = h_{j,i}`  
    :math:`g_{ij,kl} = g_{kl,ij}`  
    sym = 4:  
    :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji}`  
    sym = 8:  
    :math:`g_{ij,kl} = g_{kl,ij} = g_{ji,lk} = g_{lk,ji} = g_{kj,il} = g_(il,kj) = g_(li,jk) = g_(jk,li)`  
    sym = 1 corresponds to no-symmetry
    where it is assumed the integrals are over real orbitals.

    The input Hamiltonian terms are expected to be sparse arrays of dimensions :math:`(N,N)` or 
    :math:`(N^2, N^2)` for the one- and two-body integrals respectively. :math:`N` represents 
    the number of basis functions, which may be either of spatial or spin-orbital type. 
    This function applies to the input array the permutations indicated by the symmetry parameter `sym` 
    to adds the missing terms. 
    Phicisist notation is used for the two-body integrals: :math:`<pq|rs>` and further details of the 
    permutations considered can be found in [this site](http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.html). 
    """    
    if not sym in [1, 2, 4, 8]:
        raise ValueError('Wrong input symmetry')
    if not nbody in [1, 2]:
        raise ValueError(f'`nbody` must be an integer, either 1 or 2, but {nbody} given')
    if sym == 1:
        return integral
    #
    # Expanding Symmetries
    #
    if nbody == 1:
        if not sym == 2:
            raise ValueError('Wrong 1-body term symmetry')
        h_ii = diags(integral.diagonal()).copy()
        integral = integral + integral.T - h_ii
    else:
        # getting nonzero elements from the 2d _sparse_ array
        pq_array, rs_array = integral.nonzero()

        for pq, rs in zip(pq_array, rs_array):
            p, q, r, s = convert_indices(n, pq, rs)
            if sym >= 2: 
                # 1. Transpose: <pq|rs>=<rs|pq>
                rs, pq = convert_indices(n, r,s,p,q)
                integral[rs, pq] = integral[pq, rs]
            if sym >= 4:
                # 2. Permute dummy indices (swap variables of particles 1 and 2): 
                # <p_1 q_2|r_1 s_2> = <q_1 p_2|s_1 r_2>
                qp, sr = convert_indices(n, q,p,s,r)
                integral[qp, sr] = integral[pq, rs]
                integral[sr, qp] = integral[rs, pq]
            if sym == 8:
                # 3. Permute orbitals of the same variable, e.g. <p_1 q_2|r_1 s_2> = <r_1 q_2|p_1 s_2>
                rq, ps = convert_indices(n, r,q,p,s)
                sp, qr = convert_indices(n, s,p,q,r)               
                integral[rq, ps] = integral[pq, rs]
                integral[ps, rq] = integral[rs, pq]
                integral[sp, qr] = integral[qp, sr]
                integral[qr, sp] = integral[sr, qp]
    return integral
