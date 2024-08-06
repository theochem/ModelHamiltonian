r"""Utils for Rauk Module."""

import re
import numpy as np
from scipy.sparse import csr_matrix


def parse_connectivity(system, atom_types=None):
    r"""

    Parse the connectivity of the system given as list of tuples.

    Parameters
    ----------
    system: list
            list of tuples that specifies sites and bonds between them
        For example, for a linear chain of 4 sites, the connectivity
        can be specified as [(C1, C2, 1), (C2, C3, 1), (C3, C4, 1)]

    Returns
    -------
    tuple: (list np.ndarray)
            First element is a list of atoms in the order
            they apperar in the lattice,
            second element is matrix that corresponds to
            the either distance matrix,
            or adjacency matrix.
    """
    '''# check if self.connectivity is a matrix
    # if so, put assign it to self.connectivity_matrix
    # and set the atom_types to None
    if isinstance(system, np.ndarray):
        system_array = csr_matrix(system)
        self.atom_types = None
        self.n_sites = self.connectivity_matrix.shape[0]

        return None, self.connectivity_matrix'''
    atoms_sites_lst = get_atoms_list(system)
    max_site = max([site for _, site in atoms_sites_lst])
    n_sites = max_site

    # how to deal with atom_types

    if atom_types is None:
        # Initialize atom_types with None, and adjust size for 0-based
        # indexing
        atom_types = [None] * max_site
        for atom, site in atoms_sites_lst:
            # Adjust site index for 0-based array index
            atom_types[site - 1] = atom
        atom_types = atom_types
    connectivity_mtrx = np.zeros((max_site, max_site))
    atoms_dist = []
    for tpl in system:
        atom1, atom2, bond = tpl[0], tpl[1], tpl[2]
        atom1_name, site1 = get_atom_type(atom1)
        atom2_name, site2 = get_atom_type(atom2)
        atoms_dist.append((atom1_name, atom2_name, bond))
        connectivity_mtrx[site1 - 1, site2 - 1] = bond
        # numbering of sites starts from 1
    atoms_dist = atoms_dist
    connectivity_mtrx = np.maximum(connectivity_mtrx, connectivity_mtrx.T)
    connectivity_matrix = csr_matrix(connectivity_mtrx)
    return atoms_sites_lst, connectivity_matrix, n_sites, atom_types


def get_atom_type(atom):
    r"""
    Extract the atom type and position index from the atom string.

    Parameters
    ----------
    atom : str
        String representing an atom, following
        the pattern "typeposition(site)".

    Returns
    -------
    tuple
        A tuple containing the atom type (including site index) as a string and
        the position index as an integer.
    """
    # The pattern matches an initial letter sequence for the atom type,
    # followed by a number for the position, and numbers in parentheses for
    # site index to be appended.
    pattern = r"([A-Za-z]+)(\d+)\((\d+)\)"
    match = re.match(pattern, atom)

    if match:
        # If the pattern matches, append the site index to the atom type
        atom_type = match.group(1) + match.group(3)
        position_index = int(match.group(2))
    else:
        # Fallback for handling cases without parentheses
        i = 1
        while i <= len(atom) and atom[-i].isdigit():
            i += 1
        i -= 1
        if i == 0:
            raise ValueError(f"Invalid atom format: {atom}")
        atom_type = atom[:-i]
        position_index = int(atom[-i:])

    return atom_type, position_index


def get_atoms_list(connectivity, return_nsites=False):
    r"""
    Process the connectivity information of a molecular compound.

    Parameters
    ----------
    connectivity : list of tuples
        List of tuples representing bonds between atoms (atom1, atom2, order).
    return_nsites : bool, optional
        Whether to return total number of sites in the system

    Returns
    -------
    atoms_sites_lst: list
        List with atom types and site indices.
    max_site : int
        Maximum site index in the system.
    """
    atoms_sites_lst = []
    max_site = 0

    for tpl in connectivity:
        atom1, atom2 = tpl[0], tpl[1]
        atom1_name, site1 = get_atom_type(atom1)
        atom2_name, site2 = get_atom_type(atom2)
        for pair in [(atom1_name, site1), (atom2_name, site2)]:
            if pair not in atoms_sites_lst:
                atoms_sites_lst.append(pair)
        if max_site < max(site1, site2):  # finding the max index of site
            max_site = max(site1, site2)
    if return_nsites:
        return atoms_sites_lst, max_site

    return atoms_sites_lst
