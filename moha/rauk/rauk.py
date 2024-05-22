import numpy as np

from moha.utils import get_atom_type

from pathlib import Path

from scipy.sparse import csr_matrix, diags

import json


def build_one_body_matrix(
        connectivity,
        atom_dictionary,
        atoms_num,
        n_sites,
        bond_dictionary):
    """
    Constructs the one-body matrix for a compound.

    Parameters:
    - connectivity (list of tuples): List of tuples where each tuple
    represents a bondbetween two atoms.
    - atom_dictionary (dict): Dictionary mapping atom names and properties
    - atoms_num (list): List of tuples (atom_name, quantity).
    - n_sites (int): Total number of sites (atoms) in the molecule.
    - bond_dictionary (dict): Dictionary mapping pairs of atom names
    to properties.

    Returns:
    - scipy.sparse.csr_matrix: one-body matrix.
    """
    # Populate diagonal and non-diagonal matrix elements
    # Get the diagonal values
    diagonal_values = [atom_dictionary[atom] for atom, _ in atoms_num]
    # Create a sparse diagonal matrix
    param_diag_mtrx = diags(
        diagonal_values, offsets=0, shape=(
            n_sites, n_sites))

    param_nodiag_mtrx = np.zeros((n_sites, n_sites))
    for atom1, atom2, _ in connectivity:
        atom1_name, site1 = get_atom_type(atom1)
        atom2_name, site2 = get_atom_type(atom2)
        key1 = atom1_name  # if atom1_coord else atom1_name
        key2 = atom2_name  # if atom2_coord else atom2_name

        param_nodiag_mtrx[site1 - 1, site2 - 1] = bond_dictionary[key1 + key2]

    param_nodiag_mtrx = param_nodiag_mtrx+ param_nodiag_mtrx.T
    param_diag_mtrx = csr_matrix(param_diag_mtrx)
    param_nodiag_mtrx = csr_matrix(param_nodiag_mtrx)
    one_body = param_diag_mtrx + param_nodiag_mtrx

    return one_body


def assign_rauk_parameters(
        connectivity,
        atom_types,
        atoms_num,
        n_sites,
        atom_dictionary,
        bond_dictionary):
    """
    Assigns Rauk parameters and constructs the one-body matrix
    using predefined values and external data for specified atom
    types and connectivity.

    Parameters:
    - connectivity (list): Connections between atoms.
    - atom_types (list): Atom types in the molecule.
    - atoms_num (list): Tuples of number and types of atoms.
    - n_sites (int): Total number of atomic sites.
    - atom_dictionary (dict, optional): Atom parameters dictionary.
      If None, loaded from JSON.
    - bond_dictionary (dict, optional): Bond parameters dictionary.
      If None, loaded from JSON.

    Returns:
    - scipy.sparse.csr_matrix: Matrix representing molecular structure.
    """

    if atom_dictionary is None:
        atom_dictionary = {}
        # Paths to the JSON files
        hx_dictionary_path = Path(__file__).parent / "hx_dictionary.json"
        hx_dictionary = json.load(open(hx_dictionary_path, "rb"))

        alpha_c = -0.414  # Value for sp2 orbital of Carbon atom.
        beta_c = -0.0533  # Value for sp2 orbitals of Carbon atom.

        # Create atom dictionary using predefined values without overlap
        # parameters
        for atom in atom_types:
            atom_dictionary[atom] = alpha_c + hx_dictionary[atom] * abs(beta_c)
    if bond_dictionary is None:
        # Paths to the JSON files
        hx_dictionary_path = Path(__file__).parent / "hx_dictionary.json"
        hx_dictionary = json.load(open(hx_dictionary_path, "rb"))

        kxy_matrix_1_path = Path(__file__).parent / "kxy_matrix_1.json"
        kxy_matrix_1_list = json.load(open(kxy_matrix_1_path, "rb"))

        # Convert list back to numpy array
        kxy_matrix_1 = np.array(kxy_matrix_1_list)
        kxy_matrix = np.minimum(
            kxy_matrix_1,
            kxy_matrix_1.T)  # Symmetric matrix

        alpha_c = -0.414  # Value for sp2 orbital of Carbon atom.
        beta_c = -0.0533  # Value for sp2 orbitals of Carbon atom.

        # Create bond dictionary using predefined values without overlap
        # parameters
        bond_dictionary = {}
        for i, atom in enumerate(atom_types):
            next_atom = atom_types[i + 1]\
                if i < len(atom_types) - 1 else atom_types[0]
            index1 = list(hx_dictionary.keys()).index(atom)
            index2 = list(hx_dictionary.keys()).index(next_atom)
            bond_key = atom + next_atom
            bond_value = kxy_matrix[index1, index2] * abs(beta_c)
            bond_dictionary[bond_key] = bond_value
            bond_dictionary[next_atom + atom] = bond_value  # Ensure symmetry

    one_body = build_one_body_matrix(
        connectivity,
        atom_dictionary,
        atoms_num,
        n_sites,
        bond_dictionary)

    return one_body
