r"""Rauk Module."""
import numpy as np

from moha.rauk.utils import get_atom_type, get_atoms_list


from pathlib import Path

from scipy.sparse import csr_matrix, diags

import json


def build_one_body(
        connectivity,
        atom_dictionary,
        bond_dictionary):
    r"""
    Construct the one-body matrix for a compound.

    Parameters
    ----------
    connectivity : list of tuples
        List of tuples where each tuple represents a bond between two atoms.
    atom_dictionary : dict
        Dictionary mapping atom names to properties.
    bond_dictionary : dict
        Dictionary mapping pairs of atom names to properties.

    Returns
    -------
    scipy.sparse.csr_matrix
        One-body matrix.
    """
    # Populate diagonal and non-diagonal matrix elements
    # Create a sparse diagonal matrix
    diagonal_values = []
    atom_indices = {}

    _, n_sites = get_atoms_list(connectivity, return_nsites=True)

    # Initialize matrices and helper dictionaries
    diagonal_values = np.zeros(n_sites)
    param_nodiag_mtrx = np.zeros((n_sites, n_sites))
    atom_indices = {}

    # Populate diagonal and non-diagonal matrix elements
    for tpl in connectivity:
        atom1, atom2 = tpl[0], tpl[1]
        for atom in [atom1, atom2]:
            atom_type, index = get_atom_type(atom)
            if index not in atom_indices:
                atom_indices[index] = atom_type  # Map index to atom type
                # Place the diagonal value the first time the index is
                # encountered
                diagonal_values[index - 1] = atom_dictionary[atom_type]

        # Get indices for non-diagonal elements
        _, site1 = get_atom_type(atom1)
        _, site2 = get_atom_type(atom2)
        idx1 = site1 - 1
        idx2 = site2 - 1

        # Fill in the non-diagonal matrix, ensuring symmetry in the assignment
        bond_key1 = ','.join([atom_indices[site1], atom_indices[site2]])
        bond_key2 = ','.join([atom_indices[site2], atom_indices[site1]])
        param_nodiag_mtrx[idx1, idx2] = bond_dictionary.get(bond_key1, 0)
        param_nodiag_mtrx[idx2, idx1] = bond_dictionary.get(
            bond_key2, 0)  # Symmetry

    # Convert matrices to sparse format and combine them
    param_diag_mtrx = diags(diagonal_values, offsets=0)
    param_nodiag_mtrx = csr_matrix(param_nodiag_mtrx)
    one_body = param_diag_mtrx + param_nodiag_mtrx

    return one_body


def assign_rauk_parameters(
        connectivity,
        atom_dictionary,
        bond_dictionary):
    r"""
    Assign Rauk parameters and constructs the one-body matrix.

    It uses a  predefined values and external data
    for specified atom types and connectivity.

    Parameters
    ----------
    connectivity : list
        Connections between atoms.
    atom_dictionary : dict, optional
        Atom parameters dictionary. If None, loaded from JSON.
    bond_dictionary : dict, optional
        Bond parameters dictionary. If None, loaded from JSON.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix representing molecular structure.
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
        for atom1, atom2, _ in connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)
            hx_value = hx_dictionary[atom1_name] * abs(beta_c)
            atom_dictionary[atom1_name] = alpha_c + hx_value
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

        for atom1, atom2, _ in connectivity:
            atom1_name, site1 = get_atom_type(atom1)
            atom2_name, site2 = get_atom_type(atom2)

            # get indices of atoms
            index1 = list(hx_dictionary.keys()).index(atom1_name)
            index2 = list(hx_dictionary.keys()).index(atom2_name)

            bond_key = ','.join([atom1_name, atom2_name])
            bond_value = kxy_matrix[index1, index2] * abs(beta_c)
            bond_dictionary[bond_key] = bond_value
            # Ensure symmetry
            bond_dictionary[','.join([atom2_name, atom1_name])] = bond_value

    else:
        # Ensure symmetry in the bond dictionary
        for key in list(bond_dictionary.keys()):
            atom1_name, atom2_name = key.split(',')
            reverse_key = ','.join([atom2_name, atom1_name])
            bond_dictionary[reverse_key] = bond_dictionary[key]

    one_body = build_one_body(
        connectivity,
        atom_dictionary,
        bond_dictionary)

    return one_body
