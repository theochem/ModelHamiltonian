r"""Utils for Rauk Module."""

import re


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


def get_atoms_list(connectivity):
    r"""
    Process the connectivity information of a molecular compound.

    Parameters
    ----------
    connectivity : list of tuples
        List of tuples representing bonds between atoms (atom1, atom2, order).

    Returns
    -------
    int
        Total number of unique atomic sites in the molecule.
    """
    atoms_sites_lst = []
    max_site = 0

    for atom1, atom2, _ in connectivity:
        atom1_name, site1 = get_atom_type(atom1)
        atom2_name, site2 = get_atom_type(atom2)
        for pair in [(atom1_name, site1), (atom2_name, site2)]:
            if pair not in atoms_sites_lst:
                atoms_sites_lst.append(pair)
        if max_site < max(site1, site2):  # finding the max index of site
            max_site = max(site1, site2)

    return atoms_sites_lst
