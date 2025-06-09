r"""Utils for Rauk Module."""

import re


def get_atom_type(atom):
    r"""Extract the atom type and position index from the atom string.

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
    r"""Process the connectivity information of a molecular compound.

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
