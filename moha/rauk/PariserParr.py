r"""PariserParr Module."""
import numpy as np

from scipy import constants

from scipy.special import gamma

from pathlib import Path

import json

from moha.rauk.rauk import build_one_body

from moha.rauk.utils import get_atom_type, get_atoms_list

import numpy as np
from scipy.special import gamma
from pathlib import Path
import json


def populate_PP_dct(
        distance,
        atom1_name,
        atom2_name,
        ionization,
        Sxy=None,
        bond_type='sigma'):
    r"""
    Calculate the beta for a bond based on atomic ionization and distance.

    Parameters
    ----------
    distance : float
        Distance between the two atoms.
    atom1_name : str
        Name of the first atom.
    atom2_name : str
        Name of the second atom.
    ionization : dict
        Dictionary containing ionization energies of atoms.

    Returns
    -------
    float
        Calculated beta parameter for the bond.
    """
    ev_H = constants.value('electron volt-hartree relationship')
    alpha_x = float(-ionization[atom1_name]) * ev_H
    alpha_y = float(-ionization[atom2_name]) * ev_H
    if Sxy is None:
        Rxy = float(distance)
        p = - 0.5 * Rxy * (alpha_x + alpha_y)
        t = abs((alpha_x - alpha_y) / (alpha_x + alpha_y))
        if bond_type == 'sigma':
            Sxy = Sxy_sigma(t, p)
        elif bond_type == 'pi':
            Sxy = Sxy_pi(t, p)
        else:
            raise ValueError(f"Invalid bond type: {bond_type}")
    beta_xy = 1.75 * Sxy * (alpha_x + alpha_y) * 0.5
    return beta_xy


def an(n, x):
    r"""
    Compute the nth term of series 'a' based on x.

    Parameters
    ----------
    n : int
        Term index.
    x : float
        Value at which series is evaluated.

    Returns
    -------
    float
        Value of the nth term of series 'a'.
    """
    sum = 0
    for k in range(1, n + 2):
        frac = 1.0 / (x**k * gamma(n - k + 1 + 1))
        sum += frac
    return gamma(n + 1) * sum


def bn(n, x):
    r"""
    Compute the nth term of series 'b' based on x.

    Parameters
    ----------
    n : int
        Term index.
    x : float
        Value at which series is evaluated.

    Returns
    -------
    float
        Value of the nth term of series 'b'.
    """
    sum = 0
    for k in range(1, n + 2):
        frac = ((-1)**(n - k)) / (x**k * gamma(n - k + 1 + 1))
        sum += frac
    return gamma(n + 1) * sum


def Bn(n, t, p):
    r"""
    Calculate the nth B parameter in the overlap integral formula.

    Parameters
    ----------
    n : int
        Term index.
    t : float
        Parameter t in the formula.
    p : float
        Parameter p in the formula.

    Returns
    -------
    float
        Calculated Bn value.
    """
    if t == 0:
        return 2 / (n + 1)
    return -np.exp(-p * t) * an(n, p * t) - np.exp(p * t) * bn(n, p * t)


def An(n, p):
    r"""
    Calculate the nth A parameter in the overlap integral formula.

    Parameters
    ----------
    n : int
        Term index.
    p : float
        Parameter p in the formula.

    Returns
    -------
    float
        Calculated An value.
    """
    return np.exp(-p) * an(n, p)


def Sxy_sigma(t, p):
    r"""
    Calculate the overlap integral Sxy for a sigma bond.

    Parameters
    ----------
    t : float
        Parameter t in the formula.
    p : float
        Parameter p in the formula.

    Returns
    -------
    float
        Calculated Sxy value.
    """
    if t == 0:
        return np.exp(-p) * (1 + p + (1 / 3) * p**2)
    else:
        A2 = An(2, p)
        A0 = An(0, p)
        B0 = Bn(0, t, p)
        B2 = Bn(2, t, p)
        return ((1 - t**2)**(3 / 2) * p**3) * (A2 * B0 - A0 * B2) / 4


def Sxy_pi(t, p):
    r"""
    Calculate the overlap integral Sxy for a pi bond.

    Parameters
    ----------
    t : float
        Parameter t in the formula.
    p : float
        Parameter p in the formula.

    Returns
    -------
    float
        Calculated Sxy value.
    """
    if t == 0:
        return np.exp(-p) * (1 + p + (2 / 5) * p**2 + (1 / 15) * (p**3))
    elif p == 0:
        return (1 - t**2)**(5 / 2)
    else:
        A4 = Bn(0, t, p) - Bn(2, t, p)
        A2 = Bn(4, t, p) - Bn(0, t, p)
        A0 = Bn(2, t, p) - Bn(4, t, p)
        return (A4 * An(4, p) + A2 * An(2, p) + A0 * An(0, p)) * \
               ((1 - t**2)**(5 / 2)) * (p**5) / 32


def compute_overlap(
        connectivity, atom_dictionary,
        bond_dictionary, orbital_overlap):
    r"""
    Compute the parameterized distance overlap matrix for a set of atoms.

    Parameters
    ----------
    connectivity : list of tuples
        List defining connectivity between atoms (atom1, atom2, order).
    atom_dictionary : dict
        Dictionary mapping atom types to properties.
    bond_dictionary : dict
        Dictionary mapping pairs of atom types to bond properties.
    orbital_overlap : dict
        Dictionary mapping pairs of atom types to orbital overlap properties.

    Returns
    -------
    scipy.sparse.csr_matrix
        The one-body matrix constructed based on the above parameters.
    """
    if atom_dictionary is None:
        ionization_path = Path(__file__).parent / "ionization.json"
        ionization = json.load(open(ionization_path, "rb"))
        atom_dictionary = {}  # alpha as first ionization potential
        ev_H = constants.value('electron volt-hartree relationship')
        for tpl in connectivity:
            atom, _ = get_atom_type(tpl[0])
            if atom not in atom_dictionary.keys():
                atom_dictionary[atom] = -ionization[atom] * ev_H
    if bond_dictionary is None:
        ionization_path = Path(__file__).parent / "ionization.json"
        ionization = json.load(open(ionization_path, "rb"))
        bond_dictionary = {}
        if orbital_overlap is None:
            for atom1, atom2, dist, bond_type in connectivity:
                atom1_name, _ = get_atom_type(atom1)
                atom2_name, _ = get_atom_type(atom2)
                bond_key_forward = ','.join([atom1_name, atom2_name])
                bond_key_reverse = ','.join([atom2_name, atom1_name])
                beta_xy = populate_PP_dct(
                    dist, atom1_name, atom2_name, ionization,
                    bond_type=bond_type)
                bond_dictionary[bond_key_forward] = beta_xy
                bond_dictionary[bond_key_reverse] = beta_xy
        else:
            for tpl in connectivity:
                atom1, atom2, dist = tpl
                atom1_name, site1 = get_atom_type(atom1)
                atom2_name, site2 = get_atom_type(atom2)
                bond_key_forward = ','.join([atom1_name, atom2_name])
                bond_key_reverse = ','.join([atom2_name, atom1_name])
                site1, site2 = site1 - 1, site2 - 1

                Sxy = orbital_overlap[site1, site2]

                beta_xy = populate_PP_dct(
                    dist, atom1_name, atom2_name, ionization, Sxy)
                bond_dictionary[bond_key_forward] = beta_xy
                bond_dictionary[bond_key_reverse] = beta_xy

    one_body = build_one_body(
        connectivity,
        atom_dictionary,
        bond_dictionary)

    return one_body


def compute_gamma(connectivity, U_xy=None, Rxy_matrix=None,
                  atom_dictionary=None, affinity_dct=None, atom_types=None):
    r"""
    Calculate the gamma values for each pair of sites.

    Parameters
    ----------
    u_onsite (list of float): List of potential energies for sites.
    Rxy_list (list of float): List of distances
    connectivity (list of tuples): Each tuple contains indices of two
                                   sites (atom1, atom2), indicating that a
                                   gamma value should be computed for
                                   this pair.
    atom_types (list of str): List of atom types in the lattice.

    Returns
    ----------
    np.ndarray: A matrix of computed gamma values for each pair.
                Non-connected pairs have a gamma value of zero.
    """
    if Rxy_matrix is None and U_xy is None:
        U_xy, Rxy_matrix = compute_u(
            connectivity, atom_dictionary, affinity_dct, atom_types)
    elif Rxy_matrix is None:
        _, Rxy_matrix = compute_u(
            connectivity, atom_dictionary, affinity_dct, atom_types)
    num_sites = len(U_xy)
    gamma_matrix = np.zeros((num_sites, num_sites))

    for tpl in connectivity:
        atom1, atom2 = tpl[0], tpl[1]
        atom1_name, site1 = get_atom_type(atom1)
        atom2_name, site2 = get_atom_type(atom2)
        # Get_atom_type returns site index starting from 1
        # So we need to subtract 1 to get the correct index
        site1, site2 = site1 - 1, site2 - 1
        Ux = U_xy[site1]
        Uy = U_xy[site2]
        Rxy = Rxy_matrix[site1][site2]
        Uxy_bar = 0.5 * (Ux + Uy)
        gamma = Uxy_bar / \
            (Uxy_bar * Rxy + np.exp(-0.5 * Uxy_bar**2 * Rxy**2))
        gamma_matrix[site1][site2] = gamma
        gamma_matrix[site2][site1] = gamma  # Ensure symmetry

    return gamma_matrix


def compute_u(connectivity, atom_dictionary, affinity_dictionary, atom_types):
    r"""
    Calculate the onsite potential energy (U) for each site.

    Parameters
    ----------
    connectivity (list of tuples): Each tuple contains indices of two sites
                                (atom1, atom2) and the distance between them.
    atom_dictionary (dict): Dictionary mapping atom types to their
                            ionization energies.
    affinity_dictionary (dict): Dictionary mapping atom types to their
                            electron affinities.
    atom_types (list of str): List of atom types in the lattice.

    Returns
    ----------
    tuple: Contains two elements:
           1. List of float: Onsite potential energies calculated
           for each site based on their ionization energy
           and electron affinity.
           2. List of float: Distances corresponding to each
           connectivity tuple.
    """
    if atom_dictionary is None:
        atom_dictionary = {}
        hx_dictionary_path = Path(__file__).parent / "hx_dictionary.json"
        hx_dictionary = json.load(open(hx_dictionary_path, "rb"))
        alpha_c = -0.414  # Value for sp2 orbital of Carbon atom.
        beta_c = -0.0533  # Value for sp2 orbitals of Carbon atom.
        for key, value in hx_dictionary.items():
            hx_value = value * abs(beta_c)
            atom_dictionary[key] = alpha_c + hx_value

    if affinity_dictionary is None:
        affinity_path = Path(__file__).parent / "affinity.json"
        affinity_dictionary = json.load(open(affinity_path, "rb"))
    if connectivity is not None:
        unique_atoms = {atom for tpl in connectivity for atom in tpl[:2]}
    else:
        unique_atoms = atom_types
        u_onsite = []
        for atom in unique_atoms:
            if atom in atom_dictionary:
                ionization = atom_dictionary[atom]
                affinity = affinity_dictionary[atom]
                U_x = ionization - affinity
                u_onsite.append(U_x)
        Rxy_matrix = np.zeros((len(unique_atoms), len(unique_atoms)))
        return u_onsite, Rxy_matrix

    num_sites = len(unique_atoms)

    u_onsite = []
    Rxy_matrix = np.zeros((num_sites, num_sites))

    i = 0
    for tpl in connectivity:
        atom1, atom2, dist = tpl[0], tpl[1], tpl[2]
        atom1_name, site1 = get_atom_type(atom1)
        atom2_name, site2 = get_atom_type(atom2)
        # Get_atom_type returns site index starting from 1
        # So we need to subtract 1 to get the correct index
        site1, site2 = site1 - 1, site2 - 1

        ionization = atom_dictionary[atom1_name]
        affinity = affinity_dictionary[atom1_name]
        U_x = ionization - affinity

        u_onsite.append(U_x)

        Rxy_matrix[site1][site2] = dist
        Rxy_matrix[site2][site1] = dist  # Ensure symmetry
    atoms_sites_lst = get_atoms_list(connectivity)
    max_site = max([site for _, site in atoms_sites_lst])
    if len(connectivity) != max_site:
        tpl = connectivity[-1]
        atom_name, _ = get_atom_type(tpl[1])
        ionization = atom_dictionary[atom_name]
        affinity = affinity_dictionary[atom_name]
        U_x = ionization - affinity

        u_onsite.append(U_x)

    return u_onsite, Rxy_matrix
