r"""PariserParr Module."""
import numpy as np

from scipy import constants

from scipy.special import gamma

from pathlib import Path

import json

from moha.rauk.rauk import build_one_body

import numpy as np
from scipy.special import gamma
from pathlib import Path
import json


def generate_alpha_beta(distance, atom1_name, atom2_name, ionization, ev_H):
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
    ev_H : float
        Conversion factor from electron volts to hartrees.

    Returns
    -------
    float
        Calculated beta parameter for the bond.
    """
    alpha_x = float(-ionization[atom1_name]) * ev_H
    alpha_y = float(-ionization[atom2_name]) * ev_H
    Rxy = distance
    p = -(((alpha_x) + (alpha_y)) * Rxy) / 2
    t = abs((alpha_x - alpha_y) / (alpha_x + alpha_y))
    beta_xy = 1.75 * (Sxy(t, p)) * ((alpha_x + alpha_y) / 2)
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
    else:
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


def Sxy(t, p):
    r"""
    Calculate the overlap integral Sxy for given parameters t and p.

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


def compute_param_dist_overlap(
        connectivity, atom_types, atoms_num, n_sites, atoms_dist,
        atom_dictionary, bond_dictionary):
    r"""
    Compute the parameterized distance overlap matrix for a set of atoms.

    Parameters
    ----------
    connectivity : list of tuples
        List defining connectivity between atoms (atom1, atom2, order).
    atom_types : list
        List of atom types present in the molecule.
    atoms_num : list
        List of tuples defining atom types and their quantities.
    n_sites : int
        Number of sites (atoms) in the molecule.
    atoms_dist : list
        List defining distances between connected atoms.
    atom_dictionary : dict
        Dictionary mapping atom types to properties.
    bond_dictionary : dict
        Dictionary mapping pairs of atom types to bond properties.

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
        for atom in atom_types:
            if atom not in atom_dictionary.keys():
                atom_dictionary[atom] = -ionization[atom] * ev_H
    if bond_dictionary is None:
        ev_H = constants.value('electron volt-hartree relationship')
        ionization_path = Path(__file__).parent / "ionization.json"
        ionization = json.load(open(ionization_path, "rb"))
        bond_dictionary = {}
        for trip in atoms_dist:
            beta_xy = generate_alpha_beta(
                trip[2], trip[0], trip[1], ionization, ev_H)
            bond_key_forward = ','.join([trip[0], trip[1]])
            bond_key_reverse = ','.join([trip[1], trip[0]])
            bond_dictionary[bond_key_forward] = beta_xy
            bond_dictionary[bond_key_reverse] = beta_xy

    one_body = build_one_body(
        connectivity,
        atom_dictionary,
        n_sites,
        bond_dictionary)

    return one_body
