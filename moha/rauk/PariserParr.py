import numpy as np

from scipy import constants

from scipy.special import gamma

from pathlib import Path

import json

from moha.rauk.rauk import build_one_body_matrix


def generate_alpha_beta(distance, atom1_name, atom2_name, ionization, ev_H):
    alpha_x = float(-ionization[atom1_name]) * ev_H
    alpha_y = float(-ionization[atom2_name]) * ev_H
    Rxy = distance
    p = -(((alpha_x) + (alpha_y)) * Rxy) / (2)
    t = abs((alpha_x - alpha_y) / (alpha_x + alpha_y))
    beta_xy = 1.75 * (Sxy(t, p)) * ((alpha_x + alpha_y) / (2))
    return beta_xy


def an(n, x):
    sum = 0
    for k in range(1, n + 2):
        frac = (1.0) / ((x**k) * gamma((n - k + 1) + 1))
        sum += frac
    return gamma(n + 1) * sum


def bn(n, x):
    sum = 0
    for k in range(1, n + 2):
        frac = ((-1)**(n - k)) / ((x**k) * gamma((n - k + 1) + 1))
        sum += frac
    return gamma(n + 1) * sum


def Bn(n, t, p):
    if t == 0:
        val = (2) / (n + 1)
    else:
        val = -np.exp(-p * t) * (an(n, p * t)) - \
            np.exp(p * t) * (bn(n, p * t))
    return val


def An(n, p):
    return (np.exp(-p)) * (an(n, p))


def Sxy(t, p):
    if t == 0:
        Sxy = (np.exp(-p)) * (1 + p + (2 / 5) * p**2 + (1 / 15) * (p**3))
    elif p == 0:
        Sxy = (1 - (t**2))**(5 / 2)
    else:
        A4 = Bn(0, t, p) - Bn(2, t, p)
        A2 = Bn(4, t, p) - Bn(0, t, p)
        A0 = Bn(2, t, p) - Bn(4, t, p)
        Sxy = (A4 * An(4, p) + A2 * An(2, p) + A0 * An(0, p)) * \
            ((1 - (t**2))**(5 / 2)) * (p**5) / 32
    return Sxy


def compute_param_dist_overlap(
        connectivity,
        atom_types,
        atoms_num,
        n_sites,
        dist_atoms,
        atom_dictionary,
        bond_dictionary):
    if atom_dictionary is None:
        ionization_path = Path(__file__).parent / "ionization.json"
        ionization = json.load(open(ionization_path, "rb"))
        atom_dictionary = {}  # alpha  as first ionization potential
        ev_H = constants.value('electron volt-hartree relationship')
        for atom in atom_types:
            if atom not in atom_dictionary.keys():
                atom_dictionary[atom] = -ionization[atom] * ev_H
            atom_dictionary = atom_dictionary
    if bond_dictionary is None:
        ev_H = constants.value('electron volt-hartree relationship')
        ionization_path = Path(__file__).parent / "ionization.json"
        ionization = json.load(open(ionization_path, "rb"))
        bond_dictionary = {}
        for trip in dist_atoms:
            beta_xy = generate_alpha_beta(
                trip[2], trip[0], trip[1], ionization, ev_H)
            bond_dictionary[trip[0] + trip[1]] = beta_xy
            bond_dictionary[trip[1] + trip[0]] = beta_xy

    one_body = build_one_body_matrix(
        connectivity,
        atom_dictionary,
        atoms_num,
        n_sites,
        bond_dictionary)

    return one_body
