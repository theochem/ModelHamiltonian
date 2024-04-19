"""File containing functions for generating hamiltonian from toml file."""

import tomllib
import numpy as np
from os.path import exists
from os.path import join
from os import makedirs
import sys
import moha
from pathlib import Path


def set_defaults(input_data):
    """
    Set defaults in the parameter dictionary\
    loaded from toml input file.

    Parameters
    ----------
    input_data: dict
        dict loaded from toml input file.

    Returns
    -------
    None
    """
    # require that defaults.toml exists
    required_default_paramfile = Path(__file__).parent / "defaults.toml"
    if not exists(required_default_paramfile):
        raise Exception("Default input file 'defaults.toml' is required.")

    # load defaults.toml data into default_data
    default_data = tomllib.load(
                       open(required_default_paramfile, "rb")
            )

    # set defaults in input_data
    for param_type in default_data.keys():
        # set required param type keys if not specified in input_data
        if param_type not in input_data.keys():
            input_data[param_type] = {}

        # set required default param values if not specified in input_data
        for param in default_data[param_type].keys():
            if param not in input_data[param_type]:
                input_data[param_type][param] = default_data[param_type][param]
                # set Carbon params as default in Huckel model
                if param_type == "model" \
                    and input_data["model"]["hamiltonian"].lower() == "huckel"\
                        and param == "alpha":
                    input_data["model"]["alpha"] = -0.414
                if param_type == "model" \
                    and input_data["model"]["hamiltonian"].lower() == "huckel"\
                        and param == "beta":
                    input_data["model"]["beta"] = -0.0533
            # make all strings lowercase for case-insensitive comparisons
            data_value = input_data[param_type][param]
            if type(data_value) == str:
                input_data[param_type][param] = data_value.lower()


def build_moha_moltype_1d(data):
    """
    Build and return hamiltonian object\
    specific to the "1d" moltype.

    Supported hamiltonians are: PPP, Huckel, \
        Hubbard, Heisenberg, Ising, and RG.
    Only diagonal form of gamma matrix for PPP model is supported:
    gamma_pq = gamma0 * delta_pq.

    Parameters
    ----------
    data: dict
        dict containing toml input data.

    Returns
    -------
    ham: moha.Ham
        model hamiltonian object.
    """
    # define parameters for 1d model
    norb = data["system"]["norb"]
    charge = float(data["model"]["charge"])
    alpha = float(data["model"]["alpha"])
    beta = float(data["model"]["beta"])
    u_onsite = float(data["model"]["u_onsite"])
    mu = float(data["model"]["mu"])
    J_eq = float(data["model"]["J_eq"])
    J_ax = float(data["model"]["J_ax"])

    # build connectivity
    connectivity = [(f"C{i}", f"C{i + 1}", 1) for i in range(1, norb)]
    if data["system"]["bc"] == "periodic":
        connectivity += [(f"C{norb}", f"C{1}", 1)]

    # build connectivity for spin models
    spin_connectivity = np.array(np.eye(norb, k=1) + np.eye(norb, k=-1))
    if data["system"]["bc"] == "periodic":
        spin_connectivity[(0, -1)] = spin_connectivity[(-1, 0)] = 1

    # create and return hamiltonian object ham
    # -- Fermion models --#
    # PPP
    if data["model"]["hamiltonian"] == "ppp":
        charge_arr = charge * np.ones(norb)
        u_onsite_arr = u_onsite * np.ones(norb)
        ham = moha.HamPPP(connectivity=connectivity, alpha=alpha, beta=beta,
                          u_onsite=u_onsite_arr, charges=charge_arr)
        return ham
    # Huckel
    elif data["model"]["hamiltonian"] == "huckel":
        ham = moha.HamHuck(connectivity=connectivity, alpha=alpha, beta=beta)
        return ham
    # Hubbard
    elif data["model"]["hamiltonian"] == "hubbard":
        u_onsite_arr = u_onsite * np.ones(norb)
        ham = moha.HamHub(connectivity=connectivity,
                          alpha=alpha, beta=beta,
                          u_onsite=u_onsite_arr)
        return ham
    # -- Spin models --#
    # Heisenberg
    elif data["model"]["hamiltonian"] == "heisenberg":
        ham = moha.HamHeisenberg(connectivity=spin_connectivity,
                                 mu=mu, J_eq=J_eq, J_ax=J_ax)
        return ham
    # Ising
    elif data["model"]["hamiltonian"] == "ising":
        ham = moha.HamIsing(connectivity=spin_connectivity, mu=mu, J_ax=J_ax)
        return ham
    # Richardson-Gaudin
    elif data["model"]["hamiltonian"] == "rg":
        ham = moha.HamRG(connectivity=spin_connectivity, mu=mu, J_eq=J_eq)
        return ham
    else:
        raise ValueError("Model hamiltonian " + data["model"]["hamiltonian"] +
                         " not supported for moltype " +
                         data["system"]["moltype"] + ".")


def dict_to_ham(data):
    """
    Generate hamiltonian from dictionary of model data.

    Parameters
    ----------
    data: dict
        dictionary containing model data.
        supported keys are specified in defaults.toml

    Returns
    -------
    hamiltonian: moha.Ham
        model hamiltonian object.
    """
    # set any missing required values as defaults
    set_defaults(data)

    # setup model hamiltonian for specified moltype
    if data["system"]["moltype"] == "1d":
        ham = build_moha_moltype_1d(data)
    else:
        raise ValueError("moltype " + data["system"]["moltype"] +
                         " not supported.")

    # get symmetry of two-electron integrals
    sym = data["system"]["symmetry"]

    # generate integrals from ham
    ham.generate_zero_body_integral()
    ham.generate_one_body_integral(dense=True, basis='spatial basis')
    ham.generate_two_body_integral(dense=False, basis='spatial basis', sym=sym)

    # save integrals if specified in toml_file
    if data["control"]["save_integrals"] == 'true':
        # save integrals to outdir if specified in toml_file
        if not exists(data["control"]["outdir"]):
            makedirs(data["control"]["outdir"], exist_ok=True)
        out_file = join(data["control"]["outdir"], data["control"]["prefix"])

        # save output
        if data["control"]["integral_format"] == "fcidump":
            out_file += ".fcidump"
            with open(out_file, "w") as fout:
                ham.save_fcidump(f=fout, nelec=data["system"]["nelec"])
        elif data["control"]["integral_format"] == "npz":
            ham.savez(out_file)
        else:
            raise ValueError("Integral output format " +
                             data["control"]["integral_format"] +
                             " not supported.")

    return ham


def from_toml(toml_file):
    """
    Generate hamiltonian from toml file.

    Parameters
    ----------
    toml_file: str
        path to toml file containing model data.

    Returns
    -------
    moha.Ham
    """
    data = tomllib.load(open(toml_file, "rb"))
    ham = dict_to_ham(data)
    return ham


if __name__ == '__main__':
    toml_file = sys.argv[1]
    data = from_toml(toml_file)
    ham = dict_to_ham(data)
