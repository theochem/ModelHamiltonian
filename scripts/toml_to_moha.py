import tomllib
import numpy as np
from os.path import exists
from os import makedirs
import sys
sys.path.insert(0, '../')
import moha

def set_defaults(input_data):
    '''
    Function for setting defaults in the parameter dictionary
    loaded from toml input file.

    Parameters
        ----------
        input_data: dict
            dict loaded from toml input file.

    Returns
        -------
        None
    '''

    # require that defaults.toml exists
    required_default_paramfile = "defaults.toml"
    if not exists(required_default_paramfile):
        raise Exception("Default input file 'defaults.toml' is required.")
    
    # load defaults.toml data into default_data
    default_data = tomllib.load(
                       open(required_default_paramfile, "rb")
            )

    # set defaults in input_data
    for param_type in default_data.keys():
        # set required param type keys if not specified in input_data
        if not param_type in input_data.keys():
            input_data[param_type] = {}

        # set required default param values if not specified in input_data
        for param in default_data[param_type].keys():
            if not param in input_data[param_type]:
                input_data[param_type][param] = default_data[param_type][param]
            # make all strings lowercase for case-insensitive comparisons
            data_value = input_data[param_type][param]
            if type(data_value) == str:
                input_data[param_type][param] = data_value.lower()



def build_moha_moltype_1d(data):
    '''
    Function that builds and returns hamiltonian object 
    specific to the "1d" moltype.

    Supported hamiltonians are: PPP, Huckel, Hubbard, and Heisenberg.

    Parameters
        ----------
        data: dict
            dict containing toml input data.

        Returns
        -------
        moha.Ham
    '''
    # define parameters for 1d model
    norb   = data["system"]["norb"]
    alpha  = data["model"]["alpha"]
    beta   = data["model"]["beta"]
    gamma0 = data["model"]["gamma0"]
    mu     = data["model"]["mu"]
    J_eq   = data["model"]["J_eq"]
    J_ax   = data["model"]["J_ax"]

    # build connectivity
    connectivity = [(f"C{i}", f"C{i + 1}", 1) for i in range(1, norb)]
    if data["system"]["bc"] == "periodic":
        connectivity += [(f"C{norb}", f"C{1}", 1)]

    # build connectivity for spin models
    spin_connectivity = np.array(np.eye(norb, k=1) + np.eye(norb, k=-1))
    if data["system"]["bc"] == "periodic":
        spin_connectivity[(0,-1)] = spin_connectivity[(-1,0)] = 1

    # create and return hamiltonian object ham
    #-- Fermion models --#
    # PPP
    if data["model"]["hamiltonian"] == "ppp":
        gamma = gamma0 * np.eye(norb)
        ham = moha.HamPPP(connectivity=connectivity, alpha=alpha, beta=beta, gamma=gamma)
        return ham
    # Huckel
    elif data["model"]["hamiltonian"] == "huckel":
        ham = moha.HamHuck(connectivity=connectivity, alpha=alpha, beta=beta)
        return ham
    # Hubbard
    elif data["model"]["hamiltonian"] == "hubbard":
        u_onsite = np.array([0.5*gamma0 for i in range(norb)])
        ham = moha.HamHub(connectivity=connectivity, alpha=alpha, beta=beta, u_onsite=u_onsite)
        return ham
    #-- Spin models --#
    # Heisenberg
    elif data["model"]["hamiltonian"] == "heisenberg":
        ham = moha.HamHeisenberg(connectivity=spin_connectivity, mu=mu, J_eq=J_eq, J_ax=J_ax)
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
                         " not supported for moltype " + data["system"]["moltype"] + ".")

def toml_to_ham(toml_file):
    '''
    Function for generating hamiltonian from toml file.
    Prints integrals to output file if specified in toml_file.

    Parameters
        ----------
        toml_file: str
            filename of toml input file

        Returns
        -------
        moha.Ham
    '''
    # load data from toml file 
    data = tomllib.load(open(toml_file, "rb"))

    # set any missing required values as defaults
    set_defaults(data)

    # setup model hamiltonian for specified moltype
    if data["system"]["moltype"] == "1d":
        ham = build_moha_moltype_1d(data)
    else:
        raise ValueError("moltype " + data["system"]["moltype"] + " not supported.")

    # generate integrals from ham
    ham.generate_zero_body_integral()
    ham.generate_one_body_integral(dense=True, basis='spatial basis')
    ham.generate_two_body_integral(dense=False, basis='spatial basis')

    # save integrals if specified in toml_file
    if data["control"]["save_intergrals"]:
        # save integrals to outdir if specified in toml_file
        if not exists(data["control"]["outdir"]):
            makedirs(data["control"]["outdir"], exist_ok=True)
        out_file = data["control"]["outdir"] + "/" + data["control"]["prefix"] + ".out"

        # save output
        if data["control"]["integral_format"] == "fcidump":
            fout = open(out_file, "w")
            ham.save_fcidump(f=fout, nelec=data["system"]["nelec"])
            fout.close()
        else:
            raise ValueError("Integral output format " + data["control"]["integral_format"] + " not supported.")

    return ham

toml_file = sys.argv[1]
ham = toml_to_ham(toml_file)
