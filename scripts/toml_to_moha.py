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
    with open(required_default_paramfile, "rb") as f:
        default_data = tomllib.load(f)
    f.close()

    # set defaults in input_data
    for param_type in default_data.keys():
        # set required param type keys if not specified in input_data
        if not param_type in input_data.keys():
            input_data[param_type] = {}

        # set required default param values if not specified in input_data
        for param in default_data[param_type].keys():
            if not param in input_data[param_type]:
                input_data[param_type][param] = default_data[param_type][param]

def build_moha_moltype_1d(data):
    '''
    Function that builds and returns hamiltonian object 
    specific to the "1d" moltype.

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

    # build connectivity
    connectivity = [(f"C{i}", f"C{i + 1}", 1) for i in range(1, norb)]
    if data["system"]["bc"] == "periodic":
        connectivity += [(f"C{norb}", f"C{1}", 1)]

    # create and return hamiltonian object ham
    if data["model"]["hamiltonian"] == "hubbard":
        u_onsite = np.array([0.5*gamma0 for i in range(norb)])
        ham = moha.HamHub(connectivity=connectivity, alpha=alpha, beta=beta, u_onsite=u_onsite)
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
    with open(toml_file, "rb") as f:
        data = tomllib.load(f)
    f.close()

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
