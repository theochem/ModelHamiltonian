import tomllib
import numpy as np
import sys  
sys.path.insert(0, '../')
import moha

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
    with open(toml_file, "rb") as f:
        data = tomllib.load(f)
    f.close()

    norb    = 0 if not "norb"   in data["system"] else data["system"]["norb"]
    nelec   = 0 if not "nelec"  in data["system"] else data["system"]["nelec"]
    alpha   = 0 if not "alpha"  in data["model"]  else data["model"]["alpha"]
    beta    = 0 if not "beta"   in data["model"]  else data["model"]["beta"]
    gamma0  = 0 if not "gamma0" in data["model"]  else data["model"]["gamma0"]

    if data["system"]["moltype"] == "1d":
        # build connectivity
        connectivity = [(f"C{i}", f"C{i + 1}", 1) for i in range(1, norb)]
        if data["system"]["bc"] == "periodic":
            connectivity += [(f"C{norb}", f"C{1}", 1)]

        # create hamiltonian object ham
        if data["model"]["hamiltonian"] == "hubbard":
            u_onsite = np.array([0.5*gamma0 for i in range(norb)])
            charges = np.ones(norb)
            ham = moha.HamHub(connectivity=connectivity, alpha=alpha, beta=beta, u_onsite=u_onsite)
        else:
            raise ValueError("Model hamiltonian " + data["model"]["hamiltonian"] + 
                             " not supported for moltype " + data["system"]["moltype"] + ".")
    else:
        raise ValueError("moltype " + data["system"]["moltype"] + " not supported.")

    # generate integrals from ham
    ham.generate_zero_body_integral()
    ham.generate_one_body_integral(dense=True, basis='spatial basis')
    ham.generate_two_body_integral(dense=False, basis='spatial basis')

    # save integrals if specified in toml_file
    if data["control"]["save_intergrals"]:
        out_file = data["control"]["outdir"] + "/" + data["control"]["prefix"] + ".out"
        if data["control"]["integral_format"] == "fcidump":
            fout = open(out_file, "w")
            ham.save_fcidump(f=fout, nelec=nelec)
            fout.close()
        else:
            raise ValueError("Integral output format " + data["control"]["integral_format"] + " not supported.")

    return ham

toml_file = sys.argv[0]
ham = toml_to_ham(toml_file)
