"""File containing functions for generating hamiltonian from toml file."""

import toml
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
    default_data = toml.load(required_default_paramfile)

    # set defaults in input_data
    for param_type in default_data.keys():
        # set required param type keys if not specified in input_data
        if param_type not in input_data.keys():
            input_data[param_type] = {}

        # set required default param values if not specified in input_data
        for param in default_data[param_type].keys():
            if param not in input_data[param_type]:
                input_data[param_type][param] = default_data[param_type][param]

                # set carbon params as default in Huckel model
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
            if isinstance(data_value, str) and param_type != "control":
                input_data[param_type][param] = data_value.lower()


def build_connectivity_1d(data):
    """
    Build adjacency matrix for 1d moltype.

    Parameters
    ----------
    data: dict
        dict containing toml input data.

    Returns
    -------
    adjacency: numpy array
        adjacency matrix
    """
    norb = data["system"]["norb"]

    if data["system"]["bc"] in ["open", "periodic"]:
        adjacency = np.eye(norb, k=1)
    else:
        raise ValueError(
            "System parameter 'bc' must be set to either 'open' or 'periodic'"
        )

    if data["system"]["bc"] == "periodic":
        adjacency[0, -1] = 1

    adjacency += adjacency.T

    return adjacency


def build_connectivity_2d(data):
    """
    Build adjacency matrix for 2d square-grid moltype.

    Parameters
    ----------
    data: dict
        dict containing toml input data.

    Returns
    -------
    adjacency: numpy array
        adjacency matrix
    """
    # set Lx and Ly
    if "Lx" not in data["system"].keys():
        raise ValueError("2d moltype was specified but Lx was not specified")
    Lx = data["system"]["Lx"]

    if "Ly" not in data["system"].keys():
        raise ValueError("2d moltype was specified but Ly was not specified")
    Ly = data["system"]["Ly"]

    nsites = Lx * Ly
    adjacency = np.zeros((nsites, nsites))
    for n in range(nsites):
        nx = n % Lx  # x index
        ny = n // Lx  # y index
        ndx = (nx + 1) % Lx  # x shift
        ndy = (ny + 1) % Ly  # y shift
        if data["system"]["bc"] == "open":
            # add x neighbours to connectivity
            if ndx != 0:  # skip edge bonds on open bc
                dn = ndx + Lx * ny
                adjacency[n, dn] = 1
            # add y neighbours to connectivity
            if ndy != 0:  # skip edge bonds on open bc
                dn = nx + Lx * ndy
                adjacency[n, dn] = 1
        elif data["system"]["bc"] == "periodic":
            # add x neighbours to connectivity
            dn = ndx + Lx * ny
            adjacency[n, dn] = 1
            # add y neighbours to connectivity
            dn = nx + Lx * ndy
            adjacency[n, dn] = 1
        else:
            raise ValueError(
                "System parameter 'bc' must be set to either 'open' "
                "or 'periodic'"
            )

    adjacency += adjacency.T

    return adjacency


def build_connectivity_molfile(data):
    """
    Build adjacency matrix for molfile moltype.

    Parameters
    ----------
    data: dict
        dict containing toml input data.

    Returns
    -------
    adjacency: numpy array
        adjacency matrix

    Notes
    -----
    Hamiltonian bonds should be defined as symbolic
    bonds in the molfile (bondtype = 0).
    """
    if "molfile" not in data["system"]:
        raise ValueError(
            "System parameter 'molfile' must be specified for"
            "moltype 'molfile'.")
    else:
        mol_file = data["system"]["molfile"]

    bonded_atoms = 0
    with open(mol_file) as f:
        bonded_atom_idx = {}
        for line_num, line in enumerate(f, start=1):
            # skip first 3 header lines
            if line_num <= 3:
                continue
            arr = line.split()
            # get number of atoms and bonds from line 4
            if line_num == 4:
                natoms = int(arr[0])
                nbonds = int(arr[1])
                adjacency = np.zeros((natoms, natoms))
            # skip lines containing list of atoms
            elif line_num <= 4 + natoms:
                continue
            # build connectivity from bonded atoms
            elif line_num <= 4 + natoms + nbonds:
                bond_type = int(arr[2])
                # skip non-symbolic bonds
                if bond_type != 0:
                    continue
                else:
                    atom1_idx = int(arr[0])
                    if atom1_idx not in bonded_atom_idx.keys():
                        bonded_atom_idx[atom1_idx] = bonded_atoms
                        bonded_atoms += 1
                    atom2_idx = int(arr[1])
                    if atom2_idx not in bonded_atom_idx.keys():
                        bonded_atom_idx[atom2_idx] = bonded_atoms
                        bonded_atoms += 1
                    mapped_atom1_idx = bonded_atom_idx[atom1_idx]
                    mapped_atom2_idx = bonded_atom_idx[atom2_idx]
                    adjacency[mapped_atom1_idx, mapped_atom2_idx] = 1
            # end reading molfile after connectivity block
            else:
                break

    data["system"]["norb"] = bonded_atoms
    data["system"]["nelec"] = bonded_atoms

    adjacency = adjacency[:bonded_atoms, :bonded_atoms]
    adjacency += adjacency.T

    return adjacency


def build_connectivity_smiles(data):
    """
    Build adjacency matrix for smiles moltype.

    Parameters
    ----------
    data: dict
        dict containing toml input data.

    Returns
    -------
    adjacency: numpy array
        adjacency matrix

    Notes
    -----
    Hamiltonian bonds should be defined as symbolic
    bonds in the molfile (bondtype = 0).
    """
    if "smiles" not in data["system"]:
        raise ValueError(
            "System parameter 'smiles' must be specified for"
            "moltype 'smiles'.")
    else:
        smiles = data["system"]["smiles"]

    # import rdkit here to avoid import error if smiles is not used
    from rdkit.Chem import MolFromSmiles, rdmolops
    mol = MolFromSmiles(smiles)
    adjacency = rdmolops.GetAdjacencyMatrix(mol)

    data["system"]["norb"] = adjacency.shape[0]
    data["system"]["nelec"] = adjacency.shape[0]
    return adjacency


def build_moha(data):
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
    # build connectivity for moltype
    print("System is:", data["system"]["moltype"])
    if data["system"]["moltype"] == "1d":
        adjacency = build_connectivity_1d(data)
    elif data["system"]["moltype"] == "2d":
        adjacency = build_connectivity_2d(data)
        # fixing double counting of bonds if periodic bc and Lx <= 2 or Ly <= 2
        adjacency[np.where(adjacency == 2)] = 1
    elif data["system"]["moltype"] == "molfile":
        adjacency = build_connectivity_molfile(data)
    elif data["system"]["moltype"] == "smiles":
        adjacency = build_connectivity_smiles(data)
    else:
        raise ValueError("Moltype " + data["system"]["moltype"] +
                         " not supported.")

    # define parameters for model
    norb = data["system"]["norb"]
    charge = float(data["model"]["charge"])
    alpha = float(data["model"]["alpha"])
    beta = float(data["model"]["beta"])
    u_onsite = float(data["model"]["u_onsite"])
    mu = float(data["model"]["mu"])
    J_eq = float(data["model"]["J_eq"])
    J_ax = float(data["model"]["J_ax"])

    # create and return hamiltonian object ham
    # -- Fermion models --#
    # PPP
    if data["model"]["hamiltonian"] == "ppp":
        charge_arr = charge * np.ones(norb)
        u_onsite_arr = u_onsite * np.ones(norb)
        ham = moha.HamPPP(connectivity=adjacency, alpha=alpha, beta=beta,
                          u_onsite=u_onsite_arr, charges=charge_arr)
        return ham
    # Huckel
    elif data["model"]["hamiltonian"] == "huckel":
        ham = moha.HamHuck(connectivity=adjacency, alpha=alpha, beta=beta)
        return ham
    # Hubbard
    elif data["model"]["hamiltonian"] == "hubbard":
        u_onsite_arr = u_onsite * np.ones(norb)
        ham = moha.HamHub(connectivity=adjacency,
                          alpha=alpha, beta=beta,
                          u_onsite=u_onsite_arr)
        return ham
    # -- Spin models --#
    # Heisenberg
    elif data["model"]["hamiltonian"] == "heisenberg":
        ham = moha.HamHeisenberg(connectivity=adjacency,
                                 mu=mu, J_eq=J_eq, J_ax=J_ax)
        return ham
    # Ising
    elif data["model"]["hamiltonian"] == "ising":
        ham = moha.HamIsing(connectivity=adjacency, mu=mu, J_ax=J_ax)
        return ham
    # Richardson-Gaudin
    elif data["model"]["hamiltonian"] == "rg":
        ham = moha.HamRG(connectivity=adjacency, mu=mu, J_eq=J_eq)
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

    # setup model hamiltonian
    ham = build_moha(data)

    # get symmetry of two-electron integrals
    sym = data["system"]["symmetry"]

    # generate integrals from ham
    ham.generate_zero_body_integral()
    ham.generate_one_body_integral(dense=True, basis='spatial basis')
    ham.generate_two_body_integral(dense=False, basis='spatial basis', sym=sym)

    # save integrals if specified in toml_file
    if data["control"]["save_integrals"]:
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
    data = toml.load(toml_file)
    ham = dict_to_ham(data)
    return ham


if __name__ == '__main__':
    toml_file = sys.argv[1]
    ham = from_toml(toml_file)
