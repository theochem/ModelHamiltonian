import numpy as np 

from scipy import constants 

from scipy.special import gamma

from moha.utils import get_atom_type

from pathlib import Path

from scipy.sparse import csr_matrix, diags
import os
import json

def build_one_body_matrix(connectivity, atom_dictionary, atoms_num, n_sites, bond_dictionary):
    # Populate diagonal and non-diagonal matrix elements
    param_diag_mtrx = np.zeros((n_sites, n_sites))
    for atom, site in atoms_num:
        key = atom
        param_diag_mtrx[site - 1, site - 1] = atom_dictionary[key]
    print(param_diag_mtrx)

    param_nodiag_mtrx = np.zeros((n_sites, n_sites))
    for atom1, atom2, _ in connectivity:
        atom1_name, site1 = get_atom_type(atom1)
        atom2_name, site2 = get_atom_type(atom2)
        key1 = atom1_name #if atom1_coord else atom1_name
        key2 = atom2_name #if atom2_coord else atom2_name

        #print((site1 - 1, site2 - 1), key1 + key2, self.bond_dictionary[key1 + key2])
        param_nodiag_mtrx[site1 - 1, site2 - 1] = bond_dictionary[key1 + key2]
        
    param_nodiag_mtrx = np.minimum(param_nodiag_mtrx, param_nodiag_mtrx.T)
    param_nodiag_mtrx = param_nodiag_mtrx
    param_diag_mtrx = csr_matrix(param_diag_mtrx)
    param_nodiag_mtrx = csr_matrix(param_nodiag_mtrx)
    
    one_body = param_diag_mtrx + param_nodiag_mtrx
    
    return one_body

def assign_rauk_parameters(connectivity, atom_types, atoms_num, n_sites):
    r"""Assigns the alpha and beta value from Rauk's table in matrix form"""


    # Paths to the JSON files
    hx_dictionary_path = Path(__file__).parent / "hx_dictionary.json"
    hx_dictionary = json.load(open(hx_dictionary_path, "rb"))
    
    kxy_matrix_1_path = Path(__file__).parent / "kxy_matrix_1.json"
    kxy_matrix_1_list = json.load(open(kxy_matrix_1_path, "rb"))

    # Convert list back to numpy array
    kxy_matrix_1 = np.array(kxy_matrix_1_list)
    kxy_matrix = np.minimum(kxy_matrix_1, kxy_matrix_1.T) #Symmetric matrix

    alphaC = -0.414 #Value for sp2 orbital of Carbon atom.
    betaC = -0.0533 #Value for sp2 orbitals of Carbon atom.
    
    
    # Create atom dictionary using predefined values without overlap parameters
    atom_dictionary = {}
    atom_types.pop(0)
    print(atom_types)
    for atom in atom_types:
        atom_dictionary[atom] = alphaC + hx_dictionary[atom] * abs(betaC)
 
    #print(atom_dictionary)
    # Create bond dictionary using predefined values without overlap parameters
    bond_dictionary = {}
    for i, atom in enumerate(atom_types):
        next_atom = atom_types[i + 1] if i < len(atom_types) - 1 else atom_types[0]
        #print(atom, next_atom)
        index1 = list(hx_dictionary.keys()).index(atom)
        index2 = list(hx_dictionary.keys()).index(next_atom)
        bond_key = atom + next_atom
        bond_value = kxy_matrix[index1, index2] * abs(betaC)
        bond_dictionary[bond_key] = bond_value
        #print(bond_key, bond_value)
        bond_dictionary[next_atom + atom] = bond_value  # Ensure symmetry
    #print(bond_dictionary)
    
    one_body = build_one_body_matrix(connectivity, atom_dictionary, atoms_num, n_sites, bond_dictionary)
    return one_body


def compute_param_dist_overlap(connectivity, atom_types, atoms_num, n_sites, dist_atoms):
    
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # Paths to the JSON files
    ionization_path = os.path.join(dir_path, 'ionization.json')

    # Load the dictionary from the hx_dictionary.json file
    with open(ionization_path, 'r') as file:
        ionization = json.load(file)
        
    ### This function calculates the beta value from the Wolfsberg-Helmholz approximation and uses alpha as 
    # the first ionization potential of the atom
    ev_H = constants.value('electron volt-hartree relationship')
    def generate_alpha_beta(distance,atom1_name,atom2_name,ionization):
        alpha_x = float(-ionization[atom1_name]) * ev_H  
        alpha_y = float(-ionization[atom2_name]) * ev_H  
        Rxy = distance 
        p = -(( (alpha_x) + (alpha_y) )* Rxy) /(2) 
        t = abs((alpha_x - alpha_y )/(alpha_x + alpha_y))

        def an(n, x):
            sum = 0
            for k in range(1, n+2):
                frac = (1.0) / ((x**k)*gamma((n-k+1) + 1))
                sum += frac
            return gamma(n + 1) * sum

        def bn(n, x):
            sum = 0
            for k in range(1, n+2):
                frac = ((-1)**(n-k)) / ((x**k)*gamma((n-k+1) + 1))
                sum += frac
            return gamma(n + 1) * sum

        def Bn(n, t, p):
            if t == 0:
                val = (2) / (n + 1)
            else:
                val = -np.exp(-p*t) * (an(n, p*t)) - \
                    np.exp(p*t) * (bn(n, p*t))
            return val

        def An(n, p):
            return (np.exp(-p))*(an(n, p))

        def Sxy(t, p):
            if t == 0:
                Sxy = (np.exp(-p))*(1 + p + (2/5)*p**2 + (1/15)*(p**3))
            elif p == 0:
                Sxy = (1 - (t**2))**(5/2)
            else:
                A4 = Bn(0, t, p) - Bn(2, t, p)
                A2 = Bn(4, t, p) - Bn(0, t, p)
                A0 = Bn(2, t, p) - Bn(4, t, p)
                Sxy = (A4*An(4, p) + A2*An(2, p) + A0*An(0, p)) * \
                    ((1 - (t**2))**(5/2))*(p**5)/32
            return Sxy

        beta_xy = 1.75*(Sxy(t, p)) * ((alpha_x + alpha_y)/(2))

        return beta_xy
    atom_types.pop(0)
    atom_dictionary = {} #Defines alpha values as first ionization potential 
    for atom in atom_types:
        if atom not in atom_dictionary.keys():
            atom_dictionary[atom] = -ionization[atom] * ev_H  #eV to Hartree
        atom_dictionary = atom_dictionary
        
    bond_dictionary = {}
    print(dist_atoms)
    for trip in dist_atoms:
        beta_xy = generate_alpha_beta(trip[2],trip[0],trip[1],ionization) 
        bond_dictionary[trip[0]+trip[1]] = beta_xy
        bond_dictionary[trip[1]+trip[0]] = beta_xy
        
    one_body = build_one_body_matrix(connectivity, atom_dictionary, atoms_num, n_sites, bond_dictionary)
    
    return one_body




    
    
