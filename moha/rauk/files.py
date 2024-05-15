import json
import numpy as np

hx_dictionary = { #The order of the elements is important 
    "C":  0.0, 
    "B":  0.45, 
    "N2":-0.51,
    "N3":-1.37,
    "O1":-0.97,
    "O2":-2.09,
    "F": -2.71,
    "Si":  0.0,
    "P2": -0.19,
    "P3": -0.75,
    "S1": -0.46,
    "S2": -1.11,
    "Cl": -1.48
    }

#kxy elements
kxy_matrix_1 = np.array([
[-1.0  , 0.    ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-0.73 , -0.87 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-1.02 , -0.66 , -1.09 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-0.89 , -0.53 , -0.99 , -0.98 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-1.06 , -0.60 , -1.14 , -1.13 , -1.26 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-0.66 , -0.35 , -0.80 , -0.89 , -1.02 , -0.95 ,  0.   ,  0.   ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-0.52 , -0.26 , -0.65 , -0.77 , -0.92 , -0.94 , -1.04 ,  0.   ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-0.75 , -0.57 , -0.72 , -0.43 , -0.65 , -0.24 , -0.17 , -0.64 ,  0.   ,  0.   , 0.    ,  0.    , 0.   ],
[-0.77 , -0.53 , -0.78 , -0.55 , -0.75 , -0.31 , -0.21 , -0.62 , -0.63 ,  0.   , 0.    ,  0.    , 0.   ],
[-0.76 , -0.54 , -0.81 , -0.64 , -0.82 , -0.39 , -0.22 , -0.52 , -0.58 , -0.63 , 0.    ,  0.    , 0.   ],
[-0.81 , -0.51 , -0.83 , -0.68 , -0.84 , -0.43 , -0.28 , -0.61 , -0.65 , -0.65 , -0.68 ,  0.    , 0.   ],
[-0.69 , -0.44 , -0.78 , -0.73 , -0.85 , -0.54 , -0.32 , -0.40 , -0.48 , -0.60 , -0.58 , -0.63  , 0.   ],
[-0.62 , -0.41 , -0.77 , -0.80 , -0.88 , -0.70 , -0.51 , -0.34 , -0.35 , -0.55 , -0.52 , -0.59  ,-0.68 ],
])




kxy_matrix_1_list = kxy_matrix_1.tolist()


import sys  
sys.path.insert(0, '../')

import os
import json
# Your data here...

dir_path = r'C:\Users\giova\ModelHamiltonian\moha\rauk'

# Write dictionary to JSON file in the 'rauk' folder
with open(os.path.join(dir_path, 'hx_dictionary.json'), 'w') as f:
    json.dump(hx_dictionary, f)

# Write numpy array to JSON file in the 'rauk' folder
with open(os.path.join(dir_path, 'kxy_matrix_1.json'), 'w') as f:
    json.dump(kxy_matrix_1_list, f)

