Occupation-based Hamiltonians
=============================
Pariser-Parr-Pople(PPP) Hamiltonian model
-----------------------------------------
This model is used to describe molecules with π orbitals.

.. autoclass:: moha.HamPPP
    :members:

    .. automethod:: __init__

**Example Usage:**  
The example shown below is a linear chain of 4 carbon atoms (butadiene). 

.. code-block:: python

    import numpy as np
    from moha import HamPPP

    # Define connectivity as a list of tuples: (site1, site2, bond_type) 
    connectivity = [('C1', 'C2', 1), ('C2', 'C3', 1), ('C3', 'C4', 1)]
    gamma = np.array([
            [1.92, 1.00, 0.60, 0.41],
            [1.00, 1.92, 0.91, 0.60],
            [0.60, 0.91, 1.92, 1.00],
            [0.41, 0.60, 1.00, 1.92]])
    charges = np.array([1, 1, 1, 1])
    ppp = HamPPP(connectivity=connectivity, gamma=gamma, charges=charges, u_onsite=np.array([1, 1, 1, 1]))
    H = ppp.generate_one_body_integral(basis='spatial basis', dense=True)

    print("One-body Integral Matrix of Pariser-Parr-Pople Hamiltonian model:\n", H)

Hubbard Hamiltonian model
-------------------------
The Hubbard model is a simplified version of the PPP model with gamma being 0. 

.. autoclass:: moha.HamHub
    :members:

    .. automethod:: __init__

**Example Usage:**  

.. code-block:: python

    import numpy as np
    from moha import HamHub

    connectivity = [('C1', 'C2', 1), ('C2', 'C3', 1), ('C3', 'C4', 1)]
    hub = HamHub(connectivity=connectivity, u_onsite = np.array([1, 1, 1, 1]))
    H = hub.generate_one_body_integral(basis='spatial basis', dense=True)

    print("One-body Integral Matrix of Hubbard Hamiltonian model:\n", H)

Huckel Hamiltonian model
------------------------
The Huckel model is another simplified Hamiltonian model with U_onsite and gamma both being 0. 

.. autoclass:: moha.HamHuck
    :members:

    .. automethod:: __init__

**Example Usage:**  

.. code-block:: python

    import numpy as np
    from moha import HamHuck
 
    connectivity = [('C1', 'C2', 1), ('C2', 'C3', 1), ('C3', 'C4', 1)]
    huckel = HamHuck(connectivity=connectivity)
    H = huckel.generate_one_body_integral(basis='spatial basis', dense=True)

    print("One-body Integral Matrix of Huckel Hamiltonian model:\n", H)