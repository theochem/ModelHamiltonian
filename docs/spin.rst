Spin-based Hamiltonians
=============================

This section covers Hamiltonians used to describe spin systems in quantum mechanics.

- **Heisenberg Model**: Models spin interactions with exchange coupling.
- **Ising Model**: A simplified version where only Z-axis interactions matter.
- **Richardson-Gaudin Model**: Used in exactly solvable quantum many-body problems.

----

Heisenberg Hamiltonian
----------------------
The **XXZ Heisenberg Model** describes quantum spin interactions between sites.

.. autoclass:: moha.HamHeisenberg
    :members:

    .. automethod:: __init__

**Example Usage:**  
This example initializes a **3-site Heisenberg Hamiltonian** with specified interaction terms and prints the one-body integral matrix.

.. code-block:: python

    from moha import HamHeisenberg
    import numpy as np

    # Define spin interaction parameters
    mu = np.array([0.1, 0.2, 0.3])
    J_eq = np.array([
        [1.0, 0.5, 0.3], 
        [0.5, 1.0, 0.5], 
        [0.3, 0.5, 1.0]
    ])
    J_ax = np.array([
        [0.5, 0.2, 0.1], 
        [0.2, 0.5, 0.2], 
        [0.1, 0.2, 0.5]
    ])

    # Create Heisenberg model instance
    H = HamHeisenberg(mu=mu, J_eq=J_eq, J_ax=J_ax)

    # Print the one-body integral matrix
    print("One-body Integral Matrix:\n", H.generate_one_body_integral(dense=True))

----

Ising Hamiltonian
-----------------
The **Ising Model** is a simplified case of the Heisenberg model with only axial interactions.

.. autoclass:: moha.HamIsing
    :members:

    .. automethod:: __init__

**Example Usage:**  
This example initializes a **3-site Ising Hamiltonian** and prints the one-body integral matrix.

.. code-block:: python

    from moha import HamIsing
    import numpy as np

    # Define spin interaction parameters
    mu = np.array([0.1, 0.2, 0.3])
    J_ax = np.array([
        [1.0, 0.5, 0.2], 
        [0.5, 1.0, 0.5], 
        [0.2, 0.5, 1.0]
    ])

    # Create Ising model instance
    H = HamIsing(mu=mu, J_ax=J_ax)

    # Print the one-body integral matrix
    print("One-body Integral Matrix:\n", H.generate_one_body_integral(dense=True))

----

Richardson-Gaudin Hamiltonian
-----------------------------
The **Richardson-Gaudin Model** is an exactly solvable quantum many-body system.

.. autoclass:: moha.HamRG
    :members:

    .. automethod:: __init__

**Example Usage:**  
This example initializes a **3-site Richardson-Gaudin Hamiltonian** and prints the one-body integral matrix.

.. code-block:: python

    from moha import HamRG
    import numpy as np

    # Define interaction parameters
    mu = np.array([0.1, 0.2, 0.3])
    J_eq = np.array([
        [0.5, 0.3, 0.2], 
        [0.3, 0.5, 0.3], 
        [0.2, 0.3, 0.5]
    ])

    # Create Richardson-Gaudin model instance
    H = HamRG(mu=mu, J_eq=J_eq)

    # Print the one-body integral matrix
    print("One-body Integral Matrix:\n", H.generate_one_body_integral(dense=True))
