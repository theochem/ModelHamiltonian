..
    : This file is part of ModelHamiltonian.
    :
    : ModelHamiltonian is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : ModelHamiltonian is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with ModelHamiltonian. If not, see <http://www.gnu.org/licenses/>.

Installation
############

Dependencies
============

The following programs/libraries are required to run ModelHamiltonian:

-  Python_ (≥3.6)
-  NumPy_ (≥1.13)
-  SciPy_ (≥1.0)

Install dependencies
====================

The programs required to build and run ModelHamiltonian can be installed with your operating system's package
manager.

E.g., for Debian- or Ubuntu- based Linux systems:

.. code:: shell

    sudo apt-get install git python3 python3-devel python3-pip

Download ModelHamiltonian
=========================

Run the following in your shell to download ModelHamiltonian via git:

.. code:: shell

    git clone https://github.com/theochem/ModelHamiltonian.git && cd ModelHamiltonian

Install ModelHamiltonian
========================

Run the following to install ModelHamiltonian:

.. code:: shell

    python3 -m pip install .

Run the following to test ModelHamiltonian:

.. code:: shell

    pytest -v moha/test/test*
