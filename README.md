# Model Hamiltonian

This utility generates 1- and 2-electron integrals corresponding to various model Hamiltonians. The basic input is some indication of connectivity, either explicitly or as a lattice. One then specifies the Hamiltonian of interest. The output are 1- and 2-electron integrals in a format convenient for use in other (external) software packages. Notably, these integrals can be transformed into FCIDUMP format using [IOData](iodata.qcdevs.org).

&#x1f6e0;&#xfe0f; This module is currently under development; major API changes are expected soon!

## Installation

```
python3 -m pip install -e .
```

## Building Sphinx-Doc

```
python3 -m pip install --user sphinx-rtd-theme
cd docs && make html
```

## Coding Guidelines
We document our Coding Guidelines in the [QC-devs guidelines repo](https://github.com/theochem/guidelines/). We particularly suggest you review:

* [Contributing to QC-dev](https://github.com/theochem/guidelines/blob/main/contributing.md)
* [QC-Devs Code of Conduct](https://github.com/theochem/guidelines/blob/main/CodeOfConduct.md)

We also recommend installing pre-commit hooks. That ensure certain basic coding
style issues can be detected and fixed before submitting the pull request.
To set up these hooks, install [https://pre-commit.com/](pre-commit)
(e.g. using `pip install --user pre-commit`) and run `pre-commit install`.
