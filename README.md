# Model Hamiltonian

This utility generates 1- and 2-electron integrals corresponding to various model Hamiltonians. The basic input is some indication of connectivity, either explicitly or as a lattice. One then specifies the Hamiltonian of interest. The output are 1- and 2-electron integrals in a format convenient for use in other (external) software packages.


## Installation

```
python3 -m pip install -e .
```

### Subversions of the ModelHamiltonian
To install a specific subversion of the ModelHamiltonian, you can use the following command:
- For the GPT subversion:

```
python3 -m pip install -e .[gpt]
```
- For the GUI subversion:

```
python3 -m pip install -e .[gui]
```

- For the TOML subversion:

```
python3 -m pip install -e .[toml]
```

## Coding Guidelines
We document our Coding Guidelines in the [QC-devs guidelines repo](https://github.com/theochem/guidelines/). We particularly suggest you review:

* [Contributing to QC-dev](https://github.com/theochem/guidelines/blob/main/contributing.md)
* [QC-Devs Code of Conduct](https://github.com/theochem/guidelines/blob/main/CodeOfConduct.md)

We also recommend installing pre-commit hooks. That ensure certain basic coding
style issues can be detected and fixed before submitting the pull request.
To set up these hooks, install [https://pre-commit.com/](pre-commit)
(e.g. using `pip install --user pre-commit`) and run `pre-commit install`.