# Model Hamiltonian

This utility generates 1- and 2-electron integrals corresponding to various model Hamiltonians. The basic input is some indication of connectivity, either explicitly or as a lattice. One then specifies the Hamiltonian of interest. The output are 1- and 2-electron integrals in a format convenient for use in other (external) software packages.


## Installation

```
python3 -m pip install .
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

We document our default QC-Devs guidelines in the [.github repository](https://github.com/theochem/.github/).
We particularly suggest you review:

* [Contributing to QC-Devs](https://github.com/theochem/.github/blob/main/CONTRIBUTING.md)
* [QC-Devs Code of Conduct](https://github.com/theochem/.github/blob/main/CODE_OF_CONDUCT.md)

We also recommend installing pre-commit hooks. That ensure certain basic coding
style issues can be detected and fixed before submitting the pull request.
To set up these hooks, install [https://pre-commit.com/](pre-commit)
(e.g. using `pip install --user pre-commit`) and run `pre-commit install`.
