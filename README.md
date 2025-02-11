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
python3 -m pip install .[gpt]
```
- For the GUI subversion:

```
python3 -m pip install .[gui]
```

- For the TOML subversion:

```
python3 -m pip install .[toml]
```

## Coding Guidelines

We document our default QC-Devs guidelines in the [.github repository](https://github.com/theochem/.github/).
We particularly suggest you review:

* [Contributing to QC-Devs](https://github.com/theochem/.github/blob/main/CONTRIBUTING.md)
* [QC-Devs Code of Conduct](https://github.com/theochem/.github/blob/main/CODE_OF_CONDUCT.md)

We also recommend installing pre-commit hooks. That ensure certain basic coding
style issues can be detected and fixed before submitting the pull request.
To set up these hooks, install [https://pre-commit.com/](https://pre-commit.com)
(e.g. using `pip install --user pre-commit`) and run `pre-commit install`.

## Citation

If you use this code, please cite the following [publication](https://doi.org/10.1063/5.0219015):

```bibtex
@article{10.1063/5.0219015,
    author = {Chuiko, Valerii and Richards, Addison D. S. and Sánchez-Díaz, Gabriela and Martínez-González, Marco and Sanchez, Wesley and B. Da Rosa, Giovanni and Richer, Michelle and Zhao, Yilin and Adams, William and Johnson, Paul A. and Heidar-Zadeh, Farnaz and Ayers, Paul W.},
    title = {ModelHamiltonian: A Python-scriptable library for generating 0-, 1-, and 2-electron integrals},
    journal = {The Journal of Chemical Physics},
    volume = {161},
    number = {13},
    pages = {132503},
    year = {2024},
    month = {10},
    issn = {0021-9606},
    doi = {10.1063/5.0219015},
    url = {https://doi.org/10.1063/5.0219015},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0219015/20195032/132503\_1\_5.0219015.pdf},
}
```
