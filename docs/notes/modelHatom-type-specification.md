# Atom Type Specification

Atom types can be specified in several different ways. The default atom type is always a Carbon 2p-orbital, as would be engaged in a $\pi$-electron Hamiltonian.

## Specified Atom types
The user can (optionally) pass a list of atom types. If the atom types are not specified, all atoms are assumed to be $2p$ orbitals that would appear in a $\pi$ Hamiltonian. That is, all atoms are assumed to be `C`. If a list of atom types is provided, it will be used to specify the model Hamiltonian based on a [dictionary]() of atom types and the bonds between them. If an atom type is specified that is not defined in the dictionary, the program will (gracefully) fail.

Two dictionaries are provided by default. They are an atomic dictionary (one entry for every atomic symbol) and the dictionary that corresponds to Rauk's H&uuml;ckel parameters. [Rauk's dictionary](https://fdocuments.in/document/orbital-interaction-theory-of-organic-chemistry-simple-hueckel-molecular.html) contains entries for: `C, B, N2, N3, O1, O2, F, Si, P2, P3, S1, S2, Cl`. The numbers after `N`, `O`, `P`, and `S` refer to the coordination number. E.g., `N2` is a nitrogen like in pyridine and `N3` is a nitrogen like in pyrrole. Similarly, `O2` is an oxygen like in an ether (e.g. furan) and `O1` is a carbonyl oxygen (e.g. quinone). Similarly, `P3` refers to phosphorous atoms like in phosphole and `S2` refers to sulfur atoms like in thiophene. `P2` refers to a phosphorous like in phosphorine and `S1` refers to a sulfur like thiobenzophenone.

The Rauk parameters are derived from PPP calculations in:
> [F. A. Van-Catledge, *J. Org. Chem.* **45**, 4801-4802 (1980)](https://pubs.acs.org/doi/pdf/10.1021/jo01311a060)
