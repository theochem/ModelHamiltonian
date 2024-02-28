# Extended Hubbard Model Hamiltonian

## The extended Hubbard Model
The Hamiltonian is a two-electron site-interaction model, with one orbital per site. It is equivalent to the H&uuml;ckel model, with the addition on-site and off-site repulsion
terms that tend to prevent multiple electrons from occupying the same site ($U > 0$ and $V>0$).
$$
\hat{H}_{\text{Hubbard}} = \sum_{pq} h_{pq} a_p^\dagger a_q + \sum_p U_p \hat{n}_{p\alpha}\hat{n}_{p\beta} + \frac{1}{2}\sum_{p\ne q} V_{pq} (\hat{n}_{p \alpha} + \hat{n}_{p \beta})(\hat{n}_{q \alpha} + \hat{n}_{q \beta})
$$
Traditionally there are three parameters:
- an $\alpha_{pp}$ parameter for the diagonal terms $h_{pp}$, representing the energy associated with an electron on the site $p$. This is usually zero in extended Hubbard models.
- a $t_{pq}=t_{qp}$ parameter for the off-diagonal terms $h_{p \ne q}$, this is the resonance/hopping/bond term between sites $p$ and $q$.
- a $U_p$ parameter that is the onsite interaction (generally repulsion).
- a $V_{pq}$ parameter that represents the effective interaction between electrons on adjacent sites.

> Note: Traditionally Hubbard models work by assuming that orbitals on different sites do not overlap. Hubbard Hamiltonians are always restricted but we do support the imposition of an external magnetic field.

## Input Formats
### API
- Connectivity can be provided in the usual way, as a lattice, an adjacency matrix, a distance matrix, or explicit connectivity specification.
- Parameters can be specified as constants (all sites equal) or as dictionaries.

- Proposed Calling Sequence:
```
modelh.extHubbard(connectivity, alpha=0.0, t=1.0, U=1.0, V=.2, atom_types=None, atom_dictionary=None, bond_dictionary=None, Bz = 0.0)

    """Compute the 1- and 2-electron integrals associated with the Hubbard Hamiltonian.

    Parameters
    ----------
    connectivity
        an object specifying molecular connectivity
    alpha
        If alpha is a float, it specifies the site energy if all sites are equivalent.
    t
        Specifies the resonance/hopping term if all bonds are equivalent.
    U
        Specifies the on-site interaction; usually repulsive.
    V
        Specifies the off-site interaction; usually repulsive but smaller than U.
    atom_types
        A list of dimension equal to the number of sites specifying the atom type of each site.
        If a list of atom types is specified, the values of alpha and beta are ignored.
    atom_dictionary
        Contains information about alpha and U values for each atom type.
    bond_dictionary
        Contains information about beta values and V for each bond type.
    Bz
        External magnetic field in atomic units.

    Returns
    -------
    integrals_1el
        One-electron integrals.
    integrals_2el
        Two-electron integrals.
    """
```