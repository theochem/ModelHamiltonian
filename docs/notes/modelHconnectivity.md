# Specifying Connectivity in a Model Hamiltonian

Connectivity can be specified in several different ways. If multiple methods are specified, then only the first in the below list is used. (The specifications become decreasingly informative as one moves down this file.)

## One-Electron Hamiltonian matrix `h[:,:]`
If the one-electron Hamiltonian is defined, then $h_{pq}$ is used to define connectivity.
$$
h_{pq} =
\begin{cases}
h_{pq} \ne 0 \qquad &p\text{ and }q\text{ are connected} \\
0 \qquad &p\text{ and }q\text{ are not connected}
\end{cases}
$$

## Distance matrix, `distance[:,:]`
Specify a symmetric, positive-definite, upper-diagonal (with zero diagonal) matrix with distances between atoms in the system. The elements of the matrix are
$$
d_{pq} =
\begin{cases}
\text{distance between sites }p\text{ and }q \qquad &p\text{ and }q\text{ are connected} \\
0 \qquad &p\text{ and }q\text{ are not connected}
\end{cases}
$$

## Lattice specifications
**TODO** Explain this.

## Adjacency matrix, `adjacency[:,:]`
Specify a symmetric, integer-valued matrix with nonzero values for for bonded sites and `0` values for nonbonded sites.
$$
a_{pq} =
\begin{cases}
\a_{pq} \ne 0 \qquad &p\text{ and }q\text{ are connected} \\
\0 \qquad &p\text{ and }q\text{ are not connected}
\end{cases}
$$
Note that both $h_{pq}$ and $d_{pq}$ can be converted to valid adjacency matrices by replacing the nonzero values with 1.

## Connectivity
We support the format of connectivity information in [`Gaussian`](https://gaussian.com/geom/). In this case, for each atom, one specifies the atoms it is bonded to according to:
`Atom-index   Atom-index-bonded-to    bond-type    Atom-index-bonded-to    bond-type ...`
All bonds are assumed to be symmetric, so if one only lists atoms that come before the present atom in the list, that suffices. At this stage, all nonzero bond orders are treated equivalent, as `True` entries in the adjacency matrix. If two different bond-types are listed for the same atom pair, the program will fail (gracefully). The bond-type is an integer that is ordinarily interpreted as a bond order. Note that the information defined in the connectivity list can be used to directly specify the adjacency matrix.
