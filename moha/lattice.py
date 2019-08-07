"""Lattice Classes for Model Hamiltonian Systems."""

from itertools import cycle
import numpy as np


class Lattice:
    """Lattice containing LatticeSites.

    Attributes
    ----------
    sites : list of LatticeSite
        The sites comprising the lattice.
    n_sites : int
        The number of lattice sites in the lattice, K.
    adjacency_matrix : np.ndarray(K, K)
        A boolean matrix specifying the neighbours for each site, with M_i,j = 1 denoting that
        sites i and j are neighbours.

    Methods
    -------
    add_site(coords, atom_type)
        Generate a new LatticeSite and add to the lattice.
    add_sites(*args)
        Generate many LatticeSites at once and add to the lattice.

    Notes
    -----
    Most porcelain functions will generate the adjacency matrix prior to returning a class
    instance, but using the Lattice class outside of those class methods will require calling
    `gen_adjacency_matrix` after adding all LatticeSites to the Lattice.

    """

    def __init__(self, sites=list()):
        """Initialize Lattice instance

        Parameters
        ----------
        sites : list of LatticeSite
            The sites comprising the lattice.

        """

        self.sites = sites
        self.n_sites = len(self.sites)
        self.adjacency_matrix = None

    def add_site(self, coords, atom_type=""):
        """Generate a new LatticeSite and add to lattice.

        Parameters
        ----------
        coords : np.ndarray(3,)
            The Cartesian coordinates of the new site.
        atom_type : str
            The atom type of the site.

        """

        self.sites.append(LatticeSite(self.n_sites, coords, atom_type))
        self.n_sites += 1

    def add_sites(self, sites):
        """Mass generate new LatticeSites and add to lattice

        Parameters
        ----------
        sites : list of tuple
            List of Cartesian coordinates and atom_types of each new site.
            Each tuple must consist of a np.ndarray(3,) in first position and str in second pos'n.

        """
        for coords, atom_type in sites:
            self.add_site(coords=coords, atom_type=atom_type)

    def gen_adjacency_matrix(self):
        """Generate the adjacency matrix for the LatticeSites.

        Produces a square (K x K) matrix where K is the number of LatticeSites,
        where M_i,j = 1 denotes that sites i and j are neighbours.

        Notes
        -----
        Most porcelain functions will call this function prior to returning a class instance, but
        using the Lattice class outside a method will require calling this function after adding
        all LatticeSites to the Lattice.
        """
        adjacency = np.zeros((len(self.sites), len(self.sites)), dtype=bool)
        for site in self.sites:
            for neighbour in site.neighbours:
                adjacency[site.number, neighbour] = 1
        self.adjacency_matrix = adjacency

    def add_bond(self, site1, site2):
        """Define two LatticeSites as neighbours.

        Parameters
        ----------
        site1 : int
            The site number of the first LatticeSite.
        site2 : int
            The site number of the second LatticeSite.
        """
        self.sites[site1].neighbours.append(site2)
        self.sites[site2].neighbours.append(site1)

    @classmethod
    def linear(cls, n_sites, dist=1.0, axis=0, atom_types=[""]):
        """Produce a 1-D linear Lattice of evenly-spaced sites.

        Parameters
        ----------
        n_sites : int
            The number of lattice sites, K.
        dist : float
            The distance between each lattice site.
        axis : {0, 1, 2}
            The Cartesian axis along which the line proceeds.
            0 corresponds to x, 1 to y, 2 to z.
        atom_types : list of str
            The atom type of each lattice site.
            When given a list with length less than the number of sites, the pattern is repeated
            in order, i.e. ["C", "O"] will yield a lattice with even sites of type "C" and odd
            sites of type "O".
        # TODO: add pbc
        pbc : bool
            Connect endpoints for periodic boundary conditions?

        Raises
        ------
        ValueError if `atom_types` is longer than `n_sites`.

        """
        if len(atom_types) > n_sites:
            raise ValueError("Too many atom types specified")

        coords = np.zeros((n_sites, 3))
        coords[:, axis] = np.linspace(0, dist * (n_sites - 1), n_sites)
        atom_types = cycle(atom_types)
        sites = [LatticeSite(number=site, coords=coords, atom_type=atom) for (site, coords), atom in zip(enumerate(coords), atom_types)]
        lat = cls(sites=sites)
        [lat.add_bond(i, i + 1) for i in range(len(sites) - 1)]
        """
        if pbc:
            if n_sites > 2:
                lat.add_bond(0, len(sites)-1)
        """
        lat.gen_adjacency_matrix()
        return lat

    @classmethod
    def rectangular(cls, n_sites, dist=(1.0, 1.0), axis=(0, 1), atom_types=[""]):
        """Produce a 2-D rectangular Lattice of evenly-spaced sites.

        Generate 2-dimensional primitive Bravais lattices of the following types:
        Orthorhombic (rectangular): D2 point group
        Tetragonal (square): D4 point group

        Parameters
        ----------
        n_sites : tuple of int
            The number of lattice sites for each axis.
        dist : tuple of float
            The distance between each lattice site for each axis.
        axis : tuple of int
            The Cartesian axes along which the lattice is generated.
            0 corresponds to x, 1 to y, 2 to z.
        atom_types : list of str
            The atom type of each lattice site.
            When given a list with length less than the number of sites, the pattern is repeated
            in order, i.e. ["C", "O"] will yield a lattice with even sites of type "C" and odd
            sites of type "O".

        Raises
        ------
        ValueError
            If the same axis is chosen multiple times for the lattice.
            If `atom_types` is longer than `n_sites`.

        """
        if axis[0] == axis[1]:
            raise ValueError("Each lattice axis must be different")
        if len(atom_types) > np.product(n_sites):
            raise ValueError("Too many atom types specified")

        coords = np.zeros((n_sites[0] * n_sites[1], 3))
        coords[:, axis[0]] = np.tile(
            np.linspace(0, dist[0] * (n_sites[0] - 1), n_sites[0]), n_sites[1]
        )
        coords[:, axis[1]] = np.repeat(
            np.linspace(0, dist[1] * (n_sites[1] - 1), n_sites[1]), n_sites[0]
        )
        atom_types = cycle(atom_types)
        sites = [LatticeSite(number=site, coords=coords, atom_type=atom) for (site, coords), atom in zip(enumerate(coords), atom_types)]
        lat = cls(sites=sites)
        # Add neighbours along axis 0
        [
            lat.add_bond(n_sites[0] * b + a, n_sites[0] * b + a + 1)
            for a in range(n_sites[0] - 1)
            for b in range(n_sites[1])
        ]
        # Add neighbours along axis 1
        [
            lat.add_bond(n_sites[0] * b + a, n_sites[0] * (b + 1) + a)
            for a in range(n_sites[0])
            for b in range(n_sites[1] - 1)
        ]
        lat.gen_adjacency_matrix()
        return lat

    @classmethod
    def oblique(cls, n_sites, dist=(1.0, 1.0), axis=(0, 1), angle=np.pi/4, atom_types=[""]):
        """Produce a 2-D oblique Lattice of evenly-spaced sites.

        Generate 2-dimensional primitive Bravais lattices of the following types:
        Monoclinic (oblique): C2 point group

        Parameters
        ----------
        n_sites : tuple of int
            The number of lattice sites for each axis.
        dist : tuple of float
            The distance between each lattice site for each axis.
        axis : tuple of int
            The Cartesian axes along which the lattice is generated.
            0 corresponds to x, 1 to y, 2 to z.
        angle : float
            The skew angle of the lattice in radians.
        atom_types : list of str
            The atom type of each lattice site.
            When given a list with length less than the number of sites, the pattern is repeated
            in order, i.e. ["C", "O"] will yield a lattice with even sites of type "C" and odd
            sites of type "O".

        Raises
        ------
        ValueError
            If the same axis is chosen multiple times for the lattice.
            If `atom_types` is longer than `n_sites`.

        """
        # Redirect to rectangular if no angle
        if angle == 0:
            return cls.rectangular(n_sites, dist, axis)
        if axis[0] == axis[1]:
            raise ValueError("Each lattice axis must be different")
        if len(atom_types) > np.product(n_sites):
            raise ValueError("Too many atom types specified")

        coords = np.zeros((n_sites[0] * n_sites[1], 3))
        coords[:, axis[0]] = np.tile(
            np.linspace(0, dist[0] * (n_sites[0] - 1), n_sites[0]), n_sites[1]
        ) + np.repeat(np.arange(n_sites[1]), n_sites[0]) * dist[0] * np.cos(angle)
        coords[:, axis[1]] = np.repeat(np.arange(n_sites[1]), n_sites[0]) * dist[1] * np.sin(angle)
        atom_types = cycle(atom_types)
        sites = [LatticeSite(number=site, coords=coords, atom_type=atom) for (site, coords), atom in zip(enumerate(coords), atom_types)]
        lat = cls(sites=sites)
        # Add neighbours along axis 0
        [
            lat.add_bond(n_sites[0] * b + a, n_sites[0] * b + a + 1)
            for a in range(n_sites[0] - 1)
            for b in range(n_sites[1])
        ]
        # Add neighbours along axis 1
        [
            lat.add_bond(n_sites[0] * b + a, n_sites[0] * (b + 1) + a)
            for a in range(n_sites[0])
            for b in range(n_sites[1] - 1)
        ]
        lat.gen_adjacency_matrix()
        return lat

    @classmethod
    def orthorhombic(cls, n_sites, dist=(1.0, 1.0, 1.0), atom_types=[""]):
        """Produce a 3-D orthorhombic Lattice of evenly-spaced sites.

        Generate 3-dimensional primitive Bravais lattices of the following types:
        Orthorhombic: D2h point group
        Tetragonal: D4h point group
        Cubic: Oh point group

        All inputs are ordered x, y, z.

        Parameters
        ----------
        n_sites : tuple of int
            The number of lattice sites for each axis.
        dist : tuple of float
            The distance between each lattice site for each axis.
        atom_types : list of str
            The atom type of each lattice site.
            When given a list with length less than the number of sites, the pattern is repeated
            in order, i.e. ["C", "O"] will yield a lattice with even sites of type "C" and odd
            sites of type "O".

        Raises
        ------
        ValueError
            If `atom_types` is longer than `n_sites`.

        """
        if len(atom_types) > np.product(n_sites):
            raise ValueError("Too many atom types specified")

        coords = np.zeros((n_sites[0] * n_sites[1] * n_sites[2], 3))
        coords[:, 0] = np.tile(np.linspace(0, dist[0] * (n_sites[0] - 1), n_sites[0]), n_sites[1] * n_sites[2])
        coords[:, 1] = np.tile(np.repeat(np.linspace(0, dist[1] * (n_sites[1] - 1), n_sites[1]), n_sites[0]), n_sites[2])
        coords[:, 2] = np.repeat(
            np.linspace(0, dist[2] * (n_sites[2] - 1), n_sites[2]), n_sites[0] * n_sites[1]
        )
        atom_types = cycle(atom_types)
        sites = [LatticeSite(number=site, coords=coords, atom_type=atom) for (site, coords), atom in zip(enumerate(coords), atom_types)]
        lat = cls(sites=sites)
        # Add neighbours along x axis
        [
            lat.add_bond(n_sites[0] * n_sites[1] * c + n_sites[0] * b + a, n_sites[0] * n_sites[1] * c + n_sites[0] * b + a + 1)
            for a in range(n_sites[0] - 1)
            for b in range(n_sites[1])
            for c in range(n_sites[2])
        ]
        # Add neighbours along y axis
        [
            lat.add_bond(n_sites[0] * n_sites[1] * c + n_sites[0] * b + a, n_sites[0] * n_sites[1] * c + n_sites[0] * (b + 1) + a)
            for a in range(n_sites[0])
            for b in range(n_sites[1] - 1)
            for c in range(n_sites[2])
        ]
        # Add neighbours along z axis
        [
            lat.add_bond(n_sites[0] * n_sites[1] * c + n_sites[0] * b + a, n_sites[0] * n_sites[1] * (c + 1) + n_sites[0] * b + a)
            for a in range(n_sites[0])
            for b in range(n_sites[1])
            for c in range(n_sites[2] - 1)
        ]
        lat.gen_adjacency_matrix()
        return lat


class LatticeSite:
    """Lattice site on a Lattice.

    Attributes
    ----------
    number : int
        The site number (should be same as ONvector index for the site).
    coords : np.ndarray(3,)
        Cartesian coordinates of the lattice site.
    atom_type : str
        The atom type of the site.
        # TODO: determine if this will just be atom name or atom w/ hybridization
    neighbours : list of int
        List of site numbers of neighbouring sites (connected by bonds).

    """

    def __init__(self, number, coords, atom_type=""):
        """Initialize LatticeSite instance

        Parameters
        ----------
        number : int
            The site number (should be same as ONvector index for the site).
        coords : np.ndarray(3,)
            Cartesian coordinates of the lattice site.
        atom_type : str
            The atom type of the site.

        Raises
        ------
        TypeError
            If `number` is not an integer.
            If `coords` is not of type np.ndarray.
        ValueError
            If `coords` does not have shape (3,).

        """
        if not isinstance(number, int):
            raise TypeError(
                "Site number must be an integer corresponding to the site's index in"
                "the ON vector"
            )
        if not isinstance(coords, np.ndarray):
            raise TypeError("Site coordinates must be a numpy ndarray")
        if coords.shape != (3,):
            raise ValueError("Site coordinates must have shape (3,)")

        self.number = number
        self.coords = coords
        self.atom_type = atom_type
        self.neighbours = []
