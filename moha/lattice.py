"""Lattice Classes for Model Hamiltonian Systems."""
import numpy as np


class Lattice:
    """Lattice containing LatticeSites."""

    def __init__(self, sites=list()):
        """Initialize Lattice instance

        Parameters
        ----------
        sites : list of LatticeSite
            Contains all of the sites on the lattice

        """

        self.sites = sites
        self.n_sites = len(self.sites)
        self.adjacency_matrix = None

    def add_site(self, coords):
        """Generate a new LatticeSite and add to lattice.

        Parameters
        ----------
        coords : np.ndarray(3,)
            The Cartesian coordinates of the new site

        """

        self.sites.append(LatticeSite(self.n_sites, coords))
        self.n_sites += 1

    def add_sites(self, *args):
        """Mass generate new LatticeSites and add to lattice

        Parameters
        ----------
        *args : list of np.ndarray(3,)
            List of Cartesian coordinates of each new site

        """
        for coords in args:
            self.add_site(coords)

    def gen_adjacency_matrix(self):
        """Generate the adjacency matrix for the LatticeSites.

        Produces a square (n x n) matrix where n is the number of LatticeSites,
        where M_i,j = 1 denotes that sites i and j are neighbours.
        """
        adjacency = np.zeros((len(self.sites), len(self.sites)), dtype=bool)
        for site in self.sites:
            for neighbour in site.neighbours:
                adjacency[site.number, neighbour] = 1
        self.adjacency_matrix = adjacency

    def add_bond(self, site1, site2):
        """Define two LatticeSites as neighbours

        Parameters
        ----------
        site1 : int
            The site number of the first LatticeSite

        site2 : int
            The site number of the second LatticeSite
        """
        self.sites[site1].neighbours.append(site2)
        self.sites[site2].neighbours.append(site1)

    @classmethod
    def linear(cls, nodes, vec=np.array([1., 0., 0.]), pbc=True):
        """Produce a 1-D linear Lattice.

        Parameters
        ----------
        nodes : int
            The number of sites to add to the lattice

        vec : np.array
            The vector defining the distance between each node

        pbc : bool
            Connect endpoints for periodic boundary conditions?
        """
        sites = [LatticeSite(n, n*vec) for n in range(nodes)]
        lat = cls(sites=sites)
        [lat.add_bond(i, i+1) for i in range(len(sites)-1)]
        if pbc:
            if nodes > 2:
                lat.add_bond(0, len(sites)-1)
        return lat


class LatticeSite:
    """Lattice site on a Lattice.

    Attributes
    ----------
    number : int
        The site number (should be same as ONvector index for the site)

    coords : np.ndarray(3,)
        Cartesian coordinates of the lattice site

    neighbours : list of int
        List of site numbers of neighbouring sites (connected by bonds)

    """

    def __init__(self, number, coords):
        """Initialize LatticeSite instance

        Parameters
        ----------
        number : int
            The site number (should be same as ONvector index for the site)

        coords : np.ndarray(3,)
            Cartesian coordinates of the lattice site

        """
        if not isinstance(number, int):
            raise TypeError("Site number must be an integer corresponding to the site's index in"
                            "the ON vector")
        if not isinstance(coords, np.ndarray):
            raise TypeError("Site coordinates must be a numpy ndarray with shape (3,)")

        self.number = number
        self.coords = coords
        self.neighbours = []
