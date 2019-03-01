""" Lattice Classes for Model Hamiltonian Systems """
import numpy as np


class Lattice:
    """ Lattice containing LatticeSites"""

    def __init__(self, sites=[]):
        """ Initialize Lattice instance

            Parameters
            ----------
            sites : list of LatticeSite
                Contains all of the sites on the lattice
        """

        self.sites = sites
        self.Nsites = len(self.sites)
        self.adjacency_matrix = None

    def add_site(self, coord):
        """ Generate a new LatticeSite and add to lattice

            Parameters
            ----------
            coord : np.array, shape(3,)
                The Cartesian coordinates of the new site
        """

        self.sites.append(LatticeSite(self.Nsites, coord))
        self.Nsites += 1

    def add_sites(self, *args):
        """ Mass generate new LatticeSites and add to lattice

            Parameters
            ----------
            *args : list of np.array, shape (3,)
                List of Cartesian coordinates of each new site
        """
        for coord in args:
            self.add_site(coord)

    def gen_adjacency_matrix(self):
        """ Generate the adjacency matrix for the LatticeSites

            Produces a square (n x n) matrix where n is the number of LatticeSites,
            where M_i,j = 1 denotes that sites i and j are neighbours
        """
        adjacency = np.zeros((len(self.sites), len(self.sites)), dtype=bool)
        for site in self.sites:
            for neighbour in site.neighbours:
                adjacency[site.number, neighbour] = 1
        self.adjacency_matrix = adjacency

    def add_bond(self, site1, site2):
        """ Define two LatticeSites as neighbours

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
        """ Produce a 1-D linear Lattice:

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
    """ Lattice site on a Lattice """

    def __init__(self, number, coord):
        """ Initialize LatticeSite instance

            Parameters
            ----------
            number : int
                The site number (should be same as ONvector index for the site)

            coord : np.array
                Cartesian coordinates of the lattice site

            neighbours : list of int
                List of site numbers of neighbouring sites (connected by bonds)
        """

        self.number = number
        self.coord = coord
        self.neighbours = []
