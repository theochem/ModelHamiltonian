"""Tests for HilbertSpace."""

import numpy as np
from scipy.special import comb
import time

from moha.old_hamiltonian import HilbertSpace


def test_hilbert():
    """Test moha.hamiltonian.HilbertSpace"""
    t0 = time.perf_counter()
    test = HilbertSpace(n_electrons=3, n_sites=5)
    t1 = time.perf_counter()
    print(test.state_list)



