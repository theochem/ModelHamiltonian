"""Tests for HilbertSpace."""

import numpy as np

from moha.hamiltonian import HilbertSpace


def test_hilbert():
    test = HilbertSpace(2, 3)
    [print(f"{x:06b}") for x in test.states]



