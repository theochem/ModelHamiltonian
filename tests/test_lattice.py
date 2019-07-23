"""Test lattice module."""

import numpy as np
import pytest

from moha.lattice import Lattice, LatticeSite


def test_lattice_site():
    test = LatticeSite(1, np.array([0., 0., 0.]))
    assert test.number == 1
    assert np.allclose(test.coords, np.array([0., 0., 0.]))
    for i in [2, 4, 5]:
        test.neighbours.append(i)
    assert len(test.neighbours) == 3
    assert test.neighbours == [2, 4, 5]

    with pytest.raises(TypeError):
        bad_index = LatticeSite(1.1, np.array([0., 0., 0.]))
    with pytest.raises(TypeError):
        bad_coords = LatticeSite(1, [3, 4, 5])
    with pytest.raises(ValueError):
        too_many_coords = LatticeSite(1, np.array([0., 0., 0., 0.]))





