"""Tests for reading FCIDUMP files."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from moha.molkit.hamiltonians import MolHam
from moha.molkit.utils.tools import load_fcidump

FCIDUMP = """ &FCI NORB=   2,NELEC= 2,MS2=0,
  ORBSYM=1,1,
  ISYM=1,
 &END
  0.6746 1 1 1 1
  0.1809 1 1 2 2
  0.6636 2 2 2 2
  0.1809 2 1 2 1
 -1.2526 1 1 0 0
 -0.4759 2 2 0 0
  0.7137 0 0 0 0
"""


def test_load_fcidump_integrals():
    """Chemists' integrals from the file end up in physicists' notation."""
    data = load_fcidump(iter(FCIDUMP.splitlines(keepends=True)))
    assert data["nelec"] == 2
    assert data["spinpol"] == 0
    assert_allclose(data["core_energy"], 0.7137)
    assert_allclose(data["one_ints"], np.diag([-1.2526, -0.4759]))
    two = data["two_ints"]
    # (11|22) in chemists' notation is <12|12> in physicists' notation.
    assert_allclose(two[0, 1, 0, 1], 0.1809)
    assert_allclose(two[1, 0, 1, 0], 0.1809)
    # (21|21) in chemists' notation is <22|11> in physicists' notation.
    assert_allclose(two[1, 1, 0, 0], 0.1809)
    assert_allclose(two[0, 0, 1, 1], 0.1809)
    assert_allclose(two[0, 0, 0, 0], 0.6746)
    assert_allclose(two[1, 1, 1, 1], 0.6636)


def test_molham_from_fcidump(tmp_path):
    """MolHam.from_fcidump reads the file and sets the dimensions."""
    path = tmp_path / "FCIDUMP"
    path.write_text(FCIDUMP)
    ham = MolHam()
    ham.from_fcidump(str(path))
    assert ham.n_spatial == 2
    assert ham.n_spin == 4
    assert_allclose(ham.zero_body, 0.7137)


def test_load_fcidump_wrong_header():
    """A wrong header gives a ValueError that says what is wrong."""
    lines = ["&FCI NORB=2,NELEC=2,MS2=0,\n", " &END\n"]
    with pytest.raises(ValueError, match="Incorrect file header"):
        load_fcidump(iter(lines))


def test_load_fcidump_wrong_number_of_fields():
    """A data line with the wrong number of fields gives a ValueError."""
    lines = [" &FCI NORB=2,NELEC=2,MS2=0,\n", " &END\n", " 1.0 1 1 0\n"]
    with pytest.raises(ValueError, match="Expecting 5 fields"):
        load_fcidump(iter(lines))
