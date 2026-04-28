import pytest
import numpy as np
from moha.hamiltonians import AlternativeSpinHamiltonian

def test_alternative_spin_initialization():
    """Verify that the adapter correctly maps inputs to J matrices."""
    H = AlternativeSpinHamiltonian(num_spins=3, coupling_constants=1.0)

    assert H.num_spins == 3
    assert H.J_eq.shape == (3, 3)
    assert H.J_ax.shape == (3, 3)
    # Check that adjacent sites are coupled
    assert H.J_eq[0, 1] == 1.0
    assert H.J_eq[1, 0] == 1.0

def test_alternative_spin_integral_generation():
    """Verify the fermion mapping generates the correct tensor shapes."""
    H = AlternativeSpinHamiltonian(num_spins=3, coupling_constants=1.0)

    one_body = H.generate_one_body_integral(dense=True)
    two_body = H.generate_two_body_integral(dense=True)

    # 3 sites means 6 spin-orbitals (3 alpha, 3 beta)
    assert one_body.shape == (6, 6)
    assert two_body.shape == (6, 6, 6, 6)