import numpy as np
from moha.utils import antisymmetrize_two_electron_integrals

def test_antisymmetrization():
    """Test if antisymmetrization enforces the Pauli exclusion principle correctly."""
    np.random.seed(42)  # For reproducibility
    n_orb = 4  # Test with 4 orbitals

    # Initialize a symmetric two-electron integral tensor
    eri = np.random.rand(n_orb, n_orb, n_orb, n_orb)

    # Enforce symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)
    eri = 0.5 * (eri + eri.transpose(1, 0, 2, 3))  # Swap first two indices
    eri = 0.5 * (eri + eri.transpose(0, 1, 3, 2))  # Swap last two indices
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))  # Swap full pairs

    # Call the antisymmetrization function
    eri_antisym = antisymmetrize_two_electron_integrals(eri)

    # Verify antisymmetry: (pq|rs) - (qp|rs) - (pq|sr) + (qp|sr) = 0
    antisymmetric_check = eri_antisym - eri_antisym.transpose(1, 0, 2, 3) - \
                          eri_antisym.transpose(0, 1, 3, 2) + \
                          eri_antisym.transpose(1, 0, 3, 2)

    assert np.allclose(antisymmetric_check, 0), "Antisymmetrization failed!"
    


def test_hermitian_symmetry():
    """Test if enforcing Hermitian symmetry works correctly."""
    np.random.seed(42)
    n_orb = 4
    eri = np.random.rand(n_orb, n_orb, n_orb, n_orb) + 1j * np.random.rand(n_orb, n_orb, n_orb, n_orb)

    eri_hermitian = antisymmetrize_two_electron_integrals(eri, enforce_hermitian=True)

    # Check Hermitian condition: H[pq|rs] = H[rs|pq]*
    assert np.allclose(eri_hermitian, eri_hermitian.transpose(2, 3, 0, 1).conj()), "Hermitian symmetry failed"

def test_spin_symmetry():
    """Test if spin symmetry conservation works."""
    np.random.seed(42)
    n_orb = 4
    n_spin_orb = 2 * n_orb  # Doubled for spin orbitals

    eri = np.random.rand(n_spin_orb, n_spin_orb, n_spin_orb, n_spin_orb)
    eri_spin = antisymmetrize_two_electron_integrals(eri, enforce_spin_symmetry_func=None, n_spin_orbitals=n_spin_orb)

    # Check spin symmetry: only same-spin interactions should be unchanged
    for p in range(n_spin_orb):
        for q in range(n_spin_orb):
            for r in range(n_spin_orb):
                for s in range(n_spin_orb):
                    if (p % 2 == q % 2) and (r % 2 == s % 2):
                        assert np.allclose(eri_spin[p, q, r, s], eri_spin[q, p, s, r]), "Spin symmetry violated"


def test_permutational_symmetry():
    """Test if permutational symmetry is correctly enforced."""
    np.random.seed(42)
    n_orb = 4
    eri = np.random.rand(n_orb, n_orb, n_orb, n_orb)

    # Ensure input ERI is symmetric before applying antisymmetrization
    eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2))  # Symmetrize pq|rs <-> qp|sr
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))  # Symmetrize pq|rs <-> rs|pq
    eri = 0.5 * (eri + eri.transpose(3, 2, 1, 0))  # Symmetrize pq|rs <-> sr|qp

    # Apply antisymmetrization
    eri_perm = antisymmetrize_two_electron_integrals(eri, enforce_permutational_symmetry=True)

    # Compute absolute differences
    diff1 = eri_perm - eri_perm.transpose(1, 0, 3, 2)  # (pq|rs) vs (qp|sr)
    diff2 = eri_perm - eri_perm.transpose(2, 3, 0, 1)  # (pq|rs) vs (rs|pq)

    max_diff1 = np.max(np.abs(diff1))
    max_diff2 = np.max(np.abs(diff2))

    # Set an appropriate tolerance (adjust if necessary)
    atol = 1e-8

    # If test fails, print debug info
    if max_diff1 > atol or max_diff2 > atol:
        print("\n=== Debugging ERI Permutations ===")
        print("Original ERI:\n", eri)
        print("Symmetrized ERI:\n", eri_perm)
        print("Diff (pq|rs) vs (qp|sr):\n", diff1)
        print("Diff (pq|rs) vs (rs|pq):\n", diff2)
        print("\nMax Diff (pq|rs) vs (qp|sr):", max_diff1)
        print("Max Diff (pq|rs) vs (rs|pq):", max_diff2)

    # Assertions with atol to prevent precision issues
    assert np.allclose(eri_perm, eri_perm.transpose(1, 0, 3, 2), atol=atol), "Permutational symmetry failed: (pq|rs) != (qp|sr)"
    assert np.allclose(eri_perm, eri_perm.transpose(2, 3, 0, 1), atol=atol), "Permutational symmetry failed: (pq|rs) != (rs|pq)"

def test_time_reversal_symmetry():
    """Test if time-reversal symmetry is enforced."""
    np.random.seed(42)
    n_orb = 4
    eri = np.random.rand(n_orb, n_orb, n_orb, n_orb) + 1j * np.random.rand(n_orb, n_orb, n_orb, n_orb)

    eri_trs = antisymmetrize_two_electron_integrals(eri, enforce_trs=True)

    # Check time-reversal symmetry condition
    assert np.allclose(eri_trs, eri_trs.transpose(3, 2, 1, 0).conj()), "Time-reversal symmetry failed"
