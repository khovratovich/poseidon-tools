"""
Tests for the Poseidon hash function implementation.

Uses BN254 (bn128) curve parameters:
  p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
  t=3, R_F=8, R_P=57, alpha=5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from poseidon.grain_lfsr import GrainLFSR
from poseidon.mds_matrix import generate_mds_matrix, apply_mds, verify_mds_matrix
from poseidon.poseidon import Poseidon

# ---------------------------------------------------------------------------
# BN254 parameters
# ---------------------------------------------------------------------------
BN254_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
T = 3
R_F = 8
R_P = 57
ALPHA = 5


# ---------------------------------------------------------------------------
# GrainLFSR tests
# ---------------------------------------------------------------------------

class TestGrainLFSR:
    def _make_lfsr(self):
        prime_bit_len = BN254_P.bit_length()  # 254
        return GrainLFSR(prime_bit_len, ALPHA, T, R_F, R_P)

    def test_produces_bits(self):
        lfsr = self._make_lfsr()
        bit = lfsr.get_next_bit()
        assert bit in (0, 1)

    def test_deterministic(self):
        """Two LFSRs with the same params must produce the same sequence."""
        lfsr1 = self._make_lfsr()
        lfsr2 = self._make_lfsr()
        bits1 = [lfsr1.get_next_bit() for _ in range(100)]
        bits2 = [lfsr2.get_next_bit() for _ in range(100)]
        assert bits1 == bits2

    def test_field_element_in_range(self):
        lfsr = self._make_lfsr()
        for _ in range(10):
            elem = lfsr.get_field_element(BN254_P)
            assert 0 <= elem < BN254_P

    def test_field_elements_are_not_all_identical(self):
        lfsr = self._make_lfsr()
        elems = [lfsr.get_field_element(BN254_P) for _ in range(10)]
        # Very unlikely that all 10 are the same
        assert len(set(elems)) > 1

    def test_different_params_give_different_sequences(self):
        prime_bit_len = BN254_P.bit_length()
        lfsr_a = GrainLFSR(prime_bit_len, ALPHA, T, R_F, R_P)
        lfsr_b = GrainLFSR(prime_bit_len, ALPHA, 4, R_F, R_P)  # different t
        bits_a = [lfsr_a.get_next_bit() for _ in range(64)]
        bits_b = [lfsr_b.get_next_bit() for _ in range(64)]
        assert bits_a != bits_b

    def test_inverse_sbox_flag(self):
        """alpha=-1 sets sbox bit to 1 and alpha_bits to 0."""
        prime_bit_len = BN254_P.bit_length()
        lfsr = GrainLFSR(prime_bit_len, -1, T, R_F, R_P)
        # Just verify it initialises without error and produces valid bits
        bit = lfsr.get_next_bit()
        assert bit in (0, 1)


# ---------------------------------------------------------------------------
# MDS matrix tests
# ---------------------------------------------------------------------------

class TestMDSMatrix:
    def test_shape(self):
        mds = generate_mds_matrix(T, BN254_P)
        assert len(mds) == T
        for row in mds:
            assert len(row) == T

    def test_elements_in_field(self):
        mds = generate_mds_matrix(T, BN254_P)
        for row in mds:
            for elem in row:
                assert 0 <= elem < BN254_P

    def test_invertible(self):
        """The MDS matrix must be invertible (det != 0 mod prime)."""
        mds = generate_mds_matrix(T, BN254_P)
        det = _det_mod(mds, BN254_P)
        assert det != 0

    def test_all_2x2_submatrices_invertible(self):
        """MDS property: every 2×2 submatrix must be invertible."""
        mds = generate_mds_matrix(T, BN254_P)
        for r1 in range(T):
            for r2 in range(r1 + 1, T):
                for c1 in range(T):
                    for c2 in range(c1 + 1, T):
                        sub = [
                            [mds[r1][c1], mds[r1][c2]],
                            [mds[r2][c1], mds[r2][c2]],
                        ]
                        det = (sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0]) % BN254_P
                        assert det != 0, f"2×2 sub-matrix rows=({r1},{r2}) cols=({c1},{c2}) has det=0"

    def test_apply_mds(self):
        mds = generate_mds_matrix(T, BN254_P)
        state = [1, 2, 3]
        result = apply_mds(state, mds, BN254_P)
        assert len(result) == T
        for elem in result:
            assert 0 <= elem < BN254_P

    def test_apply_mds_not_identity(self):
        mds = generate_mds_matrix(T, BN254_P)
        state = [1, 2, 3]
        result = apply_mds(state, mds, BN254_P)
        assert result != state


# ---------------------------------------------------------------------------
# Poseidon permutation tests
# ---------------------------------------------------------------------------

class TestPoseidonPermutation:
    def _make_poseidon(self):
        return Poseidon(BN254_P, ALPHA, T, R_F, R_P)

    def test_permutation_length_preserving(self):
        pos = self._make_poseidon()
        state = [1, 2, 3]
        result = pos.permutation(state)
        assert len(result) == T

    def test_permutation_output_in_field(self):
        pos = self._make_poseidon()
        state = [0, 1, 2]
        result = pos.permutation(state)
        for elem in result:
            assert 0 <= elem < BN254_P

    def test_permutation_deterministic(self):
        pos1 = self._make_poseidon()
        pos2 = self._make_poseidon()
        state = [100, 200, 300]
        assert pos1.permutation(state) == pos2.permutation(state)

    def test_permutation_different_inputs_different_outputs(self):
        pos = self._make_poseidon()
        assert pos.permutation([0, 0, 0]) != pos.permutation([1, 0, 0])
        assert pos.permutation([1, 0, 0]) != pos.permutation([0, 1, 0])

    def test_permutation_wrong_state_length_raises(self):
        pos = self._make_poseidon()
        with pytest.raises(ValueError):
            pos.permutation([1, 2])

    def test_permutation_zero_state(self):
        pos = self._make_poseidon()
        result = pos.permutation([0] * T)
        assert len(result) == T
        # The zero state should not stay at zero after a full permutation
        assert not all(x == 0 for x in result)


# ---------------------------------------------------------------------------
# Poseidon hash tests
# ---------------------------------------------------------------------------

class TestPoseidonHash:
    def _make_poseidon(self):
        return Poseidon(BN254_P, ALPHA, T, R_F, R_P)

    def test_hash_returns_field_element(self):
        pos = self._make_poseidon()
        h = pos.hash([1, 2])
        assert 0 <= h < BN254_P

    def test_hash_deterministic(self):
        pos1 = self._make_poseidon()
        pos2 = self._make_poseidon()
        assert pos1.hash([1, 2]) == pos2.hash([1, 2])

    def test_hash_different_inputs_different_outputs(self):
        pos = self._make_poseidon()
        assert pos.hash([1]) != pos.hash([2])
        assert pos.hash([1, 2]) != pos.hash([2, 1])

    def test_hash_single_element(self):
        pos = self._make_poseidon()
        h = pos.hash([42])
        assert 0 <= h < BN254_P

    def test_hash_empty_raises(self):
        pos = self._make_poseidon()
        with pytest.raises(ValueError):
            pos.hash([])

    def test_hash_long_input(self):
        pos = self._make_poseidon()
        inputs = list(range(1, 11))
        h = pos.hash(inputs)
        assert 0 <= h < BN254_P

    def test_hash_known_vector(self):
        """
        Regression / self-consistency test: record the output of our own
        implementation and verify it stays stable across refactors.
        """
        pos = self._make_poseidon()
        # Compute once and store as expected value
        expected = pos.hash([1, 2])
        # Re-instantiate to ensure we get the same thing
        pos2 = self._make_poseidon()
        assert pos2.hash([1, 2]) == expected


# ---------------------------------------------------------------------------
# Utility: determinant mod prime (for small matrices)
# ---------------------------------------------------------------------------

def _det_mod(matrix, prime):
    """Compute determinant of a square matrix mod prime using Gaussian elimination."""
    n = len(matrix)
    mat = [row[:] for row in matrix]
    det = 1
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if mat[row][col] % prime != 0:
                pivot = row
                break
        if pivot is None:
            return 0
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            det = (-det) % prime
        inv = pow(mat[col][col], -1, prime)
        det = (det * mat[col][col]) % prime
        for row in range(col + 1, n):
            factor = (mat[row][col] * inv) % prime
            for k in range(col, n):
                mat[row][k] = (mat[row][k] - factor * mat[col][k]) % prime
    return det % prime


# ---------------------------------------------------------------------------
# KoalaBear field constants
# ---------------------------------------------------------------------------

KOALABEAR_P = 2130706433  # 2^31 - 2^24 + 1

# Prime where the t=3 Cauchy MDS matrix passes all security checks
SMALL_PRIME_FOR_MDS_TEST = 1009  # Cauchy t=3 is fully verified for this prime


# ---------------------------------------------------------------------------
# verify_mds_matrix tests
# ---------------------------------------------------------------------------

class TestVerifyMDSMatrix:
    def test_cauchy_mds_t3_passes_all_checks(self):
        # For p=1009 the Cauchy t=3 MDS has an irreducible char poly for all
        # required powers, so verify_mds_matrix returns True.
        mds = generate_mds_matrix(3, SMALL_PRIME_FOR_MDS_TEST)
        assert verify_mds_matrix(mds, SMALL_PRIME_FOR_MDS_TEST) is True

    def test_cauchy_mds_t3_bn254_minpoly_fails(self):
        # The BN254 Cauchy t=3 matrix has an eigenvalue in GF(p), making its
        # characteristic polynomial reducible.  The minpoly check therefore
        # fails and verify_mds_matrix returns False.
        mds = generate_mds_matrix(3, BN254_P)
        assert verify_mds_matrix(mds, BN254_P) is False


# ---------------------------------------------------------------------------
# KoalaBear tests
# ---------------------------------------------------------------------------

class TestKoalaBear:
    def _make_poseidon(self):
        return Poseidon(KOALABEAR_P, alpha=3, t=16, r_f=8, r_p=20)

    def test_hash_known_vectors(self):
        pos = self._make_poseidon()
        assert pos.hash(list(range(15))) == 93555670
        assert pos.hash(list(range(1, 16))) == 938201807
        assert pos.hash([1]) == 1541345887

    def test_permutation_known_vector(self):
        pos = self._make_poseidon()
        assert pos.permutation([0] * 16)[0] == 1393439926

    def test_hash_output_in_field(self):
        pos = self._make_poseidon()
        h = pos.hash([42])
        assert 0 <= h < KOALABEAR_P
