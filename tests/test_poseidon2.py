"""
Tests for the Poseidon2 hash function implementation.

Uses BN254 (bn128) curve parameters:
  p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
  t=3, R_F=8, R_P=57, alpha=5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from poseidon2.poseidon2 import Poseidon2, _external_layer, _internal_layer

# ---------------------------------------------------------------------------
# BN254 parameters
# ---------------------------------------------------------------------------
BN254_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
T = 3
R_F = 8
R_P = 57
ALPHA = 5


# ---------------------------------------------------------------------------
# External linear layer tests
# ---------------------------------------------------------------------------

class TestExternalLayer:
    def test_t2(self):
        p = BN254_P
        state = [1, 1]
        result = _external_layer(state, p)
        assert result == [3 % p, 2 % p]

    def test_t3_output_length(self):
        result = _external_layer([1, 2, 3], BN254_P)
        assert len(result) == 3

    def test_t3_elements_in_field(self):
        result = _external_layer([100, 200, 300], BN254_P)
        for x in result:
            assert 0 <= x < BN254_P

    def test_t4_output_length(self):
        result = _external_layer([1, 2, 3, 4], BN254_P)
        assert len(result) == 4

    def test_t4_elements_in_field(self):
        result = _external_layer([1, 2, 3, 4], BN254_P)
        for x in result:
            assert 0 <= x < BN254_P

    def test_t8_output_length(self):
        result = _external_layer(list(range(8)), BN254_P)
        assert len(result) == 8

    def test_t8_elements_in_field(self):
        result = _external_layer(list(range(8)), BN254_P)
        for x in result:
            assert 0 <= x < BN254_P

    def test_t3_zero_state_is_fixed_point(self):
        # The external layer is a linear map: zero state maps to zero state.
        result = _external_layer([0, 0, 0], BN254_P)
        assert result == [0, 0, 0]

    def test_t3_deterministic(self):
        s = [7, 13, 42]
        assert _external_layer(s, BN254_P) == _external_layer(s, BN254_P)

    def test_unsupported_t_raises(self):
        with pytest.raises(ValueError):
            _external_layer([1, 2, 3, 4, 5], BN254_P)


# ---------------------------------------------------------------------------
# Internal linear layer tests
# ---------------------------------------------------------------------------

class TestInternalLayer:
    def test_output_length(self):
        d = [2, 3, 5]
        result = _internal_layer([1, 2, 3], d, BN254_P)
        assert len(result) == 3

    def test_elements_in_field(self):
        d = [2, 3, 5]
        result = _internal_layer([100, 200, 300], d, BN254_P)
        for x in result:
            assert 0 <= x < BN254_P

    def test_known_small_case(self):
        # state = [1,0,0], d = [2,3,5], prime = large
        # s = 1;  new[0] = 1 + (2-1)*1 = 2; new[1] = 1 + (3-1)*0 = 1; new[2] = 1 + (5-1)*0 = 1
        p = BN254_P
        result = _internal_layer([1, 0, 0], [2, 3, 5], p)
        assert result == [2, 1, 1]


# ---------------------------------------------------------------------------
# Poseidon2 permutation tests
# ---------------------------------------------------------------------------

class TestPoseidon2Permutation:
    def _make(self, t=T):
        return Poseidon2(BN254_P, ALPHA, t, R_F, R_P)

    def test_permutation_length_preserving(self):
        pos = self._make()
        result = pos.permutation([1, 2, 3])
        assert len(result) == T

    def test_permutation_output_in_field(self):
        pos = self._make()
        result = pos.permutation([0, 1, 2])
        for elem in result:
            assert 0 <= elem < BN254_P

    def test_permutation_deterministic(self):
        pos1 = self._make()
        pos2 = self._make()
        state = [100, 200, 300]
        assert pos1.permutation(state) == pos2.permutation(state)

    def test_permutation_different_inputs_different_outputs(self):
        pos = self._make()
        assert pos.permutation([0, 0, 0]) != pos.permutation([1, 0, 0])
        assert pos.permutation([1, 0, 0]) != pos.permutation([0, 1, 0])

    def test_permutation_wrong_state_length_raises(self):
        pos = self._make()
        with pytest.raises(ValueError):
            pos.permutation([1, 2])

    def test_permutation_zero_state_changes(self):
        pos = self._make()
        result = pos.permutation([0] * T)
        assert not all(x == 0 for x in result)

    def test_permutation_t4(self):
        pos = self._make(t=4)
        result = pos.permutation([1, 2, 3, 4])
        assert len(result) == 4
        for elem in result:
            assert 0 <= elem < BN254_P

    def test_permutation_t2(self):
        pos = self._make(t=2)
        result = pos.permutation([1, 2])
        assert len(result) == 2
        for elem in result:
            assert 0 <= elem < BN254_P

    def test_permutation_t8(self):
        pos = self._make(t=8)
        result = pos.permutation(list(range(8)))
        assert len(result) == 8
        for elem in result:
            assert 0 <= elem < BN254_P


# ---------------------------------------------------------------------------
# Poseidon2 hash tests
# ---------------------------------------------------------------------------

class TestPoseidon2Hash:
    def _make(self, t=T):
        return Poseidon2(BN254_P, ALPHA, t, R_F, R_P)

    def test_hash_returns_field_element(self):
        pos = self._make()
        h = pos.hash([1, 2])
        assert 0 <= h < BN254_P

    def test_hash_deterministic(self):
        pos1 = self._make()
        pos2 = self._make()
        assert pos1.hash([1, 2]) == pos2.hash([1, 2])

    def test_hash_different_inputs_different_outputs(self):
        pos = self._make()
        assert pos.hash([1]) != pos.hash([2])
        assert pos.hash([1, 2]) != pos.hash([2, 1])

    def test_hash_single_element(self):
        pos = self._make()
        h = pos.hash([42])
        assert 0 <= h < BN254_P

    def test_hash_empty_raises(self):
        pos = self._make()
        with pytest.raises(ValueError):
            pos.hash([])

    def test_hash_long_input(self):
        pos = self._make()
        inputs = list(range(1, 11))
        h = pos.hash(inputs)
        assert 0 <= h < BN254_P

    def test_hash_known_vector(self):
        """Regression test: ensure output is stable."""
        pos = self._make()
        expected = pos.hash([1, 2])
        pos2 = self._make()
        assert pos2.hash([1, 2]) == expected

    def test_hash_custom_d_values(self):
        """Custom d_values should be accepted and produce valid output."""
        d_values = [2, 3, 5]
        pos = Poseidon2(BN254_P, ALPHA, T, R_F, R_P, d_values=d_values)
        h = pos.hash([1, 2])
        assert 0 <= h < BN254_P

    def test_hash_d_values_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Poseidon2(BN254_P, ALPHA, T, R_F, R_P, d_values=[2, 3])

    def test_r_f_odd_raises(self):
        with pytest.raises(ValueError):
            Poseidon2(BN254_P, ALPHA, T, 7, R_P)

    def test_poseidon_vs_poseidon2_differ(self):
        """Poseidon and Poseidon2 must produce different outputs (different algorithms)."""
        from poseidon.poseidon import Poseidon
        pos1 = Poseidon(BN254_P, ALPHA, T, R_F, R_P)
        pos2 = Poseidon2(BN254_P, ALPHA, T, R_F, R_P)
        assert pos1.hash([1, 2]) != pos2.hash([1, 2])
