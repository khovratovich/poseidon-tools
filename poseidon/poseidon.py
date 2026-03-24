"""
Poseidon permutation and hash function (pure Python).

Reference: https://eprint.iacr.org/2019/458

Example (KoalaBear field, t=16, R_F=8, R_P=20):
    >>> from poseidon.poseidon import Poseidon
    >>> KOALABEAR_P = 2130706433  # 2^31 - 2^24 + 1
    >>> pos = Poseidon(prime=KOALABEAR_P, alpha=3, t=16, r_f=8, r_p=20)
    >>> pos.hash(list(range(15)))
    93555670
    >>> pos.hash(list(range(1, 16)))
    938201807
    >>> pos.hash([1])
    1541345887
    >>> pos.permutation([0] * 16)[0]
    1393439926
"""

from .grain_lfsr import GrainLFSR
from .mds_matrix import generate_mds_matrix, apply_mds


class Poseidon:
    """
    Poseidon hash function over a prime field GF(prime).

    Uses the sponge construction with the Poseidon permutation as the underlying
    primitive.  Parameters (t, R_F, R_P, alpha) must match a valid Poseidon
    instance; no automatic security-level selection is performed here.

    Args:
        prime: Prime field modulus.
        alpha: S-box exponent.  Use -1 for the inverse S-box (x → x^{-1}).
        t: State width in field elements.
        r_f: Number of full rounds (must be even).
        r_p: Number of partial rounds.
        rate: Absorb rate in field elements.  Defaults to t-1 (capacity 1).
        mds: Optional custom t×t MDS matrix as list-of-lists.  When omitted
            the Cauchy MDS construction is used.
        round_constants: Optional flat list of (r_f + r_p)*t pre-computed
            round constants.  When omitted, constants are derived from the
            Grain LFSR.
    """

    def __init__(
        self,
        prime: int,
        alpha: int,
        t: int,
        r_f: int,
        r_p: int,
        rate: int = None,
        mds: list = None,
        round_constants: list = None,
    ):
        if r_f % 2 != 0:
            raise ValueError("r_f must be even")
        self.prime = prime
        self.alpha = alpha
        self.t = t
        self.r_f = r_f
        self.r_p = r_p
        self.rate = rate if rate is not None else t - 1
        self.capacity = t - self.rate

        total_rounds = r_f + r_p

        if round_constants is not None:
            # Accept a flat list of (r_f + r_p)*t constants and reshape.
            expected = total_rounds * t
            if len(round_constants) != expected:
                raise ValueError(
                    f"round_constants must have {expected} elements, got {len(round_constants)}"
                )
            self.round_constants = [
                list(round_constants[i * t : (i + 1) * t])
                for i in range(total_rounds)
            ]
        else:
            prime_bit_len = prime.bit_length()
            lfsr = GrainLFSR(prime_bit_len, alpha, t, r_f, r_p)
            self.round_constants = [
                [lfsr.get_field_element(prime) for _ in range(t)]
                for _ in range(total_rounds)
            ]

        self.mds = mds if mds is not None else generate_mds_matrix(t, prime)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sbox(self, x: int) -> int:
        """Apply the S-box: x^alpha mod prime (or x^{-1} if alpha == -1)."""
        if self.alpha == -1:
            return pow(x, -1, self.prime)
        return pow(x, self.alpha, self.prime)

    def _add_round_constants(self, state: list, constants: list) -> list:
        p = self.prime
        return [(state[i] + constants[i]) % p for i in range(self.t)]

    def _full_round(self, state: list, constants: list) -> list:
        state = self._add_round_constants(state, constants)
        state = [self._sbox(x) for x in state]
        state = apply_mds(state, self.mds, self.prime)
        return state

    def _partial_round(self, state: list, constants: list) -> list:
        state = self._add_round_constants(state, constants)
        state[0] = self._sbox(state[0])
        state = apply_mds(state, self.mds, self.prime)
        return state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        """
        Apply the Poseidon permutation to a state of t field elements.

        Args:
            state: List of t integers in GF(prime).

        Returns:
            Permuted state as a list of t integers.
        """
        if len(state) != self.t:
            raise ValueError(f"State must have {self.t} elements, got {len(state)}")

        state = list(state)
        half_f = self.r_f // 2
        rc_idx = 0

        # First half: R_F/2 full rounds
        for _ in range(half_f):
            state = self._full_round(state, self.round_constants[rc_idx])
            rc_idx += 1

        # Middle: R_P partial rounds
        for _ in range(self.r_p):
            state = self._partial_round(state, self.round_constants[rc_idx])
            rc_idx += 1

        # Last half: R_F/2 full rounds
        for _ in range(half_f):
            state = self._full_round(state, self.round_constants[rc_idx])
            rc_idx += 1

        return state

    def hash(self, inputs: list) -> int:
        """
        Hash a list of field elements using the sponge construction.

        Inputs are absorbed rate elements at a time; after all inputs are
        absorbed the first element of the rate portion is returned as the
        digest.

        Args:
            inputs: List of integers in GF(prime).

        Returns:
            A single field element (integer) as the hash digest.
        """
        if not inputs:
            raise ValueError("inputs must be non-empty")

        # Initialise state to all-zero
        state = [0] * self.t

        # Absorb phase: process inputs in blocks of `rate`
        for block_start in range(0, len(inputs), self.rate):
            block = inputs[block_start : block_start + self.rate]
            for i, val in enumerate(block):
                state[i] = (state[i] + val) % self.prime
            state = self.permutation(state)

        # Squeeze: return first element of rate portion
        return state[0]
