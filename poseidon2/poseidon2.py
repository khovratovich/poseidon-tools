"""
Poseidon2 permutation and hash function (pure Python).

Reference: https://eprint.iacr.org/2023/323

Poseidon2 improves on Poseidon by using two distinct linear layers:
  - M_E (external): applied in the first and last R_F/2 full rounds.
  - M_I (internal): applied in the R_P partial rounds.

This separation allows more efficient circuit implementations.
"""

from poseidon.grain_lfsr import GrainLFSR


# ---------------------------------------------------------------------------
# External linear layer helpers
# ---------------------------------------------------------------------------

# M4 is the 4×4 circulant-like matrix used as building block for t divisible by 4.
# From Poseidon2 paper (Section 2.3): M4 = [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]]
_M4 = [
    [5, 7, 1, 3],
    [4, 6, 1, 1],
    [1, 3, 5, 7],
    [1, 1, 4, 6],
]


def _mat_mul_m4(chunk: list, prime: int) -> list:
    """Multiply a 4-element vector by M4 mod prime."""
    return [
        sum(_M4[i][j] * chunk[j] for j in range(4)) % prime
        for i in range(4)
    ]


def _external_layer(state: list, prime: int) -> list:
    """
    Apply the Poseidon2 external linear layer M_E to the state.

    - t=2: M_E = [[2,1],[1,1]]
    - t=3: M_E = [[2,1,1],[1,2,1],[1,1,2]]
    - t=4: M_E = M4
    - t divisible by 4 (t>4): apply M4 to each 4-chunk then mix sums
    """
    t = len(state)
    p = prime

    if t == 2:
        x0, x1 = state
        return [(2 * x0 + x1) % p, (x0 + x1) % p]

    if t == 3:
        x0, x1, x2 = state
        s = (x0 + x1 + x2) % p
        return [(x0 + s) % p, (x1 + s) % p, (x2 + s) % p]

    if t == 4:
        return _mat_mul_m4(state, p)

    if t % 4 == 0:
        # Step 1: apply M4 to each 4-element chunk
        chunks = []
        for k in range(t // 4):
            chunks.append(_mat_mul_m4(state[4 * k : 4 * k + 4], p))

        # Step 2: compute per-position sums across all chunks
        sums = [sum(chunks[k][j] for k in range(t // 4)) % p for j in range(4)]

        # Step 3: new_state[4k+j] = chunk[k][j] + sums[j]
        result = []
        for k in range(t // 4):
            for j in range(4):
                result.append((chunks[k][j] + sums[j]) % p)
        return result

    raise ValueError(f"Unsupported state width t={t} for Poseidon2 external layer. "
                     "t must be 2, 3, 4, or a multiple of 4.")


# ---------------------------------------------------------------------------
# Internal linear layer helper
# ---------------------------------------------------------------------------

def _internal_layer(state: list, d_values: list, prime: int) -> list:
    """
    Apply the Poseidon2 internal linear layer M_I.

    M_I = all-ones matrix + diag(d_0-1, d_1-1, ..., d_{t-1}-1).

    Efficient application:
        s = sum(state)
        new_state[i] = s + (d_i - 1) * state[i]
    """
    p = prime
    s = sum(state) % p
    return [(s + (d_values[i] - 1) * state[i]) % p for i in range(len(state))]


# ---------------------------------------------------------------------------
# Poseidon2 class
# ---------------------------------------------------------------------------

class Poseidon2:
    """
    Poseidon2 hash function over a prime field GF(prime).

    Args:
        prime: Prime field modulus.
        alpha: S-box exponent.  Use -1 for the inverse S-box (x → x^{-1}).
        t: State width in field elements.  Must be 2, 3, 4, or a multiple of 4.
        r_f: Number of full rounds (must be even).
        r_p: Number of partial rounds.
        d_values: Diagonal values for the internal matrix M_I.
                  Must be a list of t distinct integers != 1.
                  Defaults to [(-2^i) mod prime for i in 0..t-1].
        rate: Absorb rate in field elements.  Defaults to t-1 (capacity 1).
    """

    def __init__(
        self,
        prime: int,
        alpha: int,
        t: int,
        r_f: int,
        r_p: int,
        d_values: list = None,
        rate: int = None,
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

        # Diagonal values for the internal linear layer
        if d_values is not None:
            if len(d_values) != t:
                raise ValueError(f"d_values must have {t} elements")
            self.d_values = [v % prime for v in d_values]
        else:
            # Default: d_i = (-2^i) mod prime  (distinct, != 1 for typical primes)
            self.d_values = [(-pow(2, i, prime)) % prime for i in range(t)]

        prime_bit_len = prime.bit_length()
        lfsr = GrainLFSR(prime_bit_len, alpha, t, r_f, r_p)

        # Round constants:
        #   - 1 initial set of t constants (added before any rounds)
        #   - R_F/2 full rounds, each with t constants
        #   - R_P partial rounds, each with 1 constant (for first element only)
        #   - R_F/2 full rounds, each with t constants
        # Total elements: t + (r_f * t) + r_p
        self._rc_initial = [lfsr.get_field_element(prime) for _ in range(t)]

        half_f = r_f // 2
        self._rc_full_first = [
            [lfsr.get_field_element(prime) for _ in range(t)]
            for _ in range(half_f)
        ]
        self._rc_partial = [lfsr.get_field_element(prime) for _ in range(r_p)]
        self._rc_full_last = [
            [lfsr.get_field_element(prime) for _ in range(t)]
            for _ in range(half_f)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sbox(self, x: int) -> int:
        if self.alpha == -1:
            return pow(x, -1, self.prime)
        return pow(x, self.alpha, self.prime)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        """
        Apply the Poseidon2 permutation to a state of t field elements.

        Structure:
          1. Add initial round constants.
          2. R_F/2 full rounds  (add t constants, S-box all, M_E).
          3. R_P partial rounds (add 1 constant to state[0], S-box state[0], M_I).
          4. R_F/2 full rounds  (add t constants, S-box all, M_E).

        Args:
            state: List of t integers in GF(prime).

        Returns:
            Permuted state as a list of t integers.
        """
        if len(state) != self.t:
            raise ValueError(f"State must have {self.t} elements, got {len(state)}")

        p = self.prime
        state = list(state)

        # Initial round-constant addition
        state = [(state[i] + self._rc_initial[i]) % p for i in range(self.t)]

        # First half: R_F/2 full rounds
        for rc in self._rc_full_first:
            state = [(state[i] + rc[i]) % p for i in range(self.t)]
            state = [self._sbox(x) for x in state]
            state = _external_layer(state, p)

        # Middle: R_P partial rounds (internal layer)
        for rc0 in self._rc_partial:
            state[0] = (state[0] + rc0) % p
            state[0] = self._sbox(state[0])
            state = _internal_layer(state, self.d_values, p)

        # Last half: R_F/2 full rounds
        for rc in self._rc_full_last:
            state = [(state[i] + rc[i]) % p for i in range(self.t)]
            state = [self._sbox(x) for x in state]
            state = _external_layer(state, p)

        return state

    def hash(self, inputs: list) -> int:
        """
        Hash a list of field elements using the sponge construction.

        Args:
            inputs: List of integers in GF(prime).

        Returns:
            A single field element as the hash digest.
        """
        if not inputs:
            raise ValueError("inputs must be non-empty")

        state = [0] * self.t

        # Absorb phase
        for block_start in range(0, len(inputs), self.rate):
            block = inputs[block_start : block_start + self.rate]
            for i, val in enumerate(block):
                state[i] = (state[i] + val) % self.prime
            state = self.permutation(state)

        # Squeeze: return first element of rate portion
        return state[0]
