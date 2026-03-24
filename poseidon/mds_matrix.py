"""
MDS matrix generation and application for Poseidon.

Uses the Cauchy construction:
    MDS[i][j] = modular_inverse(x_i - y_j, prime)
where x_i = i and y_j = t + j.
"""


def generate_mds_matrix(t: int, prime: int) -> list:
    """
    Generate a t×t MDS matrix over GF(prime) using the Cauchy construction.

    The resulting matrix satisfies the MDS (Maximum Distance Separable) property:
    every square sub-matrix is invertible mod prime.

    Args:
        t: State width (matrix dimension).
        prime: Prime field modulus. Must satisfy prime > 2*t.

    Returns:
        A t×t list-of-lists of integers in GF(prime).
    """
    mds = []
    for i in range(t):
        row = []
        for j in range(t):
            # x_i = i, y_j = t + j  →  x_i - y_j = i - t - j  (always != 0 for prime > 2t)
            diff = (i - t - j) % prime
            row.append(pow(diff, -1, prime))
        mds.append(row)
    return mds


def apply_mds(state: list, mds: list, prime: int) -> list:
    """
    Apply an MDS matrix to a state vector over GF(prime).

    Args:
        state: List of t field elements.
        mds: t×t MDS matrix as list-of-lists.
        prime: Prime field modulus.

    Returns:
        New state after multiplying by MDS matrix mod prime.
    """
    t = len(state)
    return [
        sum(mds[i][j] * state[j] for j in range(t)) % prime
        for i in range(t)
    ]
