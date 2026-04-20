
from bounties.zerotest_verifier import (
    verify_zerotest_solution,
    verify_zerotest_solution_relaxed,
    _ext_eval_poly,
    _EXT_BETA,
    ZEROTEST_P,
    ZEROTEST_ALPHA,
    ZEROTEST_R,
    ZEROTEST_D,
    ZEROTEST_ELL,
    ZEROTEST_S,
    ZEROTEST_RF,
    ZEROTEST_RP,
    ZEROTEST_TPERM,
)
from poseidon.poseidon import Poseidon
from poseidon.mds_matrix import generate_mds_matrix

# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------

P       = ZEROTEST_P        # 2^31 − 2^24 + 1  (KoalaBear)
ALPHA   = ZEROTEST_ALPHA    # 3
R       = ZEROTEST_R        # 2  (quadratic extension)
D       = ZEROTEST_D        # 7  (max polynomial degree)
ELL     = ZEROTEST_ELL      # 8  (hash output words in ext-field elements)
S       = ZEROTEST_S        # 1  (univariate)
T_PERM  = ZEROTEST_TPERM    # 16 (state width = ell * r)
RF      = 6
RP      = 6

# ---------------------------------------------------------------------------
# RF=6, RP=6 zero-test solution
# ---------------------------------------------------------------------------

def verify_rf6_rp6_zerotest() -> bool:
    """
    Verify a known zero-test solution for RF=RP=6.

    P_hat is the flat coefficient vector of a degree-≤7 polynomial
    P : F_{p^2} → F_{p^2} such that:

        (C1)  1 ≤ degree(P) ≤ 7
        (C2)  (a_0, …, a_7) := H(P_hat)   via Poseidon1 with RF=RP=6
        (C3)  P(a_0) = 0 ∈ F_{p^2}

    P_hat encodes 8 extension-field coefficients as 16 base-field integers:
        coeff[j] = (P_hat[2j], P_hat[2j+1])  representing P_hat[2j] + P_hat[2j+1]·√3

    Returns:
        True if the solution satisfies all three conditions, False otherwise.
    """
    P_hat = [954437106, 510249971, 2130706432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    result = verify_zerotest_solution(
        P_hat,
        prime=P,
        r=R,
        d=D,
        ell=ELL,
        s=S,
        r_f=RF,
        r_p=RP,
        t_perm=T_PERM,
        alpha=ALPHA,
    )

    # Compute the hash output and evaluate P at a_0 using verifier helpers.
    mds = generate_mds_matrix(T_PERM, P)
    pos = Poseidon(prime=P, alpha=ALPHA, t=T_PERM, r_f=RF, r_p=RP, mds=mds)
    hash_out = pos.compression_mode_hash([int(v) % P for v in P_hat], out_length=ELL * R)
    a0 = (hash_out[0] % P, hash_out[1] % P)
    coeffs = [(int(P_hat[j * R]) % P, int(P_hat[j * R + 1]) % P) for j in range(D + 1)]
    p_at_a0 = _ext_eval_poly(coeffs, a0, P, beta=_EXT_BETA)

    print("RF=6, RP=6 zero-test verification")
    print(f"  P_hat    = {P_hat}")
    print(f"  a_0      = {a0}  (first ext-field hash output word)")
    print(f"  P(a_0)   = {p_at_a0}  (should be (0, 0))")
    print(f"  verify_zerotest_solution(RF=6, RP=6) -> {result}")
    return result


if __name__ == "__main__":
    ok = verify_rf6_rp6_zerotest()
    raise SystemExit(0 if ok else 1)
