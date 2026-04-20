
from bounties.cico_verifier import (
    verify_cico_solution,
    CICO_P,
    CICO_ALPHA,
    CICO_K,
    CICO_T,
    CICO_RF,
    CICO_RP,
    CICO_CONSTANTS,
)
from poseidon.poseidon import Poseidon
from poseidon.mds_matrix import generate_mds_matrix

# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------

P          = CICO_P        # 2^31 − 2^24 + 1  (KoalaBear)
ALPHA      = CICO_ALPHA    # 3
K          = CICO_K        # 2  (constrained input / output words)
T          = CICO_T        # 16 (permutation state width)
CONSTANTS  = CICO_CONSTANTS  # [C_1, C_2, C_3, C_4] = [0xc09de4, 0xee6282, 0, 0]
RF         = 6
RP         = 6

# ---------------------------------------------------------------------------
# RF=6, RP=6 CICO solution
# ---------------------------------------------------------------------------

def verify_rf6_rp6_cico() -> bool:
    """
    Verify a known CICO solution for RF=RP=6.

    free_inputs is the free portion (x_3, …, x_16) ∈ F_p^{t-k} = F_p^14
    such that applying the Poseidon permutation to

        s = [C_1, C_2, x_3, …, x_16]
          = [0xc09de4, 0xee6282, x_3, …, x_16]

    yields an output whose first k=2 words equal [C_3, C_4] = [0, 0].

    Returns:
        True if the solution satisfies all conditions (C1–C4), False otherwise.
    """
    free_inputs = [
        0x12fb0ad2, 0x7a0a1f31, 0x159cec1b, 0x2d48221,
        0x2de37f4e, 0x2bf5abd0, 0x170f9438, 0x2ffa162e,
        0x29c61b86, 0x34a0a393, 0x43494183, 0x176f34d8,
        0x17f9a567, 0x2ee36da4,
    ]

    result = verify_cico_solution(
        free_inputs,
        prime=P,
        alpha=ALPHA,
        k=K,
        t=T,
        r_f=RF,
        r_p=RP,
        constants=CONSTANTS,
    )

    # Show the full permutation output for inspection.
    mds = generate_mds_matrix(T, P)
    pos = Poseidon(prime=P, alpha=ALPHA, t=T, r_f=RF, r_p=RP, mds=mds)
    state_in = [int(c) % P for c in CONSTANTS[:K]] + [int(x) % P for x in free_inputs]
    state_out = pos.permutation_plus_linear(state_in)

    print("RF=6, RP=6 CICO verification")
    print(f"  free_inputs  = {[hex(x) for x in free_inputs]}")
    print(f"  state_in[:2] = {[hex(v) for v in state_in[:2]]}  (C_1, C_2)")
    print(f"  state_out[:2]= {[hex(v) for v in state_out[:2]]}  (should be {[hex(c) for c in CONSTANTS[K:2*K]]})")
    print(f"  verify_cico_solution(RF=6, RP=6) -> {result}")
    return result


if __name__ == "__main__":
    ok = verify_rf6_rp6_cico()
    raise SystemExit(0 if ok else 1)
