# poseidon-tools

Pure-Python implementations of the **Poseidon** and **Poseidon2** cryptographic
hash functions — no SageMath or other heavy dependencies required.

## References

- Poseidon: <https://eprint.iacr.org/2019/458>
- Poseidon2: <https://eprint.iacr.org/2023/323>

---

## Requirements

- Python 3.8+ (uses `pow(a, -1, prime)` for modular inverse)
- No third-party libraries for the core implementation

Running tests requires [pytest](https://pytest.org):

```bash
pip install pytest
```

---

## Installation

Clone the repository and add it to your Python path:

```bash
git clone https://github.com/your-org/poseidon-tools.git
cd poseidon-tools
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

## Usage

### Poseidon

```python
from poseidon import Poseidon

# BN254 field prime
p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Instantiate with standard BN254 Poseidon parameters (t=3, R_F=8, R_P=57, alpha=5)
pos = Poseidon(prime=p, alpha=5, t=3, r_f=8, r_p=57)

# Hash a list of field elements (sponge construction, returns one field element)
digest = pos.hash([1, 2])
print(hex(digest))

# Apply raw permutation to a state of t field elements
state_out = pos.permutation([0, 1, 2])
```

#### Parameters

| Parameter | Description |
|-----------|-------------|
| `prime`   | Prime field modulus |
| `alpha`   | S-box exponent (use `-1` for the inverse S-box x → x⁻¹) |
| `t`       | State width in field elements |
| `r_f`     | Number of full rounds (must be even) |
| `r_p`     | Number of partial rounds |
| `rate`    | Absorb rate (defaults to `t-1`, capacity = 1) |

---

### Poseidon2

```python
from poseidon2 import Poseidon2

p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

pos2 = Poseidon2(prime=p, alpha=5, t=3, r_f=8, r_p=57)

digest = pos2.hash([1, 2])
print(hex(digest))

state_out = pos2.permutation([0, 1, 2])
```

Poseidon2 supports `t ∈ {2, 3, 4}` and any multiple of 4.

#### Additional Parameters

| Parameter  | Description |
|------------|-------------|
| `d_values` | Diagonal values for the internal linear layer M_I (list of `t` distinct integers ≠ 1). Defaults to `[(-2^i) mod prime for i in 0..t-1]`. |

---

## Repository Layout

```
poseidon/
  grain_lfsr.py   – 80-bit Grain LFSR for round-constant generation
  mds_matrix.py   – Cauchy MDS matrix construction and application
  poseidon.py     – Poseidon permutation and sponge hash
poseidon2/
  poseidon2.py    – Poseidon2 permutation and sponge hash
tests/
  test_poseidon.py
  test_poseidon2.py
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 58 tests should pass.
