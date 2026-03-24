# Remark: This script contains functionality for GF(2^n), but currently works only over GF(p)! A few small adaptations are needed for GF(2^n).
from sage.rings.polynomial.polynomial_gf2x import GF2X_BuildIrred_list
from math import *
import itertools

###########################################################################
#p = 2**31 -1
global p,n,t,F, grain_gen, alpha, NUM_CELLS



FIELD=1
SBOX=0
def get_alpha(p):
    for alpha in range(3, p):
        if gcd(alpha, p-1) == 1:
            break
    return alpha


def get_sbox_cost(R_F, R_P, N, t):
    return int(t * R_F + R_P)

def get_size_cost(R_F, R_P, N, t):
    n = ceil(float(N) / t)
    return int((N * R_F) + (n * R_P))

def poseidon_calc_final_numbers_fixed(p, t, alpha, M, security_margin):
    # [Min. S-boxes] Find best possible for t and N
    n = ceil(log(p, 2))
    N = int(n * t)
    cost_function = get_sbox_cost
    ret_list = []
    (R_F, R_P) = find_FD_round_numbers(p, t, alpha, M, cost_function, security_margin)
    min_sbox_cost = cost_function(R_F, R_P, N, t)
    ret_list.append(R_F)
    ret_list.append(R_P)
    ret_list.append(min_sbox_cost)

    # [Min. Size] Find best possible for t and N
    # Minimum number of S-boxes for fixed n results in minimum size also (round numbers are the same)!
    min_size_cost = get_size_cost(R_F, R_P, N, t)
    ret_list.append(min_size_cost)

    return ret_list # [R_F, R_P, min_sbox_cost, min_size_cost]



def find_FD_round_numbers(p, t, alpha, M, cost_function, security_margin):
    n = ceil(log(p, 2))
    N = int(n * t)

    sat_inequiv = sat_inequiv_alpha

    R_P = 0
    R_F = 0
    min_cost = float("inf")
    max_cost_rf = 0
    # Brute-force approach
    for R_P_t in range(1, 500):
        for R_F_t in range(4, 100):
            if R_F_t % 2 == 0:
                if (sat_inequiv(p, t, R_F_t, R_P_t, alpha, M) == True):
                    if security_margin == True:
                        R_F_t += 2
                        R_P_t = int(ceil(float(R_P_t) * 1.075))
                    cost = cost_function(R_F_t, R_P_t, N, t)
                    if (cost < min_cost) or ((cost == min_cost) and (R_F_t < max_cost_rf)):
                        R_P = ceil(R_P_t)
                        R_F = ceil(R_F_t)
                        min_cost = cost
                        max_cost_rf = R_F
    return (int(R_F), int(R_P))

def sat_inequiv_alpha(p, t, R_F, R_P, alpha, M):
    N = int(log(p, 2) * t)

    if alpha > 0:
        R_F_1 = 6 if M <= ((floor(log(p, 2) - ((alpha-1)/2.0))) * (t + 1)) else 10 # Statistical
        R_F_2 = 1 + ceil(log(2, alpha) * min(M, log(p, 2))) + ceil(log(t, alpha)) - R_P # Interpolation
        R_F_3 = (log(2, alpha) * min(M, log(p, 2))) - R_P # Groebner 1
        R_F_4 = t - 1 + log(2, alpha) * min(M / float(t + 1), log(p, 2) / float(2)) - R_P # Groebner 2
        R_F_5 = (t - 2 + (M / float(2 * log(alpha, 2))) - R_P) / float(t - 1) # Groebner 3
        R_F_max = max(ceil(R_F_1), ceil(R_F_2), ceil(R_F_3), ceil(R_F_4), ceil(R_F_5))

        # Addition due to https://eprint.iacr.org/2023/537.pdf
        r_temp = floor(t / 3.0)
        over = (R_F - 1) * t + R_P + r_temp + r_temp * (R_F / 2.0) + R_P + alpha
        under = r_temp * (R_F / 2.0) + R_P + alpha
        binom_log = log(binomial(over, under), 2)
        if binom_log == inf:
            binom_log = M + 1
        cost_gb4 = ceil(2 * binom_log) # Paper uses 2.3727, we are more conservative here

        return ((R_F >= R_F_max) and (cost_gb4 >= M))
    else:
        print("Invalid value for alpha!")
        exit(1)



# For STARK TODO
# r_p_mod = R_P_FIXED % NUM_CELLS
# if r_p_mod != 0:
#     R_P_FIXED = R_P_FIXED + NUM_CELLS - r_p_mod

###########################################################################

INIT_SEQUENCE = []

#PRIME_NUMBER = p
# if FIELD == 1 and len(sys.argv) != 8:
#     print("Please specify a prime number (in hex format)!")
#     exit()
# elif FIELD == 1 and len(sys.argv) == 8:
#     PRIME_NUMBER = int(sys.argv[7], 16) # e.g. 0xa7, 0xFFFFFFFFFFFFFEFF, 0xa1a42c3efd6dbfe08daa6041b36322ef

#F = GF(PRIME_NUMBER)

def grain_sr_generator():
    bit_sequence = INIT_SEQUENCE
    for _ in range(0, 160):
        new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
        bit_sequence.pop(0)
        bit_sequence.append(new_bit)

    while True:
        new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
        bit_sequence.pop(0)
        bit_sequence.append(new_bit)
        while new_bit == 0:
            new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
            bit_sequence.pop(0)
            bit_sequence.append(new_bit)
            new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
            bit_sequence.pop(0)
            bit_sequence.append(new_bit)
        new_bit = bit_sequence[62] ^^ bit_sequence[51] ^^ bit_sequence[38] ^^ bit_sequence[23] ^^ bit_sequence[13] ^^ bit_sequence[0]
        bit_sequence.pop(0)
        bit_sequence.append(new_bit)
        yield new_bit
grain_gen = grain_sr_generator()

def grain_random_bits(num_bits):
    random_bits = [next(grain_gen) for i in range(0, num_bits)]
    # random_bits.reverse() ## Remove comment to start from least significant bit
    random_int = int("".join(str(i) for i in random_bits), 2)
    return random_int

def init_generator(field, sbox, n, t, R_F, R_P):
     # Generate initial sequence based on parameters
    bit_list_field = [_ for _ in (bin(FIELD)[2:].zfill(2))]
    bit_list_sbox = [_ for _ in (bin(SBOX)[2:].zfill(4))]
    bit_list_n = [_ for _ in (bin(FIELD_SIZE)[2:].zfill(12))]
    bit_list_t = [_ for _ in (bin(NUM_CELLS)[2:].zfill(12))]
    bit_list_R_F = [_ for _ in (bin(R_F)[2:].zfill(10))]
    bit_list_R_P = [_ for _ in (bin(R_P)[2:].zfill(10))]
    bit_list_1 = [1] * 30
    global INIT_SEQUENCE
    INIT_SEQUENCE = bit_list_field + bit_list_sbox + bit_list_n + bit_list_t + bit_list_R_F + bit_list_R_P + bit_list_1
    INIT_SEQUENCE = [int(_) for _ in INIT_SEQUENCE]
    #global grain_gen
    #grain_gen = grain_sr_generator()


def generate_constants(field, n, t, R_F, R_P, prime_number):
    #print(field,n,t,R_F,R_P,prime_number)
    round_constants = []
    # num_constants = (R_F + R_P) * t # Poseidon
    num_constants = (R_F * t) + R_P # Poseidon2

    if field == 0:
        for i in range(0, num_constants):
            random_int = grain_random_bits(n)
            round_constants.append(random_int)
    elif field == 1:
        for i in range(0, num_constants):
            random_int = grain_random_bits(n)
            while random_int >= prime_number:
                # print("[Info] Round constant is not in prime field! Taking next one.")
                random_int = grain_random_bits(n)
            round_constants.append(random_int)
            # Add (t-1) zeroes for Poseidon2 if partial round
            if i >= ((R_F/2) * t) and i < (((R_F/2) * t) + R_P):
                round_constants.extend([0] * (t-1))
    return round_constants

def print_round_constants(round_constants, n, field):
    print("Number of round constants:", len(round_constants))

    if field == 0:
        print("Round constants for GF(2^n):")
    elif field == 1:
        print("Round constants for GF(p):")
    hex_length = int(ceil(float(n) / 4)) + 2 # +2 for "0x"
    print(["{0:#0{1}x}".format(entry, hex_length) for entry in round_constants])

def create_mds_p(n, t):
    M = matrix(F, t, t)

    # Sample random distinct indices and assign to xs and ys
    while True:
        flag = True
        rand_list = [F(grain_random_bits(n)) for _ in range(0, 2*t)]
        while len(rand_list) != len(set(rand_list)): # Check for duplicates
            rand_list = [F(grain_random_bits(n)) for _ in range(0, 2*t)]
        xs = rand_list[:t]
        ys = rand_list[t:]
        # xs = [F(ele) for ele in range(0, t)]
        # ys = [F(ele) for ele in range(t, 2*t)]
        for i in range(0, t):
            for j in range(0, t):
                if (flag == False) or ((xs[i] + ys[j]) == 0):
                    flag = False
                else:
                    entry = (xs[i] + ys[j])^(-1)
                    M[i, j] = entry
        if flag == False:
            continue
        return M

def generate_vectorspace(round_num, M, M_round, NUM_CELLS):
    t = NUM_CELLS
    s = 1
    V = VectorSpace(F, t)
    if round_num == 0:
        return V
    elif round_num == 1:
        return V.subspace(V.basis()[s:])
    else:
        mat_temp = matrix(F)
        for i in range(0, round_num-1):
            add_rows = []
            for j in range(0, s):
                add_rows.append(M_round[i].rows()[j][s:])
            mat_temp = matrix(mat_temp.rows() + add_rows)
        r_k = mat_temp.right_kernel()
        extended_basis_vectors = []
        for vec in r_k.basis():
            extended_basis_vectors.append(vector([0]*s + list(vec)))
        S = V.subspace(extended_basis_vectors)

        return S

def subspace_times_matrix(subspace, M, NUM_CELLS):
    t = NUM_CELLS
    V = VectorSpace(F, t)
    subspace_basis = subspace.basis()
    new_basis = []
    for vec in subspace_basis:
        new_basis.append(M * vec)
    new_subspace = V.subspace(new_basis)
    return new_subspace

# Returns True if the matrix is considered secure, False otherwise
def algorithm_1(M, NUM_CELLS):
    t = NUM_CELLS
    s = 1
    r = floor((t - s) / float(s))

    # Generate round matrices
    M_round = []
    for j in range(0, t+1):
        M_round.append(M^(j+1))

    for i in range(1, r+1):
        mat_test = M^i
        entry = mat_test[0, 0]
        mat_target = matrix.circulant(vector([entry] + ([F(0)] * (t-1))))

        if (mat_test - mat_target) == matrix.circulant(vector([F(0)] * (t))):
            return [False, 1]

        S = generate_vectorspace(i, M, M_round, t)
        V = VectorSpace(F, t)

        basis_vectors= []
        for eigenspace in mat_test.eigenspaces_right(format='galois'):
            if (eigenspace[0] not in F):
                continue
            vector_subspace = eigenspace[1]
            intersection = S.intersection(vector_subspace)
            basis_vectors += intersection.basis()
        IS = V.subspace(basis_vectors)

        if IS.dimension() >= 1 and IS != V:
            return [False, 2]
        for j in range(1, i+1):
            S_mat_mul = subspace_times_matrix(S, M^j, t)
            if S == S_mat_mul:
                print("S.basis():\n", S.basis())
                return [False, 3]

    return [True, 0]

# Returns True if the matrix is considered secure, False otherwise
def algorithm_2(M, NUM_CELLS):
    t = NUM_CELLS
    s = 1

    V = VectorSpace(F, t)
    trail = [None, None]
    test_next = False
    I = range(0, s)
    I_powerset = list(sage.misc.misc.powerset(I))[1:]
    for I_s in I_powerset:
        test_next = False
        new_basis = []
        for l in I_s:
            new_basis.append(V.basis()[l])
        IS = V.subspace(new_basis)
        for i in range(s, t):
            new_basis.append(V.basis()[i])
        full_iota_space = V.subspace(new_basis)
        for l in I_s:
            v = V.basis()[l]
            while True:
                delta = IS.dimension()
                v = M * v
                IS = V.subspace(IS.basis() + [v])
                if IS.dimension() == t or IS.intersection(full_iota_space) != IS:
                    test_next = True
                    break
                if IS.dimension() <= delta:
                    break
            if test_next == True:
                break
        if test_next == True:
            continue
        return [False, [IS, I_s]]

    return [True, None]

# Returns True if the matrix is considered secure, False otherwise
def algorithm_3(M, NUM_CELLS):
    t = NUM_CELLS
    s = 1

    V = VectorSpace(F, t)

    l = 4*t
    for r in range(2, l+1):
        next_r = False
        res_alg_2 = algorithm_2(M^r, t)
        if res_alg_2[0] == False:
            return [False, None]

        # if res_alg_2[1] == None:
        #     continue
        # IS = res_alg_2[1][0]
        # I_s = res_alg_2[1][1]
        # for j in range(1, r):
        #     IS = subspace_times_matrix(IS, M, t)
        #     I_j = []
        #     for i in range(0, s):
        #         new_basis = []
        #         for k in range(0, t):
        #             if k != i:
        #                 new_basis.append(V.basis()[k])
        #         iota_space = V.subspace(new_basis)
        #         if IS.intersection(iota_space) != iota_space:
        #             single_iota_space = V.subspace([V.basis()[i]])
        #             if IS.intersection(single_iota_space) == single_iota_space:
        #                 I_j.append(i)
        #             else:
        #                 next_r = True
        #                 break
        #     if next_r == True:
        #         break
        # if next_r == True:
        #     continue
        # return [False, [IS, I_j, r]]

    return [True, None]

def check_minpoly_condition(M, NUM_CELLS):
    max_period = 2*NUM_CELLS
    all_fulfilled = True
    M_temp = M
    for i in range(1, max_period + 1):
        if not ((M_temp.minimal_polynomial().degree() == NUM_CELLS) and (M_temp.minimal_polynomial().is_irreducible() == True)):
            all_fulfilled = False
            break
        M_temp = M * M_temp
    return all_fulfilled

def generate_matrix(FIELD, FIELD_SIZE, NUM_CELLS):
    if FIELD == 0:
        print("Matrix generation not implemented for GF(2^n).")
        exit(1)
    elif FIELD == 1:
        mds_matrix = create_mds_p(FIELD_SIZE, NUM_CELLS)
        result_1 = algorithm_1(mds_matrix, NUM_CELLS)
        result_2 = algorithm_2(mds_matrix, NUM_CELLS)
        result_3 = algorithm_3(mds_matrix, NUM_CELLS)
        while result_1[0] == False or result_2[0] == False or result_3[0] == False:
            mds_matrix = create_mds_p(FIELD_SIZE, NUM_CELLS)
            result_1 = algorithm_1(mds_matrix, NUM_CELLS)
            result_2 = algorithm_2(mds_matrix, NUM_CELLS)
            result_3 = algorithm_3(mds_matrix, NUM_CELLS)
        return mds_matrix

def generate_matrix_full(t):
    M = None
    if t == 2:
        M = matrix.circulant(vector([F(2), F(1)]))
    elif t == 3:
        M = matrix.circulant(vector([F(2), F(1), F(1)]))
    elif t == 4:
        M = matrix(F, [[F(5), F(7), F(1), F(3)], [F(4), F(6), F(1), F(1)], [F(1), F(3), F(5), F(7)], [F(1), F(1), F(4), F(6)]])
    elif (t % 4) == 0:
        M = matrix(F, t, t)
        # M_small = matrix.circulant(vector([F(3), F(2), F(1), F(1)]))
        M_small = matrix(F, [[F(5), F(7), F(1), F(3)], [F(4), F(6), F(1), F(1)], [F(1), F(3), F(5), F(7)], [F(1), F(1), F(4), F(6)]])
        small_num = t // 4
        for i in range(0, small_num):
            for j in range(0, small_num):
                if i == j:
                    M[i*4:(i+1)*4,j*4:(j+1)*4] = 2* M_small
                else:
                    M[i*4:(i+1)*4,j*4:(j+1)*4] = M_small
    else:
        print("Error: No matrix for these parameters.")
        exit()
    return M

def generate_matrix_partial(FIELD, FIELD_SIZE, t): ## TODO: Prioritize small entries
    entry_max_bit_size = FIELD_SIZE
    if FIELD == 0:
        print("Matrix generation not implemented for GF(2^n).")
        exit(1)
    elif FIELD == 1:
        M = None
        if t == 2:
            M = matrix(F, [[F(2), F(1)], [F(1), F(3)]])
        elif t == 3:
            M = matrix(F, [[F(2), F(1), F(1)], [F(1), F(2), F(1)], [F(1), F(1), F(3)]])
        else:
            M_circulant = matrix.circulant(vector([F(0)] + [F(1) for _ in range(0, NUM_CELLS - 1)]))
            M_diagonal = matrix.diagonal([F(grain_random_bits(entry_max_bit_size)) for _ in range(0, NUM_CELLS)])
            M = M_circulant + M_diagonal
            # while algorithm_1(M, NUM_CELLS)[0] == False or algorithm_2(M, NUM_CELLS)[0] == False or algorithm_3(M, NUM_CELLS)[0] == False:
            while check_minpoly_condition(M, NUM_CELLS) == False:
                #print("1")
                M_diagonal = matrix.diagonal([F(grain_random_bits(entry_max_bit_size)) for _ in range(0, NUM_CELLS)])
                M = M_circulant + M_diagonal

        if(algorithm_1(M, NUM_CELLS)[0] == False or algorithm_2(M, NUM_CELLS)[0] == False or algorithm_3(M, NUM_CELLS)[0] == False):
            print("Error: Generated partial matrix is not secure w.r.t. subspace trails.")
            exit()
        return M

def generate_matrix_partial_small_entries(FIELD, FIELD_SIZE, NUM_CELLS):
    if FIELD == 0:
        print("Matrix generation not implemented for GF(2^n).")
        exit(1)
    elif FIELD == 1:
        M_circulant = matrix.circulant(vector([F(0)] + [F(1) for _ in range(0, NUM_CELLS - 1)]))
        combinations = list(itertools.product(range(2, 6), repeat=NUM_CELLS))
        for entry in combinations:
            M = M_circulant + matrix.diagonal(vector(F, list(entry)))
            print(M)
            # if M.is_invertible() == False or algorithm_1(M, NUM_CELLS)[0] == False or algorithm_2(M, NUM_CELLS)[0] == False or algorithm_3(M, NUM_CELLS)[0] == False:
            if M.is_invertible() == False or check_minpoly_condition(M, NUM_CELLS) == False:
                continue
            return M

def matrix_partial_m_1(matrix_partial, NUM_CELLS):
    M_circulant = matrix.identity(F, NUM_CELLS)
    return matrix_partial - M_circulant

def print_linear_layer(M, n, t):
    print("n:", n)
    print("t:", t)
    print("N:", (n * t))
    print("Result Algorithm 1:\n", algorithm_1(M, NUM_CELLS))
    print("Result Algorithm 2:\n", algorithm_2(M, NUM_CELLS))
    print("Result Algorithm 3:\n", algorithm_3(M, NUM_CELLS))
    hex_length = int(ceil(float(n) / 4)) + 2 # +2 for "0x"
    print("Prime number:", "0x" + hex(PRIME_NUMBER))
    matrix_string = "["
    for i in range(0, t):
        matrix_string += str(["{0:#0{1}x}".format(int(entry), hex_length) for entry in M[i]])
        if i < (t-1):
            matrix_string += ","
    matrix_string += "]"
    print("MDS matrix:\n", matrix_string)

def calc_equivalent_matrices(MDS_matrix_field):
    # Following idea: Split M into M' * M'', where M'' is "cheap" and M' can move before the partial nonlinear layer
    # The "previous" matrix layer is then M * M'. Due to the construction of M', the M[0,0] and v values will be the same for the new M' (and I also, obviously)
    # Thus: Compute the matrices, store the w_hat and v_hat values

    MDS_matrix_field_transpose = MDS_matrix_field.transpose()

    w_hat_collection = []
    v_collection = []
    v = MDS_matrix_field_transpose[[0], list(range(1,t))]

    M_mul = MDS_matrix_field_transpose
    M_i = matrix(F, t, t)
    for i in range(R_P_FIXED - 1, -1, -1):
        M_hat = M_mul[list(range(1,t)), list(range(1,t))]
        w = M_mul[list(range(1,t)), [0]]
        v = M_mul[[0], list(range(1,t))]
        v_collection.append(v.list())
        w_hat = M_hat.inverse() * w
        w_hat_collection.append(w_hat.list())

        # Generate new M_i, and multiplication M * M_i for "previous" round
        M_i = matrix.identity(t)
        M_i[list(range(1,t)), list(range(1,t))] = M_hat
        M_mul = MDS_matrix_field_transpose * M_i

    return M_i, v_collection, w_hat_collection, MDS_matrix_field_transpose[0, 0]

def calc_equivalent_constants(constants, MDS_matrix_field):
    constants_temp = [constants[index:index+t] for index in range(0, len(constants), t)]

    MDS_matrix_field_transpose = MDS_matrix_field.transpose()

    # Start moving round constants up
    # Calculate c_i' = M^(-1) * c_(i+1)
    # Split c_i': Add c_i'[0] AFTER the S-box, add the rest to c_i
    # I.e.: Store c_i'[0] for each of the partial rounds, and make c_i = c_i + c_i' (where now c_i'[0] = 0)
    num_rounds = R_F_FIXED + R_P_FIXED
    R_f = R_F_FIXED / 2
    for i in range(num_rounds - 2 - R_f, R_f - 1, -1):
        inv_cip1 = list(vector(constants_temp[i+1]) * MDS_matrix_field_transpose.inverse())
        constants_temp[i] = list(vector(constants_temp[i]) + vector([0] + inv_cip1[1:]))
        constants_temp[i+1] = [inv_cip1[0]] + [0] * (t-1)

    return constants_temp

def poseidon(input_words, matrix, round_constants):

    R_f = int(R_F_FIXED / 2)

    round_constants_counter = 0

    state_words = list(input_words)

    # First full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        for i in range(0, t):
            state_words[i] = (state_words[i])^alpha
        state_words = list(matrix * vector(state_words))

    # Middle partial rounds
    for r in range(0, R_P_FIXED):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        state_words[0] = (state_words[0])^alpha
        state_words = list(matrix * vector(state_words))

    # Last full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        for i in range(0, t):
            state_words[i] = (state_words[i])^alpha
        state_words = list(matrix * vector(state_words))

    return state_words

def poseidon2(input_words, matrix_full, matrix_partial, round_constants):

    R_f = int(R_F_FIXED / 2)

    round_constants_counter = 0

    state_words = list(input_words)

    # First matrix mul
    state_words = list(matrix_full * vector(state_words))

    #print(state_words)
    # First full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, NUM_CELLS):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        for i in range(0, NUM_CELLS):
            state_words[i] = (state_words[i])^alpha
        state_words = list(matrix_full * vector(state_words))
        #print(state_words)

    # Middle partial rounds
    for r in range(0, R_P_FIXED):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, NUM_CELLS):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        state_words[0] = (state_words[0])^alpha
        #print(state_words)
        #print(matrix_partial)
        state_words = list(matrix_partial * vector(state_words))
        #print(state_words)

    # Last full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, NUM_CELLS):
            state_words[i] = state_words[i] + round_constants[round_constants_counter]
            round_constants_counter += 1
        for i in range(0, NUM_CELLS):
            state_words[i] = (state_words[i])^alpha
        state_words = list(matrix_full * vector(state_words))

    return state_words

# Init
#init_generator(FIELD, SBOX, FIELD_SIZE, NUM_CELLS, R_F_FIXED, R_P_FIXED)

# Round constants
#round_constants = generate_constants(FIELD, FIELD_SIZE, NUM_CELLS, R_F_FIXED, R_P_FIXED, PRIME_NUMBER)
#print_round_constants(round_constants, FIELD_SIZE, FIELD)

# Matrix
# MDS = generate_matrix(FIELD, FIELD_SIZE, NUM_CELLS)

#generate_matrix_partial(FIELD, FIELD_SIZE, NUM_CELLS)
#MATRIX_PARTIAL_DIAGONAL_M_1 = [matrix_partial_m_1(MATRIX_PARTIAL, NUM_CELLS)[i,i] for i in range(0, NUM_CELLS)]

def to_hex(value):
    l = len(hex(p - 1))
    if l % 2 == 1:
        l = l + 1
    value = hex(int(value))[2:]
    value = "0x" + value.zfill(l - 2)
    print("from_hex(\"{}\"),".format(value))

#print("use super::poseidon::PoseidonParams;")
#print("use bellman_ce::pairing::{bls12_381::Bls12, ff::ScalarEngine, from_hex};")
#print("type Scalar = <Bls12 as ScalarEngine>::Fr;")
#print("use lazy_static::lazy_static;")
#print("use std::sync::Arc;")
#print()
#print("lazy_static! {")
#
#
## # MDS
## print("pub static ref MDS{}: Vec<Vec<Scalar>> = vec![".format(t))
## for vec in MDS:
##     print("vec![", end="")
##     for val in vec:
##         to_hex(val)
##     print("],")
## print("];")
## print()
#
## Efficient partial matrix (diagonal - 1)
#print("pub static ref MAT_DIAG{}_M_1: Vec<Scalar> = vec![".format(t))
#for val in MATRIX_PARTIAL_DIAGONAL_M_1:
#    to_hex(val)
#print("];")
#print()
#
## Efficient partial matrix (full)
#print("pub static ref MAT_INTERNAL{}: Vec<Vec<Scalar>> = vec![".format(t))
#for vec in MATRIX_PARTIAL:
#    print("vec![", end="")
#    for val in vec:
#        val
#    print("],")
#print("];")
#print()

## Round constants
#print("pub static ref RC{}: Vec<Vec<Scalar>> = vec![".format(t))
#for (i,val) in enumerate(round_constants):
#    if i % t == 0:
#        print("vec![", end="")
#    to_hex(val)
#    if i % t == t - 1:
#        print("],")
#print("];")
#print()
#
#print("pub static ref POSEIDON_{}_PARAMS: Arc<PoseidonParams<Scalar>> = Arc::new(PoseidonParams::new({}, {}, {}, {}, &MAT_DIAG{}_M_1, &RC{}));".format(t, t, alpha, R_F_FIXED, R_P_FIXED , t, t))
#
#print("}")
#print()
#print()
#

def generate_constants_pos(field, n, t, R_F, R_P, prime_number):
    round_constants = []
    num_constants = (R_F + R_P) * t

    if field == 0:
        for i in range(0, num_constants):
            random_int = grain_random_bits(n)
            round_constants.append(random_int)
    elif field == 1:
        for i in range(0, num_constants):
            random_int = grain_random_bits(n)
            while random_int >= prime_number:
                # print("[Info] Round constant is not in prime field! Taking next one.")
                random_int = grain_random_bits(n)
            round_constants.append(random_int)
    return round_constants

def test_default_pos857(p_in, t_in,RF,RP,state_in):
    n = len(p_in.bits()) # bit
    print("Test: p=")
    if(p_in==p_koala):
        print("KoalaBear")
    else:
        if(p_in==p_m31):
            print("M31")
        else:
            if(p_in == p_bls):
                print("BLS")
            else:
                print(p_in)
    print(" RF=",RF, " RP=",RP)
    global FIELD
    FIELD = 1
    global FIELD_SIZE
    FIELD_SIZE = n
    global alpha

    alpha=get_alpha(p)
    global R_F_FIXED
    R_F_FIXED=RF
    global R_P_FIXED
    R_P_FIXED=RP
    global F
    F = GF(p_in)
    init_generator(1, 0, n, t_in, 8, 57)
    global grain_gen
    grain_gen = grain_sr_generator()
    const_def = generate_constants_pos(FIELD, n, t_in, 8, 57, p_in)
    full_def =   matrix(GF(p), [['0x3d955d6c02fe4d7cb500e12f2b55eff668a7b4386bd27413766713c93f2acfcd', '0x3798866f4e6058035dcf8addb2cf1771fac234bcc8fc05d6676e77e797f224bf', '0x2c51456a7bf2467eac813649f3f25ea896eac27c5da020dae54a6e640278fda2'],['0x20088ca07bbcd7490a0218ebc0ecb31d0ea34840e2dc2d33a1a5adfecff83b43', '0x1d04ba0915e7807c968ea4b1cb2d610c7f9a16b4033f02ebacbb948c86a988c3', '0x5387ccd5729d7acbd09d96714d1d18bbd0eeaefb2ddee3d2ef573c9c7f953307'],['0x1e208f585a72558534281562cad89659b428ec61433293a8d7f0f0e38a6726ac', '0x0455ebf862f0b60f69698e97d36e8aafd4d107cae2b61be1858b23a3363642e0', '0x569e2c206119e89455852059f707370e2c1fc9721f6c50991cedbbf782daef54']])

    print(full_def)
    #print(const_def)
    if(state_in[-1]==0):
        print(" input correct", state_in)
    else:
        print(" input WRONG")
    state_out = poseidon(state_in,  full_def,const_def)
    if(state_out[-1]==0):
        print("output valid")
        print(state_out)
    else:
        print("FAIL")

def test_default_pos(p_in, t_in,RF,RP,state_in):
    n = len(p_in.bits()) # bit
    print("Test: p=")
    if(p_in==p_koala):
        print("KoalaBear")
    else:
        if(p_in==p_m31):
            print("M31")
        else:
            if(p_in == p_bls):
                print("BLS")
            else:
                print(p_in)
    print(" RF=",RF, " RP=",RP)
    global FIELD
    FIELD = 1
    global FIELD_SIZE
    FIELD_SIZE = n
    global alpha

    alpha=get_alpha(p)
    global R_F_FIXED
    R_F_FIXED=RF
    global R_P_FIXED
    R_P_FIXED=RP
    global F
    F = GF(p_in)
    init_generator(1, 0, n, t_in, RF, RP)
    global grain_gen
    grain_gen = grain_sr_generator()
    const_def = generate_constants_pos(FIELD, n, t_in, RF, RP, p_in)
    full_def =   matrix(GF(p), [['0x3d955d6c02fe4d7cb500e12f2b55eff668a7b4386bd27413766713c93f2acfcd', '0x3798866f4e6058035dcf8addb2cf1771fac234bcc8fc05d6676e77e797f224bf', '0x2c51456a7bf2467eac813649f3f25ea896eac27c5da020dae54a6e640278fda2'],['0x20088ca07bbcd7490a0218ebc0ecb31d0ea34840e2dc2d33a1a5adfecff83b43', '0x1d04ba0915e7807c968ea4b1cb2d610c7f9a16b4033f02ebacbb948c86a988c3', '0x5387ccd5729d7acbd09d96714d1d18bbd0eeaefb2ddee3d2ef573c9c7f953307'],['0x1e208f585a72558534281562cad89659b428ec61433293a8d7f0f0e38a6726ac', '0x0455ebf862f0b60f69698e97d36e8aafd4d107cae2b61be1858b23a3363642e0', '0x569e2c206119e89455852059f707370e2c1fc9721f6c50991cedbbf782daef54']])

    print(full_def)
    #print(const_def)
    if(state_in[-1]==0):
        print(" input correct", state_in)
    else:
        print(" input WRONG")
    state_out = poseidon(state_in,  full_def,const_def)
    if(state_out[-1]==0):
        print("output valid")
        print(state_out)
    else:
        print("FAIL")

def test_full_pos(p_in, t_in,RF,RP,state_in,full,round_constants):
    n = len(p_in.bits()) # bit
    global NUM_CELLS
    NUM_CELLS = t_in
    print("Test: p=",p_in)
    print(" RF=",RF, " RP=",RP)
    global FIELD
    FIELD = 1
    global FIELD_SIZE
    FIELD_SIZE = n
    global alpha

    alpha=get_alpha(p_in)
    global R_F_FIXED
    R_F_FIXED=RF
    global R_P_FIXED
    R_P_FIXED=RP
    global F
    F = GF(p_in)
    init_generator(1, 0, n, t_in, RF, RP)
    global grain_gen
    grain_gen = grain_sr_generator()
    const_def = generate_constants(FIELD, n, t_in, RF, RP, p_in)
    full_def = generate_matrix(FIELD,n,t_in)
    #print(partial_def)

    full_correct = full_def==full
    if(full_correct == False):
        print(" full matrix WRONG")
        return

    rc_correct = const_def==round_constants
    if(rc_correct == False):
        print("constants WRONG")
        #print(const_def)
        return
    if(state_in[-1]==0 ):
        print(" input OK",state_in)
    else:
        print(" input WRONG")
        return
    state_out = poseidon(state_in,  full_def, const_def)
    if(state_out[-1]==0):
        print("output OK")
        print(state_out)
    else:
        print("FAIL")

def test_default_64(p_in, t_in,RF,RP,state_in):
    n = len(p_in.bits()) # bit
    print("Test: p=")
    if(p_in==p_goldilocks):
        print("GoldiLocks")
    else:
        if(p_in==p_m31):
            print("M31")
        else:
            print(p_in)
    print(" RF=",RF, " RP=",RP)
    global FIELD
    FIELD = 1
    global FIELD_SIZE
    FIELD_SIZE = n
    global alpha

    alpha=get_alpha(p_in)
    global R_F_FIXED
    R_F_FIXED=RF
    global R_P_FIXED
    R_P_FIXED=RP
    global F
    F = GF(p_in)
    init_generator(1, 0, n, t_in, RF, RP)
    global grain_gen
    grain_gen = grain_sr_generator()
    const_def = generate_constants(FIELD, n, t_in, RF, RP, p_in)
    full_def = generate_matrix_full(t_in)
    partial_def = generate_matrix_partial(FIELD,FIELD_SIZE,t_in)
    #print(partial_def)
    #print(const_def)
    if(state_in[-1]==0 ):
        print(" input correct", state_in)
    else:
        print(" input WRONG")
        return
    state_out = poseidon2(state_in,  full_def, partial_def, const_def)
    if(state_out[-1]==0):
        print("output valid")
        print(state_out)
    else:
        print("FAIL")

def test_default(p_in, t_in,RF,RP,state_in):
    n = len(p_in.bits()) # bit
    print("Test: p=")
    if(p_in==p_koala):
        print("KoalaBear")
    else:
        if(p_in==p_m31):
            print("M31")
        else:
            print(p_in)
    print(" RF=",RF, " RP=",RP)
    global FIELD
    FIELD = 1
    global FIELD_SIZE
    FIELD_SIZE = n
    global alpha

    alpha=get_alpha(p)
    global R_F_FIXED
    R_F_FIXED=RF
    global R_P_FIXED
    R_P_FIXED=RP
    global F
    F = GF(p_in)
    init_generator(1, 0, n, t_in, RF, RP)
    global grain_gen
    grain_gen = grain_sr_generator()
    const_def = generate_constants(FIELD, n, t_in, RF, RP, p_in)
    full_def = generate_matrix_full(t_in)
    partial_def = generate_matrix_partial(FIELD,FIELD_SIZE,t_in)
    #print(partial_def)
    #print(const_def)
    if(state_in[-1]==0 and state_in[-2]==0):
        print(" input correct", state_in)
    else:
        print(" input WRONG")
        return
    state_out = poseidon2(state_in,  full_def, partial_def, const_def)
    if(state_out[-1]==0 and state_out[-2]==0):
        print("output valid")
        print(state_out)
    else:
        print("FAIL")

def test_full(p_in, t_in,RF,RP,state_in,full_in,partial_in,round_constants):
    n = len(p_in.bits()) # bit
    global NUM_CELLS
    NUM_CELLS = t_in
    print("Test: p=")
    if(p_in==p_koala):
        print("KoalaBear")
    else:
        if(p_in==p_m31):
            print("M31")
        else:
            print(p_in)
    print(" RF=",RF, " RP=",RP)
    global FIELD
    FIELD = 1
    global FIELD_SIZE
    FIELD_SIZE = n
    global alpha

    alpha=get_alpha(p_in)
    global R_F_FIXED
    R_F_FIXED=RF
    global R_P_FIXED
    R_P_FIXED=RP
    global F
    F = GF(p_in)
    init_generator(1, 0, n, t_in, RF, RP)
    global grain_gen
    grain_gen = grain_sr_generator()
    full_def = generate_matrix_full(t_in)
    partial_def = generate_matrix_partial(FIELD,FIELD_SIZE,t_in)
    #print(partial_def)

    init_generator(1, 0, n, t_in, RF, RP)
    const_def = generate_constants(FIELD, n, t_in, RF, RP, p_in)
    full_correct = full_def==full_in
    if(full_correct == False):
        print(" full matrix WRONG")
        return
    partial_correct = partial_def==partial_in
    if(partial_correct == False):
        print(" partial matrix CUSTOM")
        if(check_minpoly_condition(partial_in, t_in) == False):
             print(" partial matrix NOT MINPOLY")
             return
        else:
            if(algorithm_1(partial_in, NUM_CELLS)[0] == False or algorithm_2(partial_in, NUM_CELLS)[0] == False or algorithm_3(partial_in, NUM_CELLS)[0] == False):
                print("Error: custom partial matrix is not secure w.r.t. subspace trails.")
                return
            else:
                print(" partial matrix OK")
    rc_correct = const_def==round_constants
    if(rc_correct == False):
        print("constants WRONG")
        print(const_def)
        return
    if(state_in[-1]==0 and state_in[-2]==0):
        print(" input OK",state_in)
    else:
        print(" input WRONG")
        return
    state_out = poseidon2(state_in,  full_in, partial_in, round_constants)
    if(state_out[-1]==0 and state_out[-2]==0):
        print("output OK")
        print(state_out)
    else:
        print("FAIL")

p_koala = 2**31 - 2**24 + 1
p_baby=2**31 - 2**27 + 1
p_m31 = 2**31-1
p_bls=0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
p_goldilocks = 2**64-2**32 +1
# round numbers
#p=p_koala
#t=24
#d=3
#sec=128
#R_F_FIXED, R_P_FIXED, _, _ = poseidon_calc_final_numbers_fixed(p, t, d, sec, True)
#print("+++ R_F = {0}, R_P = {1} +++".format(R_F_FIXED, R_P_FIXED))

# round constants
p=p_baby
#n=len(p.bits())
#FIELD_SIZE=n
t=24
NUM_CELLS=t
F=GF(p)
RF=8
RP=21
init_generator(1, 0, n, t, RF, RP)
grain_gen = grain_sr_generator()
const_def = generate_constants(1, n, t, RF, RP, p)
#ex_length=10
#print(["{0:#0{1}x}".format(entry, hex_length) for entry in const_def])
#print(const_def)



#M31 6-4 Simon
state_in=[1253653436, 1786871594, 1305924322, 1544663848, 1581949976, 1400364434, 754998583, 2043969975, 185533518, 2025969807, 994876701, 1122188176, 1954375908, 724240441, 0, 0]
p=p_m31
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=6
RP=4
test_default(p,t,RF,RP,state_in)


#256 6-8 William
state_in=[37023452605574130558326904361999883803144650872871342440196074025584027629531, 33803624965874294767348858029070942954264467138937783374355563619976160210399, 0]
p=p_bls
n=len(p.bits())
FIELD_SIZE=n
t=3
NUM_CELLS=t
F=GF(p)
RF=6
RP=8
test_default_pos857(p,t,RF,RP,state_in)

#256 6-8 Augustin
state_in =  (37023452605574130558326904361999883803144650872871342440196074025584027629531, 33803624965874294767348858029070942954264467138937783374355563619976160210399, 0)
p=p_bls
n=len(p.bits())
FIELD_SIZE=n
t=3
NUM_CELLS=t
F=GF(p)
RF=6
RP=8
test_default_pos857(p,t,RF,RP,state_in)


#256 6-9 Mael
state_in =  (49174981350372926196070250608201907029468160196865045509782481794998042706016,23157595570482944668655507349862357725255958437192820063381545033931658895106, 0)
p=p_bls
n=len(p.bits())
FIELD_SIZE=n
t=3
NUM_CELLS=t
F=GF(p)
RF=6
RP=9
test_default_pos857(p,t,RF,RP,state_in)


#256 6-9 Giuseppe
state_in =  (int('0x25ac5ceccc820aad2e37a2256c0a03d9884440fef975b173ab3bdfe58c75506f',0), int('0x2cd9ab9e28721e90852b68fb464d424909c6992d671247567ca4074e59229919',0), 0)
p=p_bls
n=len(p.bits())
FIELD_SIZE=n
t=3
NUM_CELLS=t
F=GF(p)
RF=6
RP=9
test_default_pos(p,t,RF,RP,state_in)


#64 6-7 Ziyu
state_in =  (int('0x9826c90024f015ed',0), int('0x4fe1028edece4068',0), int('0xbb9a0efa783f6be0',0), int('0x445dd11f4d8c4467',0),
             int('0xbbf1456451441df7',0), int('0xb8c0e0d8422e17d6',0), int('0x03ad4e42bff95121',0), 0)
p=p_goldilocks
n=len(p.bits())
FIELD_SIZE=n
t=8
NUM_CELLS=t
F=GF(p)
RF=6
RP=7
test_default_64(p,t,RF,RP,state_in)

#64 6-7 Vitto
state_in = (int('0x588c0c67b2502af5',0), int('0x2dbb38f5173fda8c',0), int('0x7535686a3845a2ce',0), int('0xbdc64968566f29ba',0), int('0x3563bae427795f71',0), int('0x6e905203c11a3f4f',0), int('0x8c38de6f0b49798a',0), 0)
p=p_goldilocks
n=len(p.bits())
FIELD_SIZE=n
t=8
NUM_CELLS=t
F=GF(p)
RF=6
RP=7
test_default_64(p,t,RF,RP,state_in)

#64 6-8 Ziyu
state_in =  (int('0x25a7341d7eaafbab',0), int('0x27c9e8655a443680',0), int('0x84c15b5a7696b8e5',0), int('0x2aa0561891e6dcb2',0),
             int('0x99d4207071524acc',0), int('0x01d73f4eeca70b48',0), int('0xf3d9dc91de56fe43',0), 0)
p=p_goldilocks
n=len(p.bits())
FIELD_SIZE=n
t=8
NUM_CELLS=t
F=GF(p)
RF=6
RP=8
test_default_64(p,t,RF,RP,state_in)

#64 6-10 Ziyu
state_in =  (int('0xcea391abcd5ebcf7',0), int('0x9df8c713dff7001f',0), int('0x62b623a4ad4cba13',0), int('0x7fd4f3eca4f0e76b',0), int('0xa24f039fb7a4c4c6',0), int('0x9598b77377f86820',0), int('0x497f6814db6048a5',0), 0)
p=p_goldilocks
n=len(p.bits())
FIELD_SIZE=n
t=8
NUM_CELLS=t
F=GF(p)
RF=6
RP=10
test_default_64(p,t,RF,RP,state_in)

#M31 4-0 Augustin
state_in=[1375658479, 1440638775, 722672702, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p=p_m31
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=0
matrix_64 =  matrix(GF(p), [(10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3), (8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1), (2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7), (2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3), (4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1), (1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7), (1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3), (4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1), (1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7), (1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6), (4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2), (1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14), (1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12)])
matrix_64p =  matrix(GF(p), [(2009355924, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1306606973, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 723596936, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1176616768, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1684779497, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1878699402, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 785063687, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 97347483, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1700634002, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1612845569, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1942737280, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 750676168, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 537069089, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1924060712, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1495552202, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1278548271)])
const_in = [226704312, 316265343, 1472549403, 1217781199, 1395035325, 547854691, 819064116, 1191567309, 849896959, 1168130556, 2123946399, 459481733, 1824458301, 2140814884, 746408967, 1589931516, 731433451, 898611546, 1580677565, 1094533878, 661868724, 2020050173, 1248139031, 616333578, 1890924946, 1855903250, 1514428339, 1983376892, 1895278166, 1705568081, 594658801, 1094313992, 1318977300, 1994451452, 268977025, 1855149434, 1667048974, 1952889479, 1905277381, 551528522, 3734540, 499182103, 1123741764, 1917731673, 395692497, 1805260297, 965868285, 521793363, 616968553, 879153415, 1081048825, 1615116679, 326321622, 1113098951, 1594928998, 1737527403, 1589260406, 1181432704, 1186809305, 1808187817, 716091715, 1880615406, 1355666208, 2027757314]
test_full(p,16,RF,RP,state_in,matrix_64,matrix_64p,const_in)

#M31 4-0 Aurelien
state_in=[800306752, 687581107, 893716298, 1248843338, 1566084329, 661753910, 221855592, 1397986702, 336528165, 1544362242, 817554900, 2143157771, 1269543469, 1438293486, 0, 0]
p=p_m31
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=0
test_default(p,t,RF,RP,state_in)



#Koala 4-1 Augustin
state_in=[866213119, 2121603431, 399651914, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p=p_koala
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=1
matrix_64 =  matrix(GF(p), [(10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3), (8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1), (2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7), (2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3), (4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1), (1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7), (1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3), (4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1), (1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7), (1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6), (4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2), (1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14), (1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12)])
matrix_64p =  matrix(GF(p), [(1096021183, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 814460776, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 934368849, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 41893401, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1147707263, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1437422019, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1684478368, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1223082109, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 836320068, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 43278885, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 451951466, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1776951132, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 279065324, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1738691378, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1553815976, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 413947892)])
const_in = [1093942445, 1808482619, 780143177, 1179440467, 763942825, 2036272747, 835274320, 645875621, 1650848763, 739940123, 132967493, 654424995, 833204442, 1412170444, 1703384670, 954551665, 423767679, 734859160, 1161119182, 919868550, 322896177, 2007270804, 1610112249, 1744580271, 1982424230, 672961514, 122517896, 125230796, 14382823, 921034199, 108186423, 738294548, 757565670, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1799625725, 10276492, 2049088975, 1970932243, 1554525273, 976548694, 1641021206, 1149352736, 169719922, 370873080, 421489789, 148985846, 1417852836, 899302250, 1606160204, 437867667, 2105576993, 455916001, 315336534, 482884321, 1877970152, 1996360241, 1477609578, 77965710, 769088767, 1757983557, 235855642, 277667173, 1536772257, 1895067968, 1199180942, 1141925387]
test_full(p,16,RF,RP,state_in,matrix_64,matrix_64p,const_in)


#Koala 4-1 Aurelien
state_in=[430190149, 1651738314, 1336005572, 1579651201, 540586125, 435683671, 706540295, 132448139, 277657735, 2044147576, 112520025, 1024878660, 1512414864, 1598226771, 0, 0]
p=p_koala
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=1
test_default(p_koala,t,RF,RP,state_in)

#Koala 4-3 Augustin
state_in=[1704422453, 494728086, 77619236, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p=p_koala
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=3
matrix_64 =  matrix(GF(p), [(10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3), (8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1), (2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7), (2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3), (4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1), (1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7), (1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3), (4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1), (1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7), (1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6), (4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2), (1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14), (1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12)])
matrix_64p =  matrix(GF(p), [(984358666, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1303551476, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1187097714, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 715486266, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 580355068, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 742684979, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 2061358628, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1547326326, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1928456911, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 243648179, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1136816015, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 160234133, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 725395075, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1098449497, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 245114623, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 364398504)])
const_in =[571540353, 1446309629, 1904138829, 1036434216, 787668572, 1811260592, 1492940455, 277262745, 16753869, 1407277639, 382339779, 1218722160, 182359994, 1056713759, 5252551, 68291620, 1597028645, 1584636130, 499352678, 490770898, 20332875, 1573979391, 1391243319, 99909838, 898920316, 755925537, 848223180, 1093432402, 1786939066, 907800845, 340890064, 1671018686, 1761806629, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 714834644, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 456275961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1883074669, 390508812, 1603737300, 837208684, 1118793067, 2089376242, 1647437275, 61175508, 1310855311, 64172091, 1232794180, 1937352480, 2124954388, 1992064147, 201074732, 1384413847, 1426839211, 882376031, 344413866, 551067343, 803393938, 117462833, 755733468, 160396857, 955274024, 1956492130, 264064487, 1325007417, 105231208, 798017860, 1735966769, 1276795657]
test_full(p,16,RF,RP,state_in,matrix_64,matrix_64p,const_in)


#Koala 4-3 Aurelien
state_in=[122728652, 238840450, 1171191875, 425446060, 840743408, 28275788, 279254317, 1099296188, 6674620, 1339786377, 714397610, 756315142, 945717883, 1057760140, 0, 0]
p=p_koala
n=len(p_koala.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=3
#init_generator(FIELD, SBOX, n, t, RF, RP)
#m_part = generate_matrix_partial(FIELD,FIELD_SIZE,t)
#rc_in = generate_constants(FIELD, n, t, RF, RP, p_koala)
test_default(p,t,RF,RP,state_in)

#M31 4-1 Augustin
state_in=[1451396292, 745699335, 1638213306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p=p_m31
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=1
matrix_64 =  matrix(GF(p), [(10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3), (8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1), (2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7), (2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3), (4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1), (1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7), (1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3), (4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1), (1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7), (1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6), (4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2), (1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14), (1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12)])
matrix_64p =  matrix(GF(p), [(1365459094, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 688048860, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1470115723, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1500521211, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1680922619, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1272262914, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1275388246, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 844784829, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1706658472, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 2051757360, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1366089976, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 77633995, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 632856853, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 364154585, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 256285265, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 181966044)])
const_in = [1182591191, 245459526, 80186009, 1948037185, 1350436495, 1449111652, 1271541733, 618732171, 1710955289, 1090816645, 1217786254, 721732097, 686233241, 1150848696, 307414149, 1920191101, 88948756, 1329827747, 875786758, 132889202, 1074162980, 1766999771, 492235992, 2061343503, 1517024595, 534738949, 838363649, 254279956, 1553366804, 1756173309, 1473904101, 628009981, 1492201877, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1794359324, 238953411, 1956096993, 250751026, 311096708, 105425483, 2041206846, 1152320136, 1440060626, 1754852310, 1503029379, 1929380708, 813213280, 1794069570, 1395517722, 688759636, 636673543, 608825989, 1419326842, 1042805181, 59287791, 1679331971, 1533199877, 414051473, 1455969335, 448957059, 1810300544, 1866156442, 2043858914, 1574271859, 535205928, 750137461]
test_full(p,16,RF,RP,state_in,matrix_64,matrix_64p,const_in)

#M31 4-1 Aurelien
state_in=[156072055, 1750367252, 669257734, 644436946, 1738814276, 266232228, 1251957450, 1027364303, 533809060, 41920449, 1156126359, 884814381, 1263193020, 1002906903, 0, 0]
p=p_m31
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=4
RP=1
test_default(p,t,RF,RP,state_in)


#Koala-6-4 Augustin
p=p_koala
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=6
RP=4
state_in    =  [886965183, 1272215886, 1618055775, 416749689, 2026120333, 1076267744, 378759084, 1817282391, 2064291985, 422064640, 480407977, 378580554, 999725627, 1956139521, 0, 0]
matrix_64 =  matrix(GF(p), [(10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3), (8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1), (2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7), (2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3, 5, 7, 1, 3), (4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1, 4, 6, 1, 1), (1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7, 1, 3, 5, 7), (1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6, 5, 7, 1, 3), (4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2, 4, 6, 1, 1), (1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14, 1, 3, 5, 7), (1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12, 1, 1, 4, 6), (5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 10, 14, 2, 6), (4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 8, 12, 2, 2), (1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 2, 6, 10, 14), (1, 1, 4, 6, 1, 1, 4, 6, 1, 1, 4, 6, 2, 2, 8, 12)])
matrix_64p =  matrix(GF(p), [(1416712459, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1955712504, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 295700336, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1890306788, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 294300781, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1328411125, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 795028906, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 748143882, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1580830589, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 813305670, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1215852660, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 548670650, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1548158565, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1998199179, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1323426891, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1097977862)])
const_in = [1706588357, 1451927094, 1222141810, 617666351, 1618150787, 1814670308, 1179861784, 1958236404, 2003749724, 1999925747, 1139153834, 1777005169, 1772087355, 1480610793, 1960066815, 1239199012, 513009161, 1073753319, 1373095376, 179236270, 286534886, 340094459, 685250600, 1909490957, 974494750, 323533795, 870001772, 352226701, 1265939385, 1619985944, 1958973842, 1058185337, 1499595419, 1540321472, 1324591740, 1701341168, 1832844294, 170011489, 389293828, 1422290304, 2081028640, 234080385, 1214362384, 558291169, 362229513, 1098922575, 1712495442, 723740255, 1617014838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1602604581, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 797536181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 834068306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1470213609, 416984376, 2113198365, 653970188, 1265755570, 551718196, 1543047574, 581105245, 684873846, 1800813506, 331953770, 348099884, 31122366, 1075418394, 2022225673, 1764274568, 359606723, 1142777269, 1657850758, 583554982, 1680073438, 1033569629, 1059766286, 738714550, 605851464, 6102655, 1742383057, 1896946110, 1413305809, 1937054006, 72719565, 730521653, 1337092683, 309806984, 1258804743, 1795593221, 2036713279, 263899313, 724121163, 1751912886, 572051146, 1388117757, 1294093206, 1634254927, 1629326897, 1757275967, 1138533369, 1447769218]
test_full(p,16,6,4,state_in,matrix_64,matrix_64p,const_in)



#M31 6-1 Aurelien+Augustin
state_in=[1576142736, 1305678766, 27403518, 1166287757, 277522312, 1637895777, 1580628611, 1590109350, 427583191, 1032942118, 109002218, 954619980, 766949687, 803337492, 0, 0]
p=p_m31
n=len(p.bits())
FIELD_SIZE=n
t=16
NUM_CELLS=t
F=GF(p)
RF=6
RP=1
test_default(p,t,RF,RP,state_in)
︡f4ef8360-cfef-4dd6-8605-4cebc643147d︡{"stderr":"Error in lines 741-741\nTraceback (most recent call last):\n  File \"/cocalc/lib/python3.11/site-packages/smc_sagews/sage_server.py\", line 1250, in execute\n    exec(\n  File \"\", line 1, in <module>\n  File \"\", line 4, in init_generator\nNameError: name 'FIELD_SIZE' is not defined\n"}︡{"done":true}









