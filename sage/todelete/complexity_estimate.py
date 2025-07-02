from sage.all import *
from itertools import product

# Input data
data = [
    (37, 17, 17, 2),
    (25, 8, 8, 3),
    (24, 5, 5, 4),
    (56, 25, 25, 2),
    (49, 11, 11, 3),
    (37, 8, 8, 4),
    (24, 5, 5, 5),
    (75, 33, 33, 2),
    (66, 15, 15, 3),
    (60, 10, 10, 4),
    (29, 6, 6, 5)
]

# Generate all compositions of total into 'parts' non-negative integers
def compositions_with_sum(total, parts):
    return IntegerVectors(total, parts) #i.e. all tuples such that Sigma (parts) =total

# Compute partial E_i for given v, o', m, l, i
def compute_partial_Ei(v, o_prime, m, l, i):
    inner_sum = 0
    for a_vector in compositions_with_sum(2 * i, l):
        term = (-1)**i * binomial(m * l + i - 1, i)
        # term = (-1)**i * binomial(m + i - 1, i)
        for a_j in a_vector:
            term *= binomial(v + o_prime, v + a_j)
        inner_sum += term
    return inner_sum

# Find lowest o' and i such that E becomes non-positive
def find_min_oprime_and_i(v, o, m, l):
    for o_prime in range(1, o):
        E = 0
        max_i = floor(o_prime * l / 2)
        for i in range(max_i + 1):
            Ei = compute_partial_Ei(v, o_prime, m, l, i)
            E += Ei
            if E <= 0:
                return o_prime, i, E
    return None, None, None

# Compute E (second form) for given v, o, m, l
def compute_E_full(v, o, m, l):
    E = 0
    for i in range(1, floor(o * l / 2) + 1):
        inner_sum = 0
        for a_vector in compositions_with_sum(2 * i, l):
            term = (-1)**(i + 1) * binomial(m * l + i - 1, i)
            # term = (-1)**(i + 1) * binomial(m  + i - 1, i)
            for a_j in a_vector:
                term *= binomial(v + o, v + a_j)
            inner_sum += term
        E += inner_sum
    return E

# Main loop
for v, o, m, l in data:
    o_prime, i_min, E1 = find_min_oprime_and_i(v, o, m, l)
    if o_prime is not None:
        # Compute E, U and log_2(expression)
        E = compute_E_full(v, o_prime, m, l)
        U = binomial(v + o_prime, v)^l
        factor = binomial(v + 2, 2)^i_min
        try:
            result_log_1 = log(E * U * factor, 2).n()
            result_log_2 = log(E * U **2, 2).n()
        except Exception as e:
            result_log_1 = f"Error: {e}"
            result_log_2 = f"Error: {e}"

        print(f"v={v}, o={o}, m={m}, l={l} => lowest o'={o_prime}, i={i_min}")
        print(f"    E = {E}")
        print(f"    U = {U}")
        print(f"    log2(E * U * (binom(v+2,2))^i) = {result_log_1}")
        print(f"    log2(E * U^2) = {result_log_2}")
    else:
        print(f"v={v}, o={o}, m={m}, l={l} => No o' < o found with E <= 0")
