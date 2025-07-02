from sage.all import *
from ..utils.functions import *

# ----------------------- parameters -----------------------
n = 8           # ambient dimension (n ≥ k+2)
k = 2           # number of unknown vectors v_1,…,v_k, dimension of the complement space of oil space
m = 4           # number of alternating matrices Q_i, number of eqs
q = 256
F = GF(q)       # Base Finite field
seed = 20250615
set_random_seed(seed)

# ----------------------------------------------------------

# 1) Polynomial ring with k·n indeterminates
names = [f"x{r}{i}" for r in range(k) for i in range(n)]
R = PolynomialRing(F, names)
x = R.gens()
x_var = [[x[r*n + i] for i in range(n)] for r in range(k)]

# 2) Generate m UOV polar forms Q_i
def gen_poly(n: int,m: int,F: FiniteField, d:int):
    """
    n = dimension of ambient space
    m = number of eqs
    d = dimension of oil space
    """
    # Generate Oil subspace matrices Vertical O_I = [O] i.e. O_I^t * M * O_I = 0
    #                                               [I]
    """ All vectors are vertical vevtors"""
    O, O_I = generate_oil_subspace_matrix(F, d, n)
    # Generate public keys polar forms M[i]
    """    P is the original pk, M is P + P^t """
    P = generate_public_matrices(F, m, n, O)
    M = generate_list_M(P)
    
    return P, M

Q_list = gen_poly(n,m,F,n-k)[1]

"""
EXTERIOR ALGEBRA
"""

# 3) Exterior algebra Λ*(F^n)
E = ExteriorAlgebra(R, n)
e = E.gens()

# 3a) Unknown vectors v_1,...,v_k as 1-forms
v = [sum(x_var[r][i] * e[i] for i in range(n)) for r in range(k)]

# 4) Construct k-form V = v_1 ∧ ... ∧ v_k
from functools import reduce
V = reduce(lambda a, b: a * b, v)

# 5) For each Q_i, construct Q̂_i and W_i = V ∧ Q̂_i
eqns = []
for Q in Q_list:
    Q_hat = sum(Q[i, j] * e[i] * e[j] for i in range(n) for j in range(i+1, n))
    W = V * Q_hat
    eqns.extend(W.monomial_coefficients().values())

# 6) Display info
print(f"Generated {m} random skew-symmetric matrices Q_1,...,Q_{m} (n={n})")
for idx, Q in enumerate(Q_list):
    print(f"\nQ_{idx+1} = ")
    color_matrix(Q)

print(f"\nPolynomial ring R with {k*n} variables: {R.variable_names()}")
print(f"Number of equations generated: {len(eqns)} = {m} × binom({n}, {k+2})")

# print("\nPrint equations:")
# for eq in eqns:
#     print(eq,"\n")
"""
MACAULAY
"""

# Step 1: Define target degree
target_degree = k

# Step 2: Get all monomials in the ring of the correct degree
monomial_list = R.monomials_of_degree(target_degree)
mon_idx = {m.exponents()[0]: i for i, m in enumerate(monomial_list)}

# Step 3: Initialize Macaulay matrix
M = Matrix(F, len(eqns), len(monomial_list))

# Step 4: Fill Macaulay matrix row by row
for row_idx, poly in tqdm(enumerate(eqns), ncols = 100, desc = "Filling Macaulay matrix ... "):
    for mon, coeff in zip(poly.monomials(), poly.coefficients()):
        expt = mon.exponents()[0]
        col_idx = mon_idx.get(expt)
        if col_idx is not None:
            M[row_idx, col_idx] = coeff

# Step 4: Print matrix with column headers
print("\nMacaulay matrix of ",len(monomial_list),"columns and ",len(eqns),"rows.")
# color_matrix(M)


# Step 5: Number of linear independent eqs
rank = M.rank()
print(f"\nMacaulay matrix rank = linearly independent equations = {rank}")
print(f"\nv+o = {n}, v = {k}, m = {m}")

# def compute_sum(m_val, v_val, o_val):
#     return sum(
#         (-1)^i * binomial(m_val + i - 1, i) * binomial(v_val + o_val, v_val + 2*i)
#         for i in range(floor(o_val/2) + 1)
#     )
# print(compute_sum(m,k,n-k))