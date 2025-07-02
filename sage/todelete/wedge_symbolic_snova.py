from sage.all import *
from ..utils.functions import *
import os

# ----------------------- parameters -----------------------
l = 3           # Extension
n = 5          # ambient dimension (n ≥ k+2)
k = 2           # number of unknown vectors v_1,…,v_k, dimension of the complement space of oil space
m = 2          # number of alternating matrices Q_i, number of eqs
q = 16
F = GF(q)       # Base Finite field
seed = 20250615
set_random_seed(seed)

# Path to S matrices
sobj_path = os.path.join(os.path.dirname(__file__), "S_matrices.sobj")

# Load list of S matrices
S_matrices = load(sobj_path)

# Use S
S = S_matrices[l-2]
print("S = ")
print(S)
char_poly = S.charpoly()

# Construct the extension field
K = PolynomialRing(F,'x')
x = K.gens()
f = K(char_poly)

F_S = F.extension(f, 's')
s = F_S.gens()

# Homomorphism
phi = F_S.hom([S], MatrixSpace(F,l))
"""
Wedge operation
"""
# ----------------------------------------------------------
# 1) Generate m UOV polar forms Q_i
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
    
    return P

Q_list_base = gen_poly(n,m,F_S,n-k)

# Lift all matrices in Q_list_base
Q_list = []
for mat in Q_list_base:
    Q_list.append(lift_matrix(mat,phi)+lift_matrix(mat,phi).transpose())

# Redefine params
n = n*l
k = k*l

# 2) Polynomial ring with k·n indeterminates
names = [f"x{r}{i}" for r in range(k) for i in range(n)]
R = PolynomialRing(F_S, names)
x = R.gens()
x_var = [[x[r*n + i] for i in range(n)] for r in range(k)]

# 3) Exterior algebra Λ*(F^n)
E = ExteriorAlgebra(R, n)
e = E.gens()
print(e[i] for i in range(n))

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
    print(f"\nQ_{idx+1} of rank {Q.rank()}")
    color_matrix(Q)

print(f"\nPolynomial ring R with {k*n} variables: {R.variable_names()}")
print(f"Number of equations generated: {len(eqns)} = {m} × binom({n}, {k+2})")

"""
MACAULAY
"""

# Step 1: Define target degree
target_degree = k

# Step 2: Get all monomials in the ring of the correct degree
monomial_list = R.monomials_of_degree(target_degree)
mon_idx = {m.exponents()[0]: i for i, m in enumerate(monomial_list)}

# Step 3: Initialize Macaulay matrix
M = Matrix(F_S, len(eqns), len(monomial_list))

# Step 4: Fill Macaulay matrix row by row
for row_idx, poly in tqdm(enumerate(eqns), ncols = 100, desc = "Filling Macaulay matrix ... "):
    for mon, coeff in zip(poly.monomials(), poly.coefficients()):
        expt = mon.exponents()[0]
        col_idx = mon_idx.get(expt)
        if col_idx is not None:
            M[row_idx, col_idx] = coeff

# color_matrix(M)

# Step 4: Print matrix with column headers
print("\nMacaulay matrix of ",len(monomial_list),"columns and ",len(eqns),"rows.")
# color_matrix(M)


# Step 5: Number of linear independent eqs
rank = M.rank()
print(f"\nMacaulay matrix rank = {rank} = linearly independent equations")
print(f"\nParameters : v+o = {n}, v = {k}, m = {m}, l = {l}")
print(f"In theory, \nNumber of variables = {len(names)}; binom(lv+lo,lv+2) = {binomial(l*n,l*k+2)}; binom(v+o,o)^l = {binomial(n,k)**l}; and m x binom(lv+lo,lv+2) = {binomial(l*n,l*k+2)*m}; and m x binom(lv+lo,lv+2l) = {binomial(l*n,l*k+2*l)*m}")
