from sage.all import *
from ..utils.functions import *
from itertools import combinations

# ----------------------- parameters -----------------------
l = 2           # Extension
n = 4           # ambient dimension (n ≥ k+2)
k = 1           # number of unknown vectors v_1,…,v_k, dimension of the complement space of oil space
m = 5           # number of alternating matrices Q_i, number of eqs
q = 256
F = GF(q)       # Base Finite field
seed = 20250615
set_random_seed(seed)

"""
Wedge simulation
"""
# ----------------------------------------------------------
"""
variables diagonal
"""
# # 1) Polynomial ring with k·n·l indeterminates
# names = [f"x{r}{i}{j}" for r in range(k) for i in range(n) for j in range(l)]
# R = PolynomialRing(F, names)
# x = R.gens()

# # 2) Helper to access variable for (r, i, j)
# def var(r, i, j):
#     idx = (r * n + i) * l + j
#     return x[idx]

# # 3) Create blocks: each is an l x l diagonal matrix (same for circulant matrix or SNOVA)
# blocks = [[None for _ in range(n)] for _ in range(k)]
# for r in range(k):
#     for i in range(n):
#         diag_entries = [var(r, i, j) for j in range(l)]
#         blocks[r][i] = matrix(R, l, lambda a, b: diag_entries[a] if a == b else 0)

# # 4) Assemble block matrix
# X = block_matrix(k, n, blocks)
# print(X)

"""
variables normal
"""
# 1) Define the polynomial ring with k·n·l² variables
names = [f"x{r}{i}" for r in range(l*k) for i in range(l*n)]
R = PolynomialRing(F, names)
x = R.gens()

# 2) Helper: get the variable
def var(r, i):
    return x[r * n * l + i]

# 3) Build the full matrix of size (k·l) × (n·l)
X = matrix(R, k * l, n * l)

for r in range(k*l):
    for i in range(n*l):
        X[r, i] = var(r, i)

# 4) Done — X is a general kl × nl matrix of independent variables
print(X)

"""
variable block-circulant
"""
# # 1) Define polynomial ring with k·n·l variables (one per circulant block row 0)
# names = [f"x{r}{i}{j}" for r in range(k) for i in range(n) for j in range(l)]
# R = PolynomialRing(F, names)
# x = R.gens()

# # 2) Helper to get variable for block (r, i), j-th entry of first row
# def var(r, i, j):
#     return x[(r * n + i) * l + j]

# # 3) Build circulant blocks for each (r, i)
# def circulant_block(r, i):
#     first_row = [var(r, i, j) for j in range(l)]
#     return matrix(R, l, l, lambda a, b: first_row[(b - a) % l])

# # 4) Construct block matrix with circulant blocks
# blocks = [[circulant_block(r, i) for i in range(n)] for r in range(k)]
# X = block_matrix(k, n, blocks)

# # 5) Resulting matrix is (k·l) × (n·l), filled with structured variables
# print(X)
"""
Equations
"""
# 2) Generate m random matrices whose m × n blocks are all l × l circulant matrices 
def random_circulant_matrix(F, l):
    """
    Return a random l×l circulant matrix over the finite field F.
    Only the first row is sampled; the remainder is fixed by circulancy.
    """
    first_row = [F.random_element() for _ in range(l)]
    return matrix(F, l, l, lambda i, j: first_row[(j - i) % l])

def random_block_circulant_matrix(F, m, n, l):
    """
    F : finite field (e.g. GF(2), GF(2^8), GF(65537), …)
    m : number of block rows
    n : number of block columns
    l : block size           → full matrix has (m*l) × (n*l) entries

    Returns a matrix whose (i, j)-th block is a random l×l circulant matrix.
    The blocks are *independent*, so the big matrix is “block-circulant”
    in the usual cryptographic sense (circulant inside each block).
    """
    blocks = [[random_circulant_matrix(F, l) for _ in range(n)]   # n blocks per row
              for _ in range(m)]                                  # m such rows
    return block_matrix(m, n, blocks)

Q_list = []
for i in range(m):
    # mat = random_block_circulant_matrix(F,n,n,l)
    mat = random_matrix(F,n*l,n*l,density=1)
    Q_list.append(mat + mat.transpose())

"""
Wedge simulator
"""

# helper: kl×kl minors --------------------------------------------------------
def minor(I):
    I = tuple(sorted(I))                # ① make sure the key is sorted
    return X.matrix_from_columns(I).det()

minors = {I: minor(I) for I in combinations(range(n*l), k*l)}

# sign
def wedge_sign(I, p, q):
    """
    I  – k-tuple/list of the minor columns (sorted ascending)
    p,q – the two indices coming from Q, assume p<q
    """
    s = sum(i > p for i in I) + sum(i > q for i in I)
    return -1 if s & 1 else 1


# --- 2. equations ----------------------------------------------------------
eqns = []
for QQ in Q_list:                              # QQ is an n×n skew matrix
    for J in combinations(range(n*l), k*l+2):      # every (k+2)-subset
        coeff = 0
        for p, q in combinations(J, 2):        # choose Q's two indices
            if p < q and QQ[p, q] != 0:
                I = tuple(i for i in J if i not in (p, q))
                coeff += wedge_sign(I, p, q) * QQ[p, q] * minor(I)
        eqns.append(coeff)                     # must vanish

from numbers import Integral
eqns = [f for f in eqns if f != 0 and not isinstance(f, Integral)]

# 6) Display info
print(f"Generated {m} random skew-symmetric matrices Q_1,...,Q_{m} (n={n*l})")
for idx, Q in enumerate(Q_list):
    print(f"\nQ_{idx+1} of rank {Q.rank()} =")
    color_matrix(Q)

# print(f"\nPolynomial ring R with {k*n} variables: {R.variable_names()}")
print(f"Number of equations generated: {len(eqns)} = {m} × binom({n*l}, {k*l+2})")

# print("\nPrint equations:")
# for eq in eqns:
#     print(eq,"\n")
"""
MACAULAY
"""

# Step 1: Define target degree
target_degree = k*l

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
# print(M)

# Step 5: Number of linear independent eqs
rank = M.rank()
print(f"\nMacaulay matrix rank = {rank} = linearly independent equations")
print(f"\nParameters : v+o = {n}, v = {k}, m = {m}, l = {l}")
print(f"In theory, \nNumber of variables = {len(names)}; binom(v+o,o)^l = {binomial(n,k)**l}; and m x binom(lv+lo,lv+2) = {binomial(l*n,l*k+2)*m}; and m x binom(lv+lo,lv+2l) = {binomial(l*n,l*k+2*l)*m}")
