from sage.all import *
from ..utils.functions import *
from itertools import combinations

# ----------------------- parameters -----------------------
n = 6           # ambient dimension (n ≥ k+2)
k = 2           # number of unknown vectors v_1,…,v_k, dimension of the complement space of oil space
m = 10           # number of alternating matrices Q_i, number of eqs
q = 16
F = GF(q)       # Base Finite field
seed = 20250615
set_random_seed(seed)

# ----------------------------------------------------------

# 1) Polynomial ring with k·n indeterminates
names = [f"x{r}{i}" for r in range(k) for i in range(n)]
R = PolynomialRing(F, names)
x = R.gens()

# X is the k×n unknown-coordinate matrix
X = matrix(R, k, n, x)

"""
UOV SETUP
"""
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
    
    return M

# or Random matrices Setup
def gen_poly_random(n: int,m: int,F: FiniteField):
    matrices = []
    while len(matrices) != m:
        A = Matrix(F, n, n, 0)
        for i in range(n):
            for j in range(i+1, n):
                val = F.random_element()
                A[i, j] = val
                A[j, i] = -val  # For alternating matrix
        if A.rank() == n:
            matrices.append(A)
    return matrices

# Q_list = gen_poly(n,m,F,n-k)
Q_list = gen_poly_random(n,m,F)
"""
EXTERIOR ALGEBRA SETUP
"""
# helper: k×k minors --------------------------------------------------------
def minor(I):
    """det(X[:, I]) for a k-tuple I of column indices"""
    return X.matrix_from_columns(I).det()

minors = {I: minor(I) for I in combinations(range(n), k)}

# --- 2. build equations for each quadratic form Q -------------------------
eqns = []
for QQ in Q_list:                    # QQ: sage n×n skew matrix
    # loop over every (k+2)-subset of {0,…,n-1}
    for J in combinations(range(n), k+2):
        coeff = 0
        # choose the two indices contributed by Q
        for a, b in combinations(J, 2):
            if a < b:               # a, b are positions in the full set
                # sign coming from wedge ordering
                sign = (-1) ** (J.index(b) - J.index(a) - 1)
                I = tuple(i for i in J if i not in (a, b))
                coeff += sign * QQ[a, b] * minors[I]
        eqns.append(coeff)          # this must be 0


# 6) Display info
print(f"Generated {m} random skew-symmetric matrices Q_1,...,Q_{m} (n={n})")
for idx, Q in enumerate(Q_list):
    print(f"\nQ_{idx+1} is full rank = {Q.rank()==n}")
    color_matrix(Q)

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

"""
MACAULAY RANK COMPUTE
"""

# Step 5: Number of linear independent eqs
M_sparse = M.sparse_matrix()
rank = M_sparse.rank()
#Hypothesis
def compute_sum(m, v, o):
    total = 0
    upper = floor(o / 2)
    totals_by_i = []
    
    for i in range(0, upper + 1):
        coeff = (-1)**i * binomial(m + i - 1, i) * binomial(v + o, v + 2*i)
        total += coeff
        totals_by_i.append((i, total))  # Save the current index and cumulative sum

    return totals_by_i

print(f"\nMacaulay matrix rank = {rank} = linearly independent equations")
print(f"\nv+o = {n}, v = {k}, m = {m}")
print(f"\nRank hypothesis = {compute_sum(m,k,n-k)}")