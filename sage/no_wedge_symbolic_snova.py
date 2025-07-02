from sage.all import *
from ..utils.functions import *
from itertools import combinations, product

# ----------------------- parameters -----------------------
l = 2           # Extension
n = 4           # ambient dimension (n ≥ k+2)
k = 2           # number of unknown vectors v_1,…,v_k, dimension of the complement space of oil space
m = 1          # number of alternating matrices Q_i, number of eqs
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
Wedge
"""
# ----------------------------------------------------------

# 1) Polynomial ring with k·n·l indeterminates
names = [f"x{r}{i}{j}" for r in range(k) for i in range(n) for j in range(l)]
R = PolynomialRing(F, names)
x = R.gens()

# 2) Helper to access variable for (r, i, j)
def var(r, i, j):
    idx = (r * n + i) * l + j
    return x[idx]

# 3) Create blocks: each is an l x l diagonal matrix (same for circulant matrix or SNOVA)
blocks = [[None for _ in range(n)] for _ in range(k)]
for r in range(k):
    for i in range(n):
        diag_entries = [var(r, i, j) for j in range(l)]
        blocks[r][i] = matrix(R, l, lambda a, b: diag_entries[a] if a == b else 0)

# 4) Assemble block matrix
X = block_matrix(k, n, blocks)
# color_matrix(X)

# 2) Random matrices
def gen_poly_random(F, n,m):
    matrices = []
    for _ in range(m):
        A = Matrix(F, n, n, 0)
        for i in range(n):
            for j in range(i+1, n):
                val = F.random_element()
                A[i, j] = val
                A[j, i] = -val  # For alternating matrix
        matrices.append(A)
    
    return matrices

# Gen poly
Q_list_base = gen_poly_random(F_S,n,m)

# Lift all matrices in Q_list_base
Q_list = []
for mat in Q_list_base:
    Q_list.append(lift_matrix(mat,phi))

# helper: kl×kl minors --------------------------------------------------------
def minor(I):
    """det(X[:, I]) for a k-tuple I of column indices"""
    return X.matrix_from_columns(I).det()

minors = {I: minor(I) for I in combinations(range(n*l), k*l)}

# --- 2. build equations for each quadratic form Q -------------------------
eqns = []
for QQ in Q_list:                    # QQ: sage n×n skew matrix
    # loop over every (k+2)-subset of {0,…,n-1}
    for J in combinations(range(n*l), k*l+2):
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
print(f"Generated {m} random skew-symmetric matrices Q_1,...,Q_{m} (n={n*l})")
for idx, Q in enumerate(Q_list):
    print(f"\nQ_{idx+1} is full rank : {Q.rank()==n*l}")
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
M_sparse = M.sparse_matrix()
rank = M_sparse.rank()

#Hypothesis
def compute_sum_1(m, v, o, ell):
    total = 0
    upper = floor(o * ell / 2)
    totals_by_i = []
    
    for i in range(0, upper + 1):
        coeff = (-1)**i * binomial(m*ell + i - 1, i)
        inner_sum = 0

        for vec in IntegerVectors(2 * i, ell):
            term_product = prod(binomial(v + o, v + a_j) for a_j in vec)
            inner_sum += term_product

        total += coeff * inner_sum
        totals_by_i.append((i, total))  # Save the current index and cumulative sum

    return totals_by_i


def compute_sum_2(m, v, o, ell):
    upper = floor(o / 2)
    total = 0
    grouped_terms = {}
    totals_by_I = []

    # Group terms by total I = sum of multi_index
    for multi_index in product(range(0, upper + 1), repeat=ell):
        I = sum(multi_index)
        term = (-1)**I
        for i_j in multi_index:
            term *= binomial(m + i_j - 1, i_j) * binomial(v + o, v + 2 * i_j)
        grouped_terms[I] = grouped_terms.get(I, 0) + term

    # Accumulate total and store for each I
    for I in sorted(grouped_terms):
        total += grouped_terms[I]
        totals_by_I.append((I, total))  # Store a tuple: (I, cumulative sum)

    return totals_by_I

print(f"\nMacaulay matrix rank = {rank} = linearly independent equations")
print(f"\nv+o = {n}, v = {k}, m = {m}, l = {l}")

# Check Hypothesis 1
hyp1 = compute_sum_1(m, k, n - k, l)
match1 = next(((i, total) for i, total in hyp1 if total == rank), None)

# Check Hypothesis 2
hyp2 = compute_sum_2(m, k, n - k, l)
match2 = next(((i, total) for i, total in hyp2 if total == rank), None)

# Report
if match1 and not match2:
    print(f"\n✅ Hypothesis 1 is verified: i = {match1[0]}, rank = {match1[1]}")
elif match2 and not match1:
    print(f"\n✅ Hypothesis 2 is verified: I = {match2[0]}, rank = {match2[1]}")
elif match1 and match2:
    print(f"\n✅ Both hypotheses are verified!")
    print(f"  Hypothesis 1: i = {match1[0]}, rank = {match1[1]}")
    print(f"  Hypothesis 2: I = {match2[0]}, rank = {match2[1]}")
else:
    print("\n❌ Neither hypothesis matched the computed rank.")