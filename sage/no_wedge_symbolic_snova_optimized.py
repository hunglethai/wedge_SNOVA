# ─────────────────────  SNOVA – Plücker form  ─────────────────────
from sage.all import *
from itertools import combinations, islice, count, product
from ..utils.functions import lift_matrix

# 0) parameters
n, k, m, l, q = 4, 2, 1, 2, 4         # params
F         = GF(q)
set_random_seed(20250615)

# 1) pick an S matrix and build the extension 𝔽_S
S = load(os.path.join(os.path.dirname(__file__), "S_matrices.sobj"))[l-2]
K = PolynomialRing(F,'x')
F_S = F.extension(K(S.charpoly()), 's')
φ   = F_S.hom([S], MatrixSpace(F, l))   # coefficient embedding

# 2) Plücker variables  P_I  for kl–subsets of {0,…,N−1}
confirmed_num_var = binomial(n,k)**l # Proven number of variables
N, v = n * l, k * l

# subs  = list(combinations(range(N), v))
# Pring = PolynomialRing(F, [f'P{"".join(map(str,I))}' for I in subs])
# P     = dict(zip(subs, Pring.gens()))
# cols = len(P)

subs_all = list(combinations(range(N), v))          #  C(N,v) indices
def survives_snova(I):
    """
    A maximal minor det(M[:,I]) can be non-zero for an
    (l×l) block-circulant matrix  ⇔  I contains **exactly k columns
    in every shift  s = col mod l**.  See SNOVA Prop. 1.
    """
    cnt = [0] * l
    for j in I:
        cnt[j % l] += 1
    return all(c == k for c in cnt)

keep  = [I for I in subs_all if survives_snova(I)]      # size (n choose k)^l
kill  = [I for I in subs_all if not survives_snova(I)]  # the vanishing ones


# polynomial ring *only* for the surviving minors
Pring = PolynomialRing(F, [f'P{"".join(map(str,I))}' for I in keep])
P     = dict(zip(keep, Pring.gens()))
cols = len(P)

# map every vanishing minor to the zero element of the same ring
zero = Pring(0)
for I in kill:
    P[I] = zero

# 3) random full-rank alternating matrices over 𝔽_S, lifted to F
def skew_mats(field_ext, dim, how_many):
    alt  = lambda: random_matrix(field_ext, dim) - random_matrix(field_ext, dim).transpose()
    gen  = (M for M in (alt() for _ in count()) if M.rank() == dim)
    return [lift_matrix(M, φ) for M in islice(gen, how_many)]

# random big full-rank alternating matrices over 𝔽 
def big_skew_mats(field, dim_big, how_many):
    """Return <how_many> full-rank dim_big×dim_big skew matrices over <field>."""
    alt  = lambda: random_matrix(field, dim_big) - random_matrix(field, dim_big).transpose()
    good = (A for A in (alt() for _ in count()) if A.rank() == dim_big)
    return list(islice(good, how_many))

# SNOVA matrices
def snova_public_matrices_list(n, v, m, l, S,
                               P_list=None, *,
                               random_style="uov",  # "uov" or "full"
                               seed=None):
    """
    Build the family  Q_{i,j,k} = Λ_{S^j} · P_i · Λ_{S^k}
    and return it as a *list* [Q_(1,0,0), Q_(1,0,1), … , Q_(m,l-1,l-1)].

    Parameters
    ----------
    n : int      # total variables (v + o)
    v : int      # vinegars   (v)
    m : int      # number of equations
    l : int      # extension degree
    q : int      # size of GF(q)
    S : l×l matrix over GF(q) with irreducible characteristic polynomial
    P_list : optional list of m central matrices P_i (size nl × nl)
    random_style : "uov" (default)  → zero oil–oil block,
                   "full"           → completely random matrices
    seed : int or None  → reproducible randomness

    Returns
    -------
    list[Matrix]  length  m · l²   (each matrix is (nl)×(nl) over GF(q))
    """
    # --------------- basic objects
    R  = MatrixSpace(F, l, l)
    Ln = n * l

    Λ = lambda Q: block_diagonal_matrix([Q] * n)      # Λ_Q
    ΛS = [Λ(S**j) for j in range(l)]                  # Λ_{S^0}, … , Λ_{S^{l-1}}

    # --------------- draw random central matrices if none supplied
    if P_list is None:
        o = n - v

        def rand_block():  # helper for one random block over R
            return R.random_element()

        if random_style == "uov":
            # --- 1. central F_i with zero oil–oil block
            def rand_F():
                F11 = block_matrix([[rand_block() for _ in range(v)] for _ in range(v)])
                F12 = block_matrix([[rand_block() for _ in range(o)] for _ in range(v)])
                F21 = block_matrix([[rand_block() for _ in range(v)] for _ in range(o)])
                ZZ  = block_matrix([[R.zero()     for _ in range(o)] for _ in range(o)])
                return block_matrix([[F11, F12],
                                     [F21, ZZ ]])
            F_list = [rand_F() for _ in range(m)]

            # --- 2. random upper-triangular T  (I_v  T_vo; 0  I_o)
            T_vo = block_matrix([[rand_block() for _ in range(o)] for _ in range(v)])
            I_v  = block_diagonal_matrix([R.identity_matrix()]*v)
            I_o  = block_diagonal_matrix([R.identity_matrix()]*o)
            Z_vo = zero_matrix(F, o*l, v*l)
            T    = block_matrix([[I_v, T_vo],[Z_vo, I_o]])

            # --- 3. hide the oil–oil zero block
            P_list = [(T.transpose()) * F * T for F in F_list]
        elif random_style == "full":
            def random_P():
                blocks = [[rand_block() for _ in range(n)] for _ in range(n)]
                return block_matrix(blocks)
            P_list = [random_P() for _ in range(m)]
        else:
            raise ValueError("random_style must be 'uov' or 'full'")

    if len(P_list) != m:
        raise ValueError("need exactly m central matrices")

    # --------------- build and collect Q_{i,j,k} in the requested order
    Q_list = []
    for i, P in enumerate(P_list, start=1):          # 1-based i for clarity
        for j, Λj in enumerate(ΛS):                  # j = 0 … l−1
            for k, Λk in enumerate(ΛS):              # k = 0 … l−1
                Q_list.append(Λj * P * Λk)

    return Q_list

Q_list = skew_mats(F_S, n, m)           # Lifted matrix, each becomes an N×N skew matrix over F
# Q_list = big_skew_mats(F, N, m)             # random full-rank alternating matrices *directly* of size N×N 
# Q_list = snova_public_matrices_list(n,k,m,l,S, random_style = "uov") # Snova structure 

# 4) linear relations (Plücker syzygies for each Q_i)
eqns = [ sum( (-1)**((J.index(b)-J.index(a)-1)&1) * Q[a,b] *
              P[tuple(i for i in J if i not in (a,b))]
              for a,b in combinations(J,2) )
         for Q in Q_list
         for J in combinations(range(N), v+2) ]

print(f'[info] built {len(eqns)} = {m*l*l}·C({N},{v+2}) linear relations '
      f'over {len(P)} Plücker vars (C({N},{v}))')

# 5) Macaulay matrix (degree 1 here, so we just collect coefficients)
M = matrix(F, len(eqns), cols,
           lambda r,c: eqns[r].coefficient(Pring.gens()[c]))

rank = M.sparse_matrix().rank()

#Hypothesis
def compute_sum_1(m, v, o, ell):
    total = 0
    upper = floor(o * ell / 2+1)
    totals_by_i = []
    
    for i in range(0, upper + 1):
        coeff = (-1)**i * binomial(m*ell + i - 1, i)
        inner_sum = 0

        for vec in IntegerVectors(2 * i, ell):
            term_product = prod(binomial(v + o, v + a_j) for a_j in vec)
            inner_sum += term_product

        total += coeff * inner_sum
        totals_by_i.append((i, confirmed_num_var - total))  # Save the current index and cumulative sum

    return totals_by_i


def compute_sum_2(m, v, o, ell):
    total = 0
    upper = floor(o * ell / 2+1)
    totals_by_i = []
    
    for i in range(0, upper + 1):
        coeff = (-1)**i * binomial(m + i - 1, i)
        inner_sum = 0

        for vec in IntegerVectors(2 * i, ell):
            term_product = prod(binomial(v + o, v + a_j) for a_j in vec)
            inner_sum += term_product

        total += coeff * inner_sum
        totals_by_i.append((i, confirmed_num_var - total))  # Save the current index and cumulative sum

    return totals_by_i

def compute_sum_3(m, v, o, ell):
    upper = floor(o / 2)+1
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
        totals_by_I.append((I, confirmed_num_var - total))  # Store a tuple: (I, cumulative sum)

    return totals_by_I

print(f"\nMacaulay matrix rank = {rank} = linearly independent equations")
print(f"\nv+o = {n}, v = {k}, m = {m}, l = {l}, number of variables (proven) = {confirmed_num_var}")

# Check Hypothesis 1
hyp1 = compute_sum_1(m, k, n - k, l)
print(hyp1)

# Check Hypothesis 2
hyp2 = compute_sum_2(m, k, n - k, l)
print(hyp2)

# Check Hypothesis 3
hyp2 = compute_sum_3(m, k, n - k, l)
print(hyp2)