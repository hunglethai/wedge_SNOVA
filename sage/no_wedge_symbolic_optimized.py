# ─────────────────── basics ─────────────────────
from sage.all import *
from itertools import combinations, islice

n, v, m, q   = 10, 5, 2, 2          # parameters
F            = GF(q)
set_random_seed(20250615)

# ── 1. Plücker variables ⟨ P_I ⟩ ───────────────────
subs   = list(combinations(range(n), v))
Pring  = PolynomialRing(F, [f'P{"".join(map(str,I))}' for I in subs])
P      = dict(zip(subs, Pring.gens()))            # I  ↦  P_I

# ── 2. random full-rank alternating matrices ──────
def alt_mats(field, dim, how_many):
    mats = []
    while len(mats) < how_many:
        A = random_matrix(field, dim) - random_matrix(field, dim).transpose()
        if A.rank() == dim:
            mats.append(A)
    return mats

Q_list = alt_mats(F, n, m)

# ── 3. build linear relations ─────────────────────────────
eqns = [ sum( (-1)**((J.index(b)-J.index(a)-1)&1) * Q[a,b] *
              P[tuple(i for i in J if i not in (a,b))]
              for a,b in combinations(J,2) )
         for Q in Q_list
         for J in combinations(range(n), v+2) ]

print(f'[info] built {len(eqns)} linear relations over {len(P)} variables')

# ── 4. Macaulay coefficient matrix ────────────────────
M = matrix(F, len(eqns), len(P),      # dense constructor
           lambda r,c: eqns[r].coefficient(Pring.gens()[c]))

rank = M.sparse_matrix().rank()
print(f'[result] rank(M) = {rank}')

# ── 5. theoretical check  ───────────────────────────
def theo(m,v,o):                                     # o = n-v
    return [binomial(n,v) -
            sum((-1)**i * binomial(m+i-1,i) * binomial(v+o, v+2*i)
                for i in range(t+1))
            for t in range(o//2+1)]

print('rank hypothesis =', theo(m, v, n-v))
