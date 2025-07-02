# plucker_count_sage.py
# Converted from the original MAGMA script to Python-compatible SageMath code.

from sage.all import *
import time
from tqdm import tqdm

###############################################################################
# Helper constructions                                                           
###############################################################################

def _base_field(q):
    if q is None:
        return QQ
    return GF(q)


def _poly_ring(K):
    R = PolynomialRing(K, 'x')
    x = R.gen()
    return R, x


def _quotient_ring(K, l):
    R, x = _poly_ring(K)
    T = R.quotient(x**l - 1, names='X')
    X = T.gen()
    return T, X


def poly_to_seq(poly, l):
    parent = poly.parent()
    coeffs = parent.lift(poly).coefficients(sparse=False)
    coeffs += [parent.base_ring()(0)] * (l - len(coeffs))
    return coeffs


def vec_of_vars_to_matrix(tab, X, l, R):
    M = matrix(R, l, l, 0)
    for j in range(l):
        rows = [poly_to_seq(X**(j) * X**i, l) for i in range(l)]
        M += tab[j] * matrix(R, rows)
    return M

###############################################################################
# Macaulay-matrix construction                                                  
###############################################################################

def macaulay(system, K):
    t0 = time.process_time()
    monoms = set()
    for P in tqdm(system, desc="Collecting monomials"):
        monoms.update(P.monomials())
    monoms = sorted(monoms, key=lambda m: (m.degree(), m), reverse=True)
    print(f"=> # eq = {len(system)}, # monos = {len(monoms)}, degrees = {sorted({m.degree() for m in system})}")
    print(f"=> time to create {{monos}} : {time.process_time() - t0:.1f} s")

    t1 = time.process_time()
    Mac = matrix(K, len(system), len(monoms), 0)
    for i, P in enumerate(tqdm(system, desc="Filling Macaulay matrix")):
        for coeff, mon in zip(P.coefficients(), P.monomials()):
            j = monoms.index(mon)
            Mac[i, j] = coeff
    print(f"=> time to create Mac : {time.process_time() - t1:.1f} s")

    t2 = time.process_time()
    r = Mac.rank()
    print(f"=> Mac : {Mac.nrows()} × {Mac.ncols()}, rank = {r}")
    print(f"=> time to compute rank : {time.process_time() - t2:.1f} s")

    return monoms, Mac

###############################################################################
# Main driver                                                                   
###############################################################################

def plucker_count(v=2, o=1, l=3, q=3):
    K = _base_field(q)
    T, X = _quotient_ring(K, l)
    nvars = v * (v + o) * l
    var_names = [f"x_{i}" for i in range(1, nvars + 1)]
    R = PolynomialRing(K, var_names, order='degrevlex')
    vars = R.gens()

    M = matrix(R, v * l, 0)
    for k in tqdm(range(v + o), desc="Building matrix M"):
        C = matrix(R, 0, l)
        for j in range(v):
            start = k * (v * l) + j * l
            tab = [vars[start + i] for i in range(l)]
            C = C.stack(vec_of_vars_to_matrix(tab, X, l, R))
        M = M.augment(C)

    print(f"=> v = {v}, o = {o}, l = {l}")
    print(f"=> matrix M of size {M.nrows()} × {M.ncols()}","\n",M)
    n_theoretical_minors = binomial((v + o) * l, v * l)
    print(f"=> # minors in theory = {n_theoretical_minors}")

    t_min = time.process_time()
    minors = list(tqdm(M.minors(v * l), desc="Computing minors"))
    print(f"=> # minors = {len(minors)} (theory = {n_theoretical_minors})")
    print(f"=> time to compute minors : {time.process_time() - t_min:.1f} s")

    print("=> Macaulay matrix")
    monoms, Mac = macaulay(minors, K)

    theoretical_upper_bound = binomial(v+o,v)**l
    ray_estimate = binomial((v + o) * l, v * l) / l + ((l - 1) / l) * binomial(v + o, o)
    daniel_estimate = ray_estimate - binomial(v + o, 2)

    print(f"=> Theoretical bound = {theoretical_upper_bound}")
    print(f"=> Ray's estimate = {ray_estimate}")
    print(f"=> Daniel's estimate = {daniel_estimate}")

    return {
        "matrix_M": M,
        "minors": minors,
        "macaulay": Mac,
        "ray_estimate": ray_estimate,
        "daniel_estimate": daniel_estimate,
    }

###############################################################################
# CLI entry point                                                              
###############################################################################

if __name__ == "__main__":
    try:
        v = int(input("v = ") or 2)
        o = int(input("o = ") or 1)
        l = int(input("l = ") or 3)
        q = int(input("q = ") or 3)
    except (ValueError, EOFError):
        print("Invalid input – falling back to the defaults  v=2, o=1, l=3, q=3.")
        v, o, l, q = 2, 1, 3, 3
    plucker_count(v, o, l, q)
