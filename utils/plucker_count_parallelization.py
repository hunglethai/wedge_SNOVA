# Converted from the original MAGMA script to Python-compatible SageMath code.
# We parallelize the Macaulay filling step and the minors computing step using multi cores
import multiprocessing
print("Available CPU cores:", multiprocessing.cpu_count())
from sage.all import *
import time
from tqdm import tqdm
import multiprocessing

###############################################################################
# Functions                                                           
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

# Parallel processing
def _process_one_poly(args):
    i, poly, monom_dict = args
    row_dict = {}
    for coeff, mon in zip(poly.coefficients(), poly.monomials()):
        j = monom_dict.get(mon)
        if j is None:
            continue
        row_dict[j] = coeff
    return (i, row_dict)

###############################################################################
# Parallel Macaulay-matrix construction                                                  
###############################################################################

def macaulay(system, K):
    t0 = time.process_time()
    monoms = set()
    for P in tqdm(system, ncols= 100, desc="Collecting monomials"):
        monoms.update(P.monomials())
    monoms = sorted(monoms, key=lambda m: (m.degree(), m), reverse=True)
    monom_dict = {m: i for i, m in enumerate(monoms)}
    print(f"=> # eq = {len(system)}, # monos = {len(monoms)}, degrees = {sorted({m.degree() for m in system})}")
    print(f"=> time to create {{monos}} : {time.process_time() - t0:.1f} s")

    t1 = time.process_time()
    n_workers = multiprocessing.cpu_count()

    # Build sparse matrix data structure
    from collections import defaultdict
    sparse_data = defaultdict(dict)  # Will be sparse_data[i][j] = value

    if n_workers == 1:
        # Serial processing
        for i, poly in enumerate(tqdm(system, ncols= 100, desc="Filling Macaulay matrix (serial)")):
            for coeff, mon in zip(poly.coefficients(), poly.monomials()):
                j = monom_dict.get(mon)
                if j is None:
                    continue
                sparse_data[i][j] = coeff
    else:
        # Parallel processing
        from multiprocessing import Pool
        args_list = [(i, poly, monom_dict) for i, poly in enumerate(system)]
        chunksize = min(20, max(1, len(system) // (n_workers * 4)))
        
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_one_poly, args_list, chunksize=chunksize),
                total=len(system),
                ncols=100,
                desc=f"Filling Macaulay matrix (parallel, {n_workers} workers)"
            ))
        
        for i, row_dict in results:
            sparse_data[i] = row_dict

    # Build sparse matrix from collected data
    Mac = matrix(K, len(system), len(monoms), 0, sparse=True)
    for i, col_dict in sparse_data.items():
        for j, coeff in col_dict.items():
            Mac[i, j] = coeff
            
    print(f"=> time to create Mac : {time.process_time() - t1:.1f} s")

    t2 = time.process_time()
    # Convert to dense if matrix is small; use sparse otherwise
    if Mac.nrows() * Mac.ncols() < 10**7:  # ~10M elements
        Mac_dense = Mac.dense_matrix()
        r = Mac_dense.rank()
    else:
        print("=> Matrix too large for dense rank computation. Using sparse rank (may be slower)")
        r = Mac.rank()
    print(f"=> Mac : {Mac.nrows()} × {Mac.ncols()}, rank = {r}")
    print(f"=> time to compute rank : {time.process_time() - t2:.1f} s")

    return monoms, Mac

###############################################################################
# Parallel minor computation
###############################################################################

def compute_minor(args):
    M, cols = args
    return M.matrix_from_columns(cols).det()

def compute_minors_parallel(M, k, n_workers):
    from itertools import combinations 
    n_cols = M.ncols()
    total_minors = binomial(n_cols, k)
    col_indices = list(combinations(range(n_cols), k))
    
    if n_workers == 1:
        return [M.matrix_from_columns(cols).det() for cols in tqdm(col_indices, ncols=100, desc="Computing minors (serial)")]
    
    chunksize = min(100, max(1, total_minors // (n_workers * 4)))
    args_list = [(M, cols) for cols in col_indices]
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        minors = list(tqdm(
            pool.imap(compute_minor, args_list, chunksize=chunksize),
            total=total_minors,
            ncols=100,
            desc=f"Computing minors (parallel, {n_workers} workers)"
        ))
    
    return minors


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
    for k in tqdm(range(v + o), ncols= 100, desc="Building matrix M"):
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
    k_size = v * l
    n_workers = multiprocessing.cpu_count()
    minors = compute_minors_parallel(M, k_size, n_workers)
    print(f"=> # minors = {len(minors)} (theory = {n_theoretical_minors})")
    print(f"=> time to compute minors : {time.process_time() - t_min:.1f} s")

    print("=> Macaulay matrix")
    monoms, Mac = macaulay(minors, K)

    hung_estimate = binomial(v+o,v)**l
    ray_estimate = binomial((v + o) * l, v * l) / l + ((l - 1) / l) * binomial(v + o, o)
    daniel_estimate = ray_estimate - binomial(v + o, 2)

    print(f"=> Hung's estimate = {hung_estimate}")
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