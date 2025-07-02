from sage.all import *
import os

# Parameters to test
params = [
    (2, GF(4)),
    (3, GF(4)),
    (4, GF(4)),
    (5, GF(4)),
]

matrices = []

def random_symmetric_irred_charpoly_matrix(F, l, *,
                                           max_attempts=10000,
                                           seed=None):
    r"""
    Return a random symmetric `l × l` matrix over the finite field `F`
    whose characteristic polynomial is **irreducible of degree `l`.**

    Parameters
    ----------
    F :  a finite field  (instance of ``GF(q)``)
    l :  positive int    (the matrix size / desired polynomial degree)
    max_attempts : int, optional
        Stop and raise ``RuntimeError`` after this many failed tries.
    seed : int or None, optional
        Pass an integer for reproducible randomness.

    Returns
    -------
    Matrix over F   -- symmetric, non-singular, char-poly irreducible.

    Raises
    ------
    RuntimeError    if no suitable matrix is found after ``max_attempts``.
    """
    if seed is not None:
        set_random_seed(seed)

    for attempt in range(1, max_attempts + 1):
        # 1) build a random symmetric matrix
        M = Matrix(F, l, l, 0)
        for i in range(l):
            for j in range(i, l):
                a = F.random_element()
                M[i, j] = a
                M[j, i] = a

        # 2) quick rejection: determinant must be non-zero
        if M.is_singular():
            continue

        # 3) check irreducibility of the characteristic polynomial
        charpoly = M.charpoly()
        if charpoly.is_irreducible():
            return M

    raise RuntimeError(f"no symmetric matrix with irreducible "
                       f"char-poly found after {max_attempts} attempts")

for l, F in params:
    M = random_symmetric_irred_charpoly_matrix(F, l)
    matrices.append(M)

# Save .sobj file to the same directory as this script
output_path = os.path.join(os.path.dirname(__file__), "S_matrices.sobj")
save(matrices, output_path)
