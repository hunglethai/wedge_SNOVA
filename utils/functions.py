from sage.all import *
from tqdm import tqdm
from itertools import combinations

"""
EXTERIOR ALGEBRA TOOLS
"""
def alternating_to_wedge(F: FiniteField, A: Matrix):
    """
    Convert an n×n alternating matrix A over a field K to the exterior 2-form ∑_{i<j} A[i,j] e_i∧e_j
    in the exterior algebra Λ(V) with generators e0,…,e{n-1}.

    INPUT:
        F: Finite field
        A  -- square Sage matrix; must satisfy A + A.transpose() == 0

    OUTPUT:
        omega -- element of degree 2 in Λ(V)

    """
    # Basic sanity check (raise if A is not alternating)
    if A.nrows() != A.ncols():
        raise ValueError("Matrix must be square.")
    if A + A.transpose() != 0:
        raise ValueError("Matrix must be alternating (skew-symmetric).")
    
    n = A.nrows()

    # Exterior algebra Λ(V) with default generators e0,…,e{n-1}
    E = ExteriorAlgebra(F, n)
    e = E.gens()

    # Build the 2-form
    omega = E.zero()
    for i in range(n):
        for j in range(i+1, n):
            omega += A[i, j] * (e[i] * e[j])   # * is ∧ in Sage

    return omega

def minor(F: FiniteField, matrix: Matrix, cols):
    """
    Compute maximal minors from a list of columns
    """
    return matrix.matrix_from_columns(list(cols)).determinant()

"""
PRINTING TOOLS
"""

# Print Matrix with Symbolic Labels
def print_labeled_matrix(M: Matrix, label_map):
    """
    Print matrix with extension field coefficients
    """
    max_label_len = max(len(label) for label in label_map.values())
    for row in M.rows():
        aligned_row = [
            label_map[e].rjust(max_label_len) for e in row
        ]
        print("  [", ", ".join(aligned_row), "]")

# Visualize matrices where non-zero elements are represented by black dots and zero elements by white dots
def color_matrix(matrix: Matrix):
    """
    Function to print a large matrix in a compact way, using:
    - Filled squares (■) for non-zero elements.
    - Empty squares (□) for zero elements.
    
    Args:
    matrix (Matrix): A matrix over a finite field.

    Prints:
    A compact grid-based representation of the matrix.
    """
    for i in range(matrix.nrows()):
        row_str = ""
        for j in range(matrix.ncols()):
            if matrix[i, j] == 0:
                row_str += "⬚ "
            else:
                row_str += "■ "  # Filled square for non-zero
        print(row_str.strip())  # Print each row of squares

"""
UOV tools
"""

# Generate a random full rank matrix
def random_full_rank_matrix(F: FiniteField, m: int, n: int):
    """
    Generate a random matrix of size m x n which is full rank
    """
    while True:
        # Generate a random m x n matrix over the specified field (default is rational field QQ)
        A = random_matrix(F, m, n)
        
        # Check if the rank is full (i.e., min(m, n))
        if A.rank() == min(m, n):
            return A

def random_upper_triangular_matrix(F: FiniteField, m: int, n: int) -> Matrix:
    """
    Generates a random upper-triangular matrix of size m x n over the specified field.
    
    m: number of rows
    n: number of columns
    F: Finite field
    
    Returns:
        A random upper-triangular matrix of size m x n.
    """
    # Initialize an empty matrix of size m x n
    M = Matrix(F, m, n)
    
    # Fill in the upper triangular part with random elements
    for i in range(m):
        for j in range(i, n):
            M[i, j] = F.random_element()  # Fill the upper triangular part with random values
    
    # Elements below the diagonal are automatically zero, as we do not modify them
    
    return M

# Generate oil subspace matrix i.e. the secret key
def generate_oil_subspace_matrix(F: FiniteField, d: int, n: int) -> Matrix:
    """
    F: Finite Field
    d: dimension of the oil subspace
    n: oil and vinegar variables i.e. o + v i.e. dimension fo the ambient space
    
    Return 
    A matrix which is the basis of the oil subspace
    """
    A = random_full_rank_matrix(F,d,n-d)
    I = matrix.identity(F,d)
    O = block_matrix(1,2,[A,I], subdivise = True)
    
    return A.transpose(), O.transpose()

# "Upper()" function as in UOV specs
def upper(F: FiniteField, M: Matrix) -> Matrix:
    """
    Skew symmetric matrix A means that A + A^t = 0
    Returns the unique upper triangular matrix X such that X + M is skew-symmetric.
    
    M: A square matrix
    """
    # Step 1: Calculate M + M^T (transpose of M)
    S = -M - M.transpose()
    
    # Step 2: Create a matrix X with the lower triangular part of M + M^T
    n = M.nrows()
    X = Matrix(F, n, n)
    
    for i in range(n):
        # Force the diagonal to be zero 
        X[i, i] = -M[i, i]
        for j in range(i+1,n):
            # Copy upper triangular part 
            X[i, j] = S[i, j]
    
    # Return the matrix X
    return X

# Check if a matrix is indeed skew-symmetric
def is_skew_symmetric(F: FiniteField ,M: Matrix) -> bool:
    """
    Verify a matrix is skew-symmetric
    """
    # Get the number of rows (and columns, assuming it's square)
    n = M.nrows()
    
    # Iterate through the matrix and check the skew-symmetric condition
    for i in range(n):
        for j in range(i, n):
            if M[i, j] != -M[j, i]:
                return False
    return True

# Generate a matrix that vanishes on the oil space i.e. public key matrices
def generate_public_matrices(F: FiniteField, m: int, n: int, O: Matrix) -> list:
    """
    F: Finite Field
    m: number of equations i.e. number of matrices
    n: oil + vinegar variables i.e. o + v i.e. dimension of the matrices
    O: oil subspace of size (n-d) x d
    
    Return 
    A list of matrices which vanish on O
    """
    d = O.ncols()
    list_P = []
    
    for i in range(m):
        P_1 = random_upper_triangular_matrix(F,n-d,n-d)
        P_2 = random_full_rank_matrix(F,n-d,d)
        P_3 = upper(F,O.transpose()*P_1*O + O.transpose()*P_2)
        P_4 = zero_matrix(F,d,n-d)
        P = block_matrix(2,2,[[P_1,P_2],[P_4,P_3]])
        list_P.append(P)
    
    return list_P

# Generate a list of matrices where each matrix is the sum of the corresponding matrix in the input list and its transpose i.e M = P + P^t i.e symplectic forms UOV
def generate_list_M(list_P) -> list:
    """
    Generate a list of matrices where each matrix is the sum of the corresponding 
    matrix in the input list and its transpose.
    
    Parameters:
    ----------
    list_P : list of matrices

    Returns:
    -------
    list_M : list of matrices
        A list of matrices where each matrix M is the sum of the corresponding 
        matrix P in `list_P` and its transpose P^T.
    """
    
    # Create an empty list to store the matrices in list_M
    list_M = []
    
    # Iterate over each matrix in list_P
    for P in list_P:
        # Compute P + P^T (transpose of P)
        M = P + P.transpose()
        # Append the resulting matrix to list_M
        list_M.append(M)
    
    return list_M

# Run generate_public_matrices until all matrices are full-rank
def regenerate_until_full_rank(F: FiniteField, m: int, n: int, O: Matrix) -> tuple[list,list]:
    """
    Check if all matrices in list_M are full rank, otherwise regenerate list_M.
    
    Args:
    F : Finite Field (GF)
    m : Number of rows of the matrices
    n : Number of columns of the matrices
    O : Oil subspace matrix 
    
    Returns:
    list_P: List of public matrices (public key)
    list_M : List of full-rank matrices
    """
    while True:
        # Generate the public matrices
        list_P = generate_public_matrices(F, m, n, O)
        
        # Generate list_M from list_P
        list_M = generate_list_M(list_P)
        
        # Check if all matrices in list_M and list_P are full rank
        all_full_rank = True
        for M in list_M:
            if M.rank() < n:
                all_full_rank = False
                break
        for P in list_P:
            if P.rank() < n:
                all_full_rank = False
                break
        
        # If all matrices are full rank, return list_M
        if all_full_rank:
            return list_P, list_M

# Check if a subspace A (having basis X) is invariant under the linear transformation T
def is_invariant_subspace(F: FiniteField, X: Matrix, T: Matrix) -> bool:
    """
    Check if the subspace A spanned by X is invariant under the linear transformation T.
    
    Args:
        F: The finite field over which the matrices are defined.
        X: A matrix of size m x n representing the basis of the subspace A (with m basis vectors).
        T: A matrix of size n x n representing the linear transformation.

    Returns:
        True if the subspace A is invariant under T, False otherwise.
    """
    # Create the subspace A spanned by the rows of X
    A = span(X.rows())

    # Iterate over each basis vector in X
    for v in X.rows():
        # Apply the transformation T to the vector v
        T_v = v*T

        # Check if the transformed vector T_v lies in the span of the basis of A
        if not T_v in A:
            return False

    # If all transformed vectors lie in A, the subspace is invariant under T
    return True

# Create a function that checks if a subspace L vanishes in a list of matrices M[i] i.e. UOV public keys
def check_uov_vanishing(F: FiniteField, L , M_list: list) -> bool:
    """
    Check if for all row vectors x, y in subspace L and for all matrices M in M_list,
    we have x * M * y^T = 0.
    
    Args:
        F: A finite field.
        L: A subspace of vectors over F (as a MatrixSpace or a list of row vectors).
        M_list: A list of matrices over F.

    Returns:
        True if for all x, y in L, x * M * y^T = 0 for each M in M_list, False otherwise.
    """
    # Get the basis matrix of L 
    basis_matrix = L.basis_matrix()

    # Iterate over all pairs of row vectors (x, y) from the basis matrix
    for M in tqdm(M_list, ncols = 100, desc = "Check if UOV public keys vanish on Oil subpace ... "):
        # Check that for all row vectors x, y in the row space, x * M * y^T = 0
        for i in range(basis_matrix.nrows()):
            for j in range(basis_matrix.nrows()):
                x = basis_matrix.row(i)  # row vector x
                y = basis_matrix.row(j)  # row vector y
                
                # Calculate the result of x * M * y^T
                result = x * M * y.column()  # y.column() converts y to a column vector
                
                # Ensure the result is treated as a scalar and compare with 0
                if result != 0:
                    return False
    return True

# Construct the isotropic subspace basis from a symplectic basis noted that if m = n//2 it is a Lagrangian        
def compute_isotropic_subspace_basis(F: FiniteField, L: list, M: list, m: int, n: int, lag_or_not: bool, o: int = None):
    """
    This function computes o-dimensional isotropic subspace basis form a symplectic basis.

    Args:
        L (list): A list of matrices which are the symplectic basis, from which submatrices will be extracted.
        M (list): A list of matrices that are symplectic forms.
        n (int): Dimension of the Finite Field i.e. n = 2o
        m (int): Number of rows in the isotropic subspace basis i.e. the isotropic subspace is m-dimensional. If m = n//2 we are dealing with lagrangians.
        F: A finite field or identity matrix to check against.
        lag_or_not: We are computing o-dimensional isotropic subspace basis or not ? i.e we are computing Lagrangians or not ?
            If True, the function will automatically set o = n // 2.
            If False, it will ask the user to input the value for o.
        o (int, optional): The number of rows to select. Will be automatically set if lag_or_not is True.

    Returns:
        matrices_list: A list containing all o-dimensional isotropic subspace basis that are full-rank of m symplectic from of dimension n.
        all_check_pass (bool): Boolean flag indicating if all submatrices pass the Lagrangian check.
    """
    # Set o based on lag_or_not flag
    if lag_or_not:
        o = n // 2
        print(f"We are computing Lagrangian basis, setting o to {o} (n // 2)")
    elif o is None:  # If lag_or_not is False and no value for o is provided
        o = int(input("We are computing isotropic basis of dimension o, please enter the value of o: "))

    # Initialize a list to store all submatrices of size o x n that meet the criteria
    L_submatrix = []
    
    # Iterate over all possible matrices in L
    for l in tqdm(range(m), ncols=100, desc=f"Computing all {o}-dimensional isotropic subspaces of each of {m} symplectic basis ... "):
        L_list = []
        
        # Loop over possible k values
        for k in range(o + 1):
            first_half_rows = range(n//2)
            second_half_rows = range(n//2, n)
            
            # Choose k rows from the first half
            for first_set in combinations(first_half_rows, k):
                # Exclude the corresponding second half rows
                forbidden_rows = [u + n//2 for u in first_set]
                valid_second_half_rows = [row for row in second_half_rows if row not in forbidden_rows]
                
                # Choose o - k rows from the second half
                for second_set in combinations(valid_second_half_rows, o - k):
                    # Combine selected rows
                    selected_rows = list(first_set) + list(second_set)
                    
                    # Extract submatrix
                    submatrix = L[l][selected_rows, :]
                    
                    # Check for full rank using Sage's rank function
                    if submatrix.rank() == o:
                        L_list.append(submatrix)
        
        L_submatrix.append(L_list)
    
    # Check if the submatrices are isotropic and full rank
    all_check_pass = True
    for i in range(m):
        for j in range(len(L_submatrix[i])):
            if (L_submatrix[i][j] * M[i] * L_submatrix[i][j].transpose()).is_zero() == False or (L_submatrix[i][j].rank() != L_submatrix[i][j].nrows()):
                all_check_pass = False
                break  # Exit loop if any check fails
    
    # Only return and print results if all checks pass
    if all_check_pass:
        print(f"All submatrices are {o}-dimensional isotropic subspace basis and full rank.")
        return L_submatrix
    else:
        print("Some submatrices failed the check. No output returned.")
        return None

# Computes the matrix T such that T * L_0 = L_1 * P for two submatrices L_0 and L_1 are in matrices_list
def compute_transformation_T(matrices_list: list, P: Matrix):
    """
    Computes the matrix T such that T * L_0 = L_1 * P for two submatrices L_0 and L_1 of size m * n, 
    where P is of size n * n. The function will find the pair (L_0, L_1) that satisfies the condition
    and return T and the submatrices if a solution is found.

    Args:
        matrices_list: A list of submatrices from which L_0 and L_1 are extracted.
        P: The matrix P of size n * n used in the transformation.

    Returns:
        T (matrix or None): The transformation matrix T if a valid one is found, otherwise None.
        L_0_0 (matrix or None): The submatrix L_0_0 if a solution is found, otherwise None.
        L_0_1 (matrix or None): The submatrix L_0_1 if a solution is found, otherwise None.
        found_solution (bool): Flag indicating if a solution was found.
    """
    
    # Initialize a flag to track if a solution is found
    found_solution = False
    T = None
    L_0_0 = None
    L_0_1 = None
    m = matrices_list[0].nrows()
    n = matrices_list[0].ncols()

    # Loop over all possible pairs (i, j)
    for i in tqdm(range(len(matrices_list)), ncols=100, desc="Computing T such that T*L_0 = L_1*P ..."):
        for j in range(len(matrices_list)):
            try:
                # Get the i-th and j-th submatrices from matrices_list
                L0_i = matrices_list[i]
                L0_j = matrices_list[j]
                L0_j_P = L0_j*P

                # Compute the last m columns of L0_i and L0_j_P
                L0_i_tail = L0_i[:, -m:]
                L0_j_P_tail = L0_j_P[:, -m:]

                # Compute T = L0_j_P_tail * L0_i_tail.inverse() 
                T = L0_j_P_tail * L0_i_tail.inverse() 

                # Check if the transformation T satisfies the condition
                if T * L0_i != L0_j * P:
                    # If the condition does not hold, continue to the next (i, j)
                    continue
                
            except Exception as e:
                # If any error occurs (e.g., singular matrix), skip to the next pair (i, j)
                # print(e)
                continue  # Go to the next (i, j) pair

            # If the condition holds, output the result
            print(f"\nCheck T*L[0][{i}] == L[0][{j}]*P: {T * L0_i == L0_j * P}, rank T is full? {T.rank() == T.nrows()}")
            
            # Store the submatrices L_0_0 and L_0_1 as per the condition
            L_0_0 = L0_i
            L_0_1 = L0_j

            # Output the results and the matrices
            print("Condition holds!")
            print(f"L_0_0 (from index {i})", L_0_0, "\n")
            print(f"L_0_1 (from index {j})", L_0_1, "\n")

            # Set the flag to True since a solution was found
            found_solution = True

            # Break the inner loop (j loop) since we found a valid pair
            break
        if found_solution:
            # Break the outer loop (i loop) as well since we have found a solution
            break

    # If no solution was found after all iterations, print "No solution"
    if not found_solution:
        print("No solution found !")
    else:
        print(f"Check T*L_0_0 == L_0_1*P ? {T * L_0_0 == L_0_1 * P}")

    # Return the results
    return T, L_0_0, L_0_1

# Compute a non-trivial invariant subspace of a linear transformation P: F_q^n -> F_q^n
# def compute_invariant_subspace(F: FiniteField, P: Matrix) -> list:
#     """
#     Computes a non-trivial invariant subspace of a linear transformation P: F -> F

#     Args:
#         F: Finite field.
#         P: The matrix P of size n * n used in the transformation.

#     Returns:
#         list of basis vectors of the invariant subspace
#     """
#     # Step 1: Compute the characteristic polynomial of P
#     char_poly = P.charpoly()
#     print(char_poly)
#     # Step 2: Factor the characteristic polynomial to find potential eigenvalues
#     factors = char_poly.factor()
    
#     # Step 3: For each factor, solve for the null space of (P - λI)
#     for factor in factors:
#         poly_P = factor[0](P)
#         # Define a symbolic variable lambda
#         var('lambda')
        
#         # Compute the characteristic matrix (lambda * I - P)
#         I = identity_matrix(F, poly.nrows())  # Identity matrix
#         char_matrix = lambda * I - poly_P
        
#         # Compute the determinant of (lambda * I - P)
#         char_poly = char_matrix.det()

#         # Solve the characteristic equation: det(lambda * I - P) = 0
#         eigenvalues = solve(char_poly == 0, lambda)
#         for eigenvalue in eigenvalues:
#             # Construct the matrix (P - λI)
#             matrix_shifted = poly - eigenvalue * I
#             # Compute the null space (kernel) of this matrix
#             null_space = matrix_shifted.right_kernel()
            
#             # Step 4: Return a non-trivial subspace (i.e., non-zero eigenspace)
#             if null_space.dimension() > 0:
#                 return null_space.basis()  # Return the basis vectors for the invariant subspace
#         return []  # Return an empty list if no non-trivial subspace is found

# Define a function anti identity matrix that has rank n = 2k
def create_anti_identity_matrix(F: FiniteField,k: int) -> Matrix:
    """
    Generate a matrix like this
    
    [0 1 | 0 0]
    [1 0 | 0 0]
    -----------
    [0 0 | 0 1]
    [0 0 | 1 0]

    Args:
        F (FiniteField): Finite Field
        k (int): 1/2 dimension of the generated matrix 

    Returns:
        Matrix:     
                [0 1 | 0 0]
                [1 0 | 0 0]
                -----------
                [0 0 | 0 1]
                [0 0 | 1 0]
    """
    # Define the 2x2 block
    A = matrix(F,[[0, 1], [1, 0]])

    # Create a list of n blocks, each of size 2x2
    blocks = [A for i in range(k)]

    anti_identity = block_diagonal_matrix(blocks)

    # Return the result
    return anti_identity

# Function to check if a subspace is totally isotropic
def is_totally_isotropic(subspace_basis: list['Vector'], A: Matrix) -> bool:
    """
    Check if a basis of a vector subspace is totally isotropic

    Parameters:
    -----------
    subspace_basis: a list of vectors
    
    A: Matrix
    -----------

    Returns:
        bool: True/False
    """
    for v in subspace_basis:
        for w in subspace_basis:
            if (v * A * w) != 0:
                return False
    return True

# Brute-force search for totally isotropic subspace
def find_totally_isotropic_subspace(F: FiniteField,A: Matrix,  dim_subspace: int) -> list['Vector']:
    """
    Finds a totally isotropic subspace of a given dimension for a bilinear form matrix A over a finite field F.
    
    Parameters:
    -----------
    A : sage.matrix.matrix_space.Matrix
        A square Sage matrix representing the bilinear or quadratic form. It should be symmetric and have dimensions n x n.
    
    F : sage.rings.finite_rings.finite_field.FiniteField
        The finite field over which the matrix A is defined.
    
    dim_subspace : int
        The dimension of the desired isotropic subspace. This should be less than or equal to the dimension of the matrix A (i.e., dim_subspace <= n).
    
    Returns:
    --------
    list of sage.modules.free_module_element.Vector:
        A list of vectors that form the basis for the totally isotropic subspace of the given dimension.
    
    Raises:
    -------
    ValueError:
        If the desired dimension is larger than the dimension of the matrix A or if other invalid conditions are met.
    """
    
    n = A.nrows()  # Dimension of the vector space
    
    # Generate all possible subspaces of given dimension
    vector_space = VectorSpace(F, n)
    all_combinations = list(vector_space.subspaces(dim_subspace))
    # print(all_combinations)
    
    for subspace in tqdm(all_combinations, desc="Searching for totally isotropic subspace"):
        # Get a basis for the subspace
        subspace_basis = subspace.basis()
        
        # Check if this subspace is totally isotropic
        if is_totally_isotropic(subspace_basis, A):
            return subspace_basis  # Return the basis of the isotropic subspace
    
    return None  # No totally isotropic subspace found

# Function to generate matrix from basis vectors using a loop
def generate_matrix_from_basis(F: FiniteField,basis: list['Vector']) -> Matrix:
    """
    Generate a matrix from a list of vectors in a basis

    Args:
        F (_type_): Finite field
        basis (_type_): list of vectors

    Returns:
        representation_matrix: A matrix that represents the basis
    """
    # Create an empty matrix with the same number of rows as the vectors and as many columns as there are basis vectors
    rows = len(basis[0])  # The length of each vector gives the number of rows
    cols = len(basis)     # The number of basis vectors gives the number of columns
    representation_matrix = matrix(F, rows, cols)  # Initialize the matrix over F
    
    # Populate the matrix by adding each vector as a column
    for j, vec in enumerate(basis):
        for i in range(rows):
            representation_matrix[i, j] = vec[i]  # Place each element in the appropriate position
    
    return representation_matrix

# Function to check if isotropic_matrix.transpose() * L.transpose() * A * L * isotropic_matrix == 0
def check_condition(F: FiniteField, isotropic_matrix: Matrix, A: Matrix, L: Matrix) -> bool:
    """
    Check if isotropic_matrix.transpose() * L.transpose() * A * L * isotropic_matrix == 0

    Args:
        isotropic_matrix (Matrix): Target matrix to check
        A (Matrix): bilinear form or quadratic form representation matrix
        L (Matrix): A change of basis matrix if any
        F (FiniteField): Finite field

    Returns:
        True/False: True if == 0
    """
    # Ensure L is not a zero matrix
    if L.is_zero():
        return False  # L is zero, reject it
    # Proceed with the original condition check
    n, k = isotropic_matrix.nrows(), isotropic_matrix.ncols()
    result = isotropic_matrix.transpose() * L.transpose() * A * L * isotropic_matrix
    return (result.is_zero())

# Brute-force search for matrix L such that isotropic_matrix.transpose() * L.transpose() * A * L * isotropic_matrix == 0
def find_L_for_condition(F: FiniteField, isotropic_matrix: Matrix, A: Matrix) -> list[Matrix]:
    """
    Brute-force search for matrix L such that isotropic_matrix.transpose() * L.transpose() * A * L * isotropic_matrix == 0

    Args:
        isotropic_matrix (Matrix): A matrix
        A (Matrix): bilinear form or quadratic form representation matrix
        F (FiniteField): Finite field

    Returns:
        list[Matrix]: List of L
    """
    n = A.nrows()  # Dimension n
    q = F.order()

    # Generate all possible matrices of size n * n over F
    # This will be slow for large n and q, as we're brute-forcing it
    num_matrices = q ** (n * n)
    print(f"Brute-forcing {num_matrices} matrices of size {n}x{n} over GF({q})...")
    
    valid_L_matrices = []  # List to store all valid L matrices found

    # Progress bar using tqdm
    for entries in tqdm(cartesian_product_iterator([F]*n*n), total=int(num_matrices), desc="Searching for L"):
        # Reshape the list of entries into a matrix L of size n * n
        L = matrix(F, n, n, entries)
        
        # Check if L satisfies the condition
        if check_condition(isotropic_matrix, A, L, F):
            valid_L_matrices.append(L)  # Add valid L to the list

    # Return the list of valid L matrices
    return valid_L_matrices

# Brute-force search for matrices L such that L.transpose() * A * L == 0
def brute_force_search_isotropic_matrices(F: FiniteField,A: Matrix,  m: int) -> list:
    """
    Brute-force search for a matrix L of size n x m such that L^T * A * L == 0.

    Parameters:
    -----------
    A : Matrix
        A square matrix of size n x n representing the bilinear form or quadratic form.
    F : FiniteField
        The finite field over which the entries of matrix A, L are defined.
    m : int
        The number of columns of matrix L.

    Returns:
    --------
    Matrix or None:
        Returns a matrix L of size n x m that satisfies L^T * A * L == 0,
        or None if no such matrix is found.
    """
    n = A.nrows()
    valid_matrices = []
    
    # Total combinations
    total_combinations = len(F) ** (n * m)

    # Iterate over all possible values for matrix L in the finite field
    for values in tqdm(itertools.product(F, repeat=n * m), total=total_combinations, desc="Searching for valid matrices"):
        # Create a matrix L from the current combination of values
        L = matrix(F, n, m, values)
        
        # Check if L^T * A * L == 0
        if (L.transpose() * A * L).is_zero():
            valid_matrices.append(L)  # Add the valid matrix to the list

    return valid_matrices  # Return the list of valid matrices

# Check if a matrix A is an alternating matrix
def check_alternating_matrix(F: FiniteField, A: Matrix) -> bool:
    """
    Check if a matrix is an alternating matrix

    Args:
        A (Matrix): Any matrix
        F (FiniteField): Finite field

    Returns:
        bool: True/False
    """
    n_rows = A.nrows()
    n_cols = A.ncols()
    
    # The matrix must be square
    if n_rows != n_cols:
        return False
    
    # Check if all elements in the diagonal is zero or not:
    for i in range(n_rows):
        if A[i,i] != 0:
            return False
        
    # Check if the matrix is skew-symmetric, i.e. A = - A.transpose()
    for i in range(n_rows):
        for j in range(i+1,n_cols):
            if A[i,j] != A[j,i]:
                return False
            
    return True

# Diagonalize top-left 2x2 block
def diagonalize_2x2_alternating_matrix(F: FiniteField, A: Matrix) -> Matrix:
    """
    Diagonalize the top-left 2x2 block

    Args:
        A (Matrix): Matrix
        F (FiniteFiel): Finite field

    Returns:
        Matrix: Diagonalized-top-left-block matrix
    """
    # Check if the input matrix is a full-rank alternating matrix
    if check_alternating_matrix(F,A) == False:
        print("A is not a full-rank alternating matrix")
        return None    
    
    if A.is_zero():
        return matrix.identity(F,A.nrows())
    
    # Number of rows and columns
    n_rows = A.nrows()
    n_cols = A.ncols()
    
    # Check of A[0,1] != 0
    if A[0,1] != 0:   
        # Create an identity matrix of size (n_rows - 2) x (n_cols - 2)
        I = matrix.identity(F,n_rows -2)
        
        # Create top-left matrix
        L_1 = matrix(F,[[1,0],[0,A[0,1].inverse()]])
        
        # Create top-right matrix
        L_2 = zero_matrix(F,2,n_cols-2)
        
        # Create bottom-left matrix
        list_values = []
        for i in range(2,n_rows):
            list_values.append(A[0,i]*A[0,1].inverse())
            
        L_3 = matrix(F,[[0]*(n_rows - 2),list_values])
        L_3 = L_3.transpose()
        
        # Construct L
        L = block_matrix(F,[[L_1,L_2],[L_3,I]],subdivise = False)
        
        return L
    
    if A[0,1] == 0:
        # print("\n Switching ... \n")
        # Search for any A[0,i] != 0, it must exist at least one since A is a full-rank matrix
        for i in range(0,n_cols):
            if A[0,i] != 0:
                switch_index = i # This is the switching index
                break

        # Create a column switching matrix of size similar to A
        I = matrix.identity(F,n_rows)
        
        # Swap columns i and j
        I.swap_columns(i, 1)
        
        # Return the matrix A_new that has column 2 and i+1 switched
        A_switched = I*A*I.transpose()
        
        # Create bottom-righ matrix i.e. an identity matrix of size (n_rows - 2) x (n_cols - 2)
        K = matrix.identity(F,n_rows -2)
        
        # Create top-left matrix
        L_1 = matrix(F,[[1,0],[0,A_switched[0,1].inverse()]])
        
        # Create top-right matrix
        L_2 = zero_matrix(F,2,n_cols-2)
        
        # Create bottom-left matrix
        list_values = []
        for i in range(2,n_rows):
            list_values.append(A_switched[0,i]*A_switched[0,1].inverse())
            
        L_3 = matrix(F,[[0]*(n_rows - 2),list_values])
        L_3 = L_3.transpose()
        
        # Construct L
        L = block_matrix(F,[[L_1,L_2],[L_3,K]],subdivise = False)
        
        return L*I # Return switched L
    
# Diagonalize an alternating matrix
def diagonalize_full_alternating_matrix(F: FiniteField, A: Matrix) -> Matrix:
    """
    Return a matrix L such that L is invertible and L*A*L.transpose() = 
    
    [0 1 0 0]
    [1 0 0 0]
    [0 0 0 1]
    [0 0 1 0]

    Args:
        A (Matrix): an alternating matrix i.e. A is skew-symmetric and has zeros in its diagonal
        F (FiniteField): Finite Field

    Returns:
        Matrix: Matrix L
    """
    
    # Rows and Columns
    n_rows = A.nrows()
    n_cols = A.ncols()
    # A_original = A.copy()
    
    # Diagonalizing block by block
    list_i_A_L = []
    
    for i in range(n_rows):
        if i == 0:
            # Get the diagonalizing matrix of the first block
            L = diagonalize_2x2_alternating_matrix(F,A)
            
            # Get new A
            A = L*A*L.transpose()
            
            # Extract a submatrix of size (n_rows - 1) x (n_cols - 1)
            A = A.submatrix(1, 1, A.nrows()-1, A.ncols()-1)
            
            # Append to the list
            list_i_A_L.append((i,A,L))
            
        if i != 0:
            A = list_i_A_L[i-1][1]

            # Get the diagonalizing matrix of the first block as bottomright corner
            L = diagonalize_2x2_alternating_matrix(F,A)
            
            # Get new A
            A = L*A*L.transpose()
            
            # Get the identity matrix of size i * i as topleft corner
            I = matrix.identity(F,i)

            # Construct the block matrix
            L = block_diagonal_matrix(I,L)

            # Extract a submatrix of size (n_rows - 1) x (n_cols - 1)
            A = A.submatrix(1, 1, A.nrows()-1, A.ncols()-1)
                        
            # Append to the list
            list_i_A_L.append((i,A,L))
                        
    # Initialize the result_L as an identity matrix of the appropriate size
    result_L = matrix.identity(F, n_rows)   
    
    # Iterate through list_i_A_L and multiply all matrices L
    for _, _, L in list_i_A_L[::-1]:
        result_L *= L  # Multiply result by each matrix L

    # The result_L now contains the product of all matrices in list_L such that result_L*A*result_L.transpose() is almost anti identity
    
    # Final permutation matrix to get anti identity matrix
    list_P = []
    
    for i in range(2, n_rows, 2):
        P = identity_matrix(n_rows)
        P[i, i-2] = 1   # Add a 1 in the row two steps above
        list_P.append((i,P))
    
    result_P = matrix.identity(F, n_rows) 
    for _, P in list_P[::-1]:
        result_P *= P 

    return result_P*result_L

def zero_block_form(F: FiniteField, M: Matrix) -> tuple:
    """
    Given a skew-symmetric matrix M over the finite field F (of characteristic 2),
    this function finds a permutation of basis vectors (i.e., a permutation matrix P)
    such that P * M * P^T has a largest possible zero principal block in the top-left.
    It returns a pair (P, B), where B = P*M*P^T.

    Args:
        F (FiniteField):  A Sage finite field of characteristic 2.
        M (Matrix):       An n×n matrix over F which is skew-symmetric (i.e., M^T = -M = M in char 2).

    Returns:
        (Matrix, Matrix): A tuple (P, B) where
            • P is the n×n permutation matrix over F,
            • B = P * M * P^T has its largest zero top-left block.
    """
    # Ensure M is square and over the given field
    n = M.nrows()
    if M.ncols() != n:
        raise ValueError("M must be square.")
    if M.base_ring() != F:
        raise ValueError("The base ring of M must be the finite field F.")

    # Search for the largest zero principal submatrix
    # Loop k from n down to 1
    for k in reversed(range(1, n + 1)):
        for S in combinations(range(n), k):
            rows_and_cols = list(S)
            sub = M[rows_and_cols, rows_and_cols]
            if sub.is_zero():
                # Build the permutation list p = list(S) + remaining indices
                S_set = set(S)
                p = list(S) + [i for i in range(n) if i not in S_set]

                # Construct the permutation matrix P over F:
                P = Matrix(F, n)
                for i in range(n):
                    P[i, p[i]] = 1

                # Compute B = P * M * P^T
                B = P * M * P.transpose()
                return P, B

    # If no nontrivial zero principal block found, return the identity and M as-is
    P = identity_matrix(F, n)
    return P, M

def roots_over_finite_field(poly_F, want_extension_roots=False):
    """
    Input:
      - poly_F : a univariate polynomial in F[x], where F = GF(q).
      - want_extension_roots : if False, only return roots in F; if True, also
                                return each root in its splitting field for higher-degree factors.

    Output:
      - A list of (root, multiplicity) pairs.  If a factor is linear over F,
        root ∈ F; if it’s irreducible of deg>1 and want_extension_roots=True,
        you get roots in a small extension K/F.  If want_extension_roots=False,
        you’ll see the irreducible factor itself (with exponent) instead of its 
        algebraic roots.  
    """
    # 1) Factor over F = the base finite field
    facs = poly_F.factor()

    result = []
    for (fac, exp) in facs:
        if fac.degree() == 1:
            # it’s x - a, with a ∈ F
            a = fac.roots()[0][0]
            result += [ (a, exp) ]
        else:
            # degree > 1: irreducible over F
            if want_extension_roots:
                # build the splitting field K for 'fac'
                K = fac.splitting_field('theta')
                facK = fac.change_ring(K)
                for (r, m1) in facK.roots(multiplicities=True):
                    # total multiplicity = exp * m1
                    result.append((r, exp * m1))
            else:
                # just record the irreducible polynomial itself
                result.append((fac, exp))

    return result

"""
SNOVA functions
"""
# Lifting
def lift_matrix(M: Matrix, phi):
    """
    Apply the homomorphism phi entrywise to matrix M,
    and return the corresponding block matrix.

    INPUT:
        - M: a matrix over some ring or field (typically a finite field extension)
        - phi: a ring homomorphism mapping entries of M to matrices over another ring

    OUTPUT:
        - A block matrix where each entry M[i, j] is replaced with phi(M[i, j])
    """
    blocks = [[phi(M[i, j]) for j in range(M.ncols())] for i in range(M.nrows())]
    return block_matrix(blocks)



