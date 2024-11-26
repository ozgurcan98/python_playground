import jax
import jax.numpy as jnp
from jax import random

def PGD_Poisson_2D(nn,max_terms,max_fp_iter,epsilon,epsilon_tilde,Fx,Fy,dl_m,d_m,du_m,
                   dl_s,d_s,du_s):
    """
    PGD solution of Poisson's equation in 2D
    Args:
        nn: Number of nodes in each direction (assumed to be equal)
        max_terms: Maximum number of modes
        max_fp_iter: Maximum number of fixed-point iterations
        epsilon: Fixed-point tolerance
        epsilon_tilde: Enrichment tolerance
        Fx & Fy: Source function in decomposed form
        dl_, d_, du_m: Lower, diagonal and upper diagonal terms of mass matrix
        dl_, d_, du_s: Lower, diagonal and upper diagonal terms of stiffness matrix
    Returns:
        X_sol: x-modes
        Y_sol: y-modes
    """
    X_sol = jnp.zeros((nn,max_terms))
    Y_sol = jnp.zeros((nn,max_terms))

    key = random.PRNGKey(758493)
    for term in range(max_terms):
        # Initialization of the FP loop
        Sx = random.uniform(key, shape=(nn,))
        Sy = random.uniform(key, shape=(nn,))
        Sx = jnp.ones(nn)
        Sy = jnp.ones(nn)
        # Modify the last and end nodal values:
        Sx = Sx.at[0].set(0)
        Sy = Sy.at[0].set(0)
        Sx = Sx.at[-1].set(0)
        Sy = Sy.at[-1].set(0)

        # Fixed point iterations:
        for iter in range(max_fp_iter):
            # Store the old values of Sx & Sy for later comparison
            Sx_old = Sx
            Sy_old = Sy

            # Solve for Sx:
            # LHS coefficients:
            alpha_x = bilinear_form_with_multiplication(Sy,Sy, dl_m, d_m, du_m)
            beta_x  = bilinear_form_with_multiplication(Sy,Sy, dl_s, d_s, du_s)

            # RHS terms:
            # Calculate the contribution of separable source functions:
            #RHS = 

        X_sol = X_sol.at[:,term].set(Sx)
        Y_sol = Y_sol.at[:,term].set(Sy)

    return X_sol, Y_sol

def bilinear_form_with_multiplication(Y, X, dl, d, du):
    """
    Compute Y.T @ M @ X by first computing w = Y.T @ M and then w @ X.
    Args:
        Y: Input vector of shape (N,).
        X: Input vector of shape (N,).
        dl: Sub-diagonal entries of M (length N-1).
        d: Diagonal entries of M (length N).
        du: Super-diagonal entries of M (length N-1).
    Returns:
        result: The scalar value of Y.T @ M @ X.
    """
    N = len(Y)
    w = jnp.zeros_like(Y)

    # Compute w using the tridiagonal structure
    w = w.at[0].set(d[0] * Y[0] + dl[0] * Y[1])  # First row
    for i in range(1, N - 1):
        w = w.at[i].set(dl[i - 1] * Y[i - 1] + d[i] * Y[i] + du[i] * Y[i + 1])  # Middle rows
    w = w.at[N - 1].set(du[N - 2] * Y[N - 2] + d[N - 1] * Y[N - 1])  # Last row

    # Compute final result
    result = jnp.dot(w, X)

    return result

def fem_mass_matrix_terms(h, N):
    """
    Compute the diagonal, lower diagonal, and upper diagonal terms of 
    a 1D mass matrix for linear finite elements with uniform elements.

    Args:
        h: Element length (uniform mesh).
        N: Number of nodes.

    Returns:
        dl: Lower diagonal entries (length N-1).
        d: Diagonal entries (length N).
        du: Upper diagonal entries (length N-1).
    """
    # Off-diagonal (lower and upper)
    a = h / 6  # Sub-diagonal and super-diagonal terms

    # Diagonal
    d = jnp.full(N, h / 3)  # Initialize with boundary contributions
    d = d.at[1:N-1].set(2 * h / 3)  # Internal nodes receive overlap contributions

    # Create lower and upper diagonals
    dl = jnp.full(N - 1, a)
    du = jnp.full(N - 1, a)
    
    return dl, d, du

def fem_stiffness_matrix_terms(h, N):
    """
    Compute the diagonal, lower diagonal, and upper diagonal terms of 
    a 1D stifness matrix for linear finite elements with uniform elements.

    Args:
        h: Element length (uniform mesh).
        N: Number of nodes.

    Returns:
        dl: Lower diagonal entries (length N-1).
        d: Diagonal entries (length N).
        du: Upper diagonal entries (length N-1).
    """
    # Off-diagonal (lower and upper)
    a = -1 / h  # Sub-diagonal and super-diagonal terms

    # Diagonal
    d = jnp.full(N, 1 / h)  # Initialize with boundary contributions
    d = d.at[1:N-1].set(2 / h)  # Internal nodes receive overlap contributions

    # Create lower and upper diagonals
    dl = jnp.full(N - 1, a)
    du = jnp.full(N - 1, a)
    
    return dl, d, du
