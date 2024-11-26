import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax.linalg import tridiagonal_solve
import time

#jax.config.update("jax_enable_x64", True)

def fem_mass_mv_vectorized(h, x):
    """
    Multiply the 1D FEM mass matrix with a vector (vectorized implementation).
    Args:
        h: Element length (uniform mesh).
        x: Input vector (length n).
    Returns:
        y: Resultant vector (length n).
    """
    n = len(x)
    b = h / 3  # Diagonal entries
    a = h / 6  # Sub- and super-diagonal entries

    # Compute the product
    y = jnp.zeros_like(x)

    # Middle nodes: use slicing
    y = y.at[1:-1].set(a * x[:-2] + 2 * b * x[1:-1] + a * x[2:])

    # First and last nodes
    y = y.at[0].set(b * x[0] + a * x[1])  # First node
    y = y.at[-1].set(a * x[-2] + b * x[-1])  # Last node

    return y


# Define the source term function f(x)
f_func = lambda x: jnp.pi**2 * jnp.sin(jnp.pi * x)

# Solve the Poisson equation
L = 1.0       # Length of the domain
n = 100000        # Number of elements
nnx = n+1     # Number of nodes

# Element length:
h = L / n
# Nodal positions:
x = jnp.linspace(0, L, nnx)

# LHS terms (Stiffness matrix with uniform elements)
# Tridiagonal coefficients
dl = -jnp.ones(nnx) / h       # Sub-diagonal
dl = dl.at[0].set(0)
dl = dl.at[-1].set(0)

du = -jnp.ones(nnx) / h       # Super-diagonal
du = du.at[-1].set(0)
du = du.at[0].set(0)

d  = 2 * jnp.ones(nnx) / h    # Main diagonal
d = d.at[0].set(1)
d = d.at[-1].set(1)

# Construct RHS vector
start_time = time.time()
rhs = f_func(x)
rhs = fem_mass_mv_vectorized(h, rhs)
rhs = rhs.at[0].set(0); rhs = rhs.at[-1].set(0)
rhs = jnp.expand_dims(rhs,1)
end_time = time.time()
execution_time_rhs = end_time - start_time

# Solve the tridiagonal system
start_time = time.time()
u_internal = tridiagonal_solve(dl, d, du, rhs)  # Remove batch dimension
end_time = time.time()
execution_time = end_time - start_time

# Plot results
plt.plot(x,u_internal)
print("Execution time: ",execution_time)
print("Execution time for rhs: ",execution_time_rhs)
# Solution error:
#Error = jnp.linalg.norm(jnp.sin(jnp.pi * x) - u_internal.T) / jnp.linalg.norm(jnp.sin(jnp.pi * x))
#print("Error: ", Error)

plt.show()
