import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax.linalg import tridiagonal_solve
import time

from PGD_2D_Functions import *

# This code solves 2D Poisson's equation with PGD

# Problem parameters (parameter values in x and y directions are equal to each other):
Lx = 1.     # Domain length in x
nx = 10     # Number of elements in x-direction
nnx = nx+1  # Number of nodes in x-direction

x = jnp.linspace(0,Lx,nnx) # Nodal positions in x-direction
h = x[1]-x[0]

# PGD parameters:
# Fixed point tolerance:
epsilon = 1E-6
# PGD enrichment tolerance:
epsilon_tilde = 1E-3
# Maximum number of modes
max_terms = 20
# Maximum number of iterations in the fixed-point loop
max_fp_iter = 20

# Separated source functions:
Fx = -8*jnp.pi**2*jnp.sin(2*jnp.pi*x)
Fy = jnp.sin(2*jnp.pi*x)

# Calculate diagonal, upper & lower diagonal terms of mass matrix:
dl_m, d_m, du_m = fem_mass_matrix_terms(h, nnx)
dl_s, d_s, du_s = fem_stiffness_matrix_terms(h, nnx)

X_sol,Y_sol = PGD_Poisson_2D(nnx,max_terms,max_fp_iter,epsilon,epsilon_tilde,Fx,Fy,dl_m,d_m,du_m,dl_s,d_s,du_s)

# u = X_sol @ Y_sol.T
# X,Y = jnp.meshgrid(x ,x)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf = ax.plot_surface(X, Y, u, cmap='viridis')
#fig.colorbar(surf)
#plt.show()