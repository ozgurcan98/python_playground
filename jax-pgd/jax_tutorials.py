import jax
import jax.numpy as jnp
import time
import numpy as np

# Define the iteration function
def g(x):
    return jnp.cos(x)

def g_numpy(x):
    return np.cos(x)

# Define the fixed-point iteration in NumPy
def fixed_point_iter_numpy(g, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_next = g(x)
        if np.abs(x_next - x) < tol:
            break
        x = x_next
    return x

# Define the fixed-point iteration function
def fixed_point_iter_jax(x0, tol=1e-6, max_iter=100):
    def body_fun(carry):
        x_prev, iter_count = carry
        x_next = g(x_prev)
        return (x_next, iter_count + 1)

    def cond_fun(carry):
        x_prev, iter_count = carry
        return (jnp.abs(g(x_prev) - x_prev) > tol) & (iter_count < max_iter)

    x_final, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, 0))
    return x_final

# Initial guess
x0 = jnp.array(0.5)
# Run the JIT-compiled function
start_time = time.time()
x_fixed = fixed_point_iter_jax(x0)
end_time = time.time()
execution_time = end_time - start_time

print("Non-JIT version solution: ",x_fixed)
print("Non-JIT timing: ",execution_time)

# Wrap the function with jax.jit
fixed_point_iter_jax_jit = jax.jit(fixed_point_iter_jax)

# Measure execution time for NumPy
x0_np = 0.5
start_time_numpy = time.time()
fixed_point_numpy = fixed_point_iter_numpy(g_numpy, x0_np)
numpy_time = time.time() - start_time_numpy

print("numpy version solution: ",fixed_point_numpy)
print("numpy timing: ",numpy_time)

# Initial guess
x0 = jnp.array(0.5)
# Run the JIT-compiled function
start_time = time.time()
x_fixed = fixed_point_iter_jax_jit(x0)
end_time = time.time()
execution_time_jit = end_time - start_time

print("JIT version solution: ",x_fixed)
print("JIT timing: ",execution_time_jit)