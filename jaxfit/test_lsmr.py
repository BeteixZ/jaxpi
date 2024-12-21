import jax.numpy as jnp
from lsmr_jit import lsmr
import jax
import timeit
import jax.random as random
jax.config.update('jax_enable_x64', True)
import os
# memory not preallocated
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Example usage
key = random.PRNGKey(0)
A = random.normal(key, (500, 500))
b = random.normal(key, (500,))

# import line_profiler
# from lsmr_jit import _iteration_body, _sym_ortho
# 
# def main():
#    x = lsmr(A, b, atol=1e-9, btol=1e-9, conlim=1e8, maxiter=1000 , show=False)
# 
# 
# lp = line_profiler.LineProfiler()
# m = lp(main)
# lp.add_function(_iteration_body)
# lp.add_function(_sym_ortho)
# m()
# lp.print_stats()


from jax import profiler

profiler.start_trace(log_dir='./tmp')

# 執行您的函數
lsmr(A, b, atol=1e-9, btol=1e-9, conlim=1e8, maxiter=1000, show=False)

profiler.stop_trace()
print("done")