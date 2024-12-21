import jax
import jax.numpy as jnp
from jax import lax, jit, Array
from jax.numpy.linalg import norm
from functools import partial
from typing import Tuple, Optional, Any, Dict
from jax.debug import print as jprint

# Define messages globally for status reporting
MSG = (
    'The exact solution is x = 0, or x = x0, if x0 was given  ',
    'Ax - b is small enough, given atol, btol                  ',
    'The least-squares solution is good enough, given atol     ',
    'The estimate of cond(Abar) has exceeded conlim            ',
    'Ax - b is small enough for this machine                   ',
    'The least-squares solution is good enough for this machine',
    'Cond(Abar) seems to be too large for this machine         ',
    'The iteration limit has been reached                      '
)

HDG1 = '   itn      x(1)       norm r    norm Ar'
HDG2 = ' compatible   LS    norm A   cond A\n'


def _print_iteration_info(variables: Dict, show: bool) -> None:
    """Print iteration information when show is True"""

    def true_fn(v):
        test1 = v['normr'] / v['normb']
        test2 = v['normar'] / (v['normA'] * v['normr'])

        # Format string with specific decimal places and field widths
        jprint("{:6d} {:12.5e} {:10.3e} {:10.3e} {:8.1e} {:8.1e} {:8.1e} {:8.1e}",
               v['itn'], v['x'][0], v['normr'], v['normar'],
               test1, test2, v['normA'], v['condA'])
        return None

    def false_fn(v):
        return None
    
    itn = variables['itn']
    log10_itn = jnp.floor(jnp.log10(itn + 1e-8))  # 加上一個小值以避免 log(0)
    base = 10 ** log10_itn
    should_print = (itn <= 10) | ((itn % base) == 0)

    lax.cond(
        show & should_print,
        true_fn,
        false_fn,
        variables
    )


def _print_final_info(variables: Dict, show: bool) -> None:
    """Print final information when show is True"""

    def true_fn(v):
        test1 = v['normr'] / v['normb']
        test2 = v['normar'] / (v['normA'] * v['normr'])

        # Format string with specific decimal places and field widths
        jprint("{:6d} {:12.5e} {:10.3e} {:10.3e} {:8.1e} {:8.1e} {:8.1e} {:8.1e}",
               v['itn'], v['x'][0], v['normr'], v['normar'],
               test1, test2, v['normA'], v['condA'])

        return None

    def false_fn(v):
        return None

    lax.cond(show, true_fn, false_fn, variables)

@jit
def _sym_ortho(a: float, b: float) -> Tuple[float, float, float]:
    """
    Stable implementation of Givens rotation using JAX.
    """

    def b_zero_case():
        return jnp.sign(a), 0.0, jnp.abs(a)

    def a_zero_case():
        return 0.0, jnp.sign(b), jnp.abs(b)

    def b_greater_case():
        tau = a / b
        s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
        return c, s, r

    def a_greater_equal_case():
        tau = b / a
        c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
        return c, s, r

    return lax.cond(
        b == 0,
        b_zero_case,
        lambda: lax.cond(
            a == 0,
            a_zero_case,
            lambda: lax.cond(
                jnp.abs(b) > jnp.abs(a),
                b_greater_case,
                a_greater_equal_case
            )
        )
    )


def _iteration_condition(variables: Dict, maxiter: int) -> Array:
    """Check if iteration should continue"""
    return jnp.logical_and(
        variables['itn'] < maxiter,
        variables['istop'] == 0
    )


def _check_convergence(variables: Dict, atol: float, btol: float, conlim: float, maxiter: int) -> Array:
    """Check convergence criteria"""
    test1 = variables['normr'] / variables['normb']
    test2 = variables['normar'] / (variables['normA'] * variables['normr'])
    test3 = 1 / variables['condA']
    rtol = btol + atol * variables['normA'] * variables['normx'] / variables['normb']

    return jnp.where(
        variables['itn'] >= maxiter, 7,
        jnp.where(1 + test3 <= 1, 6,
                  jnp.where(1 + test2 <= 1, 5,
                            jnp.where(1 + test1 <= 1, 4,
                                      jnp.where(test3 <= 1 / conlim, 3,
                                                jnp.where(test2 <= atol, 2,
                                                          jnp.where(test1 <= rtol, 1, 0)))))))


def _iteration_body(A: Any, damp: float, atol: float, btol: float, conlim: float, maxiter: int, show: bool):
    """Single iteration of LSMR algorithm"""

    @jit
    def body(variables: Dict) -> Dict:
        # Unpack variables
        x, h, hbar = variables['x'], variables['h'], variables['hbar']
        u, v = variables['u'], variables['v']
        alpha, beta = variables['alpha'], variables['beta']

        # Bidiagonalization step
        u = u * (-alpha) + jnp.dot(A, v)
        beta = norm(u)
        u = u / beta
        v = v * (-beta) + jnp.dot(A.T, u)
        alpha = norm(v)
        v = v / alpha

        # Construct and apply rotations
        chat, shat, alphahat = _sym_ortho(variables['alphabar'], damp)
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Update solution
        rhobarold = variables['rhobar']
        zetaold = variables['zeta']
        thetabar = variables['sbar'] * rho
        rhotemp = variables['cbar'] * rho
        cbar, sbar, rhobar = _sym_ortho(variables['cbar'] * rho, thetanew)
        zeta = cbar * variables['zetabar']
        zetabar = -sbar * variables['zetabar']

        # Update vectors
        hbar = hbar * (-thetabar * rho / (variables['rho'] * rhobarold)) + h
        x = x + (zeta / (rho * rhobar)) * hbar
        h = h * (-thetanew / rho) + v

        # Update norms and condition numbers
        normA2_new = variables['normA2'] + beta * beta + alpha * alpha
        normA = jnp.sqrt(normA2_new)

        maxrbar = jnp.maximum(variables['maxrbar'], rhotemp)
        minrbar = jnp.minimum(variables['minrbar'], rhotemp)
        condA = maxrbar / minrbar

        normr = jnp.sqrt(variables['d'] + (variables['betad'] - variables['taud']) ** 2 + variables['betadd'] ** 2)
        normar = jnp.abs(zetabar)
        normx = norm(x)

        # Update all variables
        new_vars = {
            'x': x,
            'h': h,
            'hbar': hbar,
            'itn': variables['itn'] + 1,
            'istop': variables['istop'],
            'normA2': normA2_new,
            'normA': normA,
            'maxrbar': maxrbar,
            'minrbar': minrbar,
            'normx': normx,
            'normb': variables['normb'],
            'normr': normr,
            'normar': normar,
            'condA': condA,
            'u': u,
            'v': v,
            'alpha': alpha,
            'beta': beta,
            'zetabar': zetabar,
            'alphabar': alphabar,
            'rho': rho,
            'rhobar': rhobar,
            'cbar': cbar,
            'sbar': sbar,
            'betadd': variables['betadd'],
            'betad': variables['betad'],
            'rhodold': variables['rhodold'],
            'tautildeold': variables['tautildeold'],
            'thetatilde': variables['thetatilde'],
            'zeta': zeta,
            'taud': variables['taud'],
            'd': variables['d'],
            'maxiter': variables['maxiter'],
        }

        # Check stopping criteria
        new_vars['istop'] = _check_convergence(new_vars, atol, btol, conlim, maxiter)
        _print_iteration_info(new_vars, show)
        return new_vars

    return body


@partial(jit, static_argnums=(4, 5, 6, 7))
def lsmr(A: Any,
         b: jnp.ndarray,
         damp: float = 0.0,
         atol: float = 1e-6,
         btol: float = 1e-6,
         conlim: float = 1e8,
         maxiter: Optional[int] = None,
         show: bool = False,
         x0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, int, int, float, float, float, float, float]:
    """
    JAX-optimized iterative solver for least-squares problems.
    """
    # Initialize shapes and parameters
    m, n = A.shape
    minDim = min([m, n])
    maxiter = minDim if maxiter is None else maxiter
    dtype = jnp.result_type(A, b, float)

    # Print initial information if show is True
    if show:
        print(' ')
        print('LSMR Least-squares solution of  Ax = b')
        print(f'The matrix A has {m} rows and {n} columns]\t')
        # jprint('damp = %20.14e' % damp)
        jprint('atol = {:8.2e} conlim = {:8.2e}\t',atol,conlim)
        jprint('btol = {:8.2e} maxiter = {:5d}',btol,maxiter)
        jprint(HDG1, HDG2)

    # Initialize vectors
    x = jnp.zeros(n, dtype=dtype) if x0 is None else jnp.asarray(x0, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype).reshape(-1)

    # Initialize norms and initial values
    normb = norm(b)
    beta = norm(b - A @ x) if x0 is not None else norm(b)
    u = (b - A @ x) if x0 is not None else b.copy()

    # Early termination checks
    def early_termination():
        return x, 0, 0, beta, 0.0, 0.0, 1.0, 0.0

    def continue_computation():
        u_normalized = u / beta
        v = jnp.dot(A.T, u_normalized)
        alpha = norm(v)
        v = v / alpha

        # Initialize iteration variables with matching structure
        init_vars = {
            'x': x,
            'h': v.copy(),
            'hbar': jnp.zeros(n, dtype=dtype),
            'itn': 0,
            'istop': 0,
            'normA2': alpha * alpha,
            'normA': jnp.sqrt(alpha * alpha),  # Added to match output structure
            'maxrbar': 0.0,
            'minrbar': jnp.finfo(dtype).max,
            'normx': 0.0,
            'normb': normb,
            'normr': beta,
            'normar': alpha * beta,
            'condA': 1.0,  # Added to match output structure
            'u': u_normalized,
            'v': v,
            'alpha': alpha,
            'beta': beta,
            'zetabar': alpha * beta,
            'alphabar': alpha,
            'rho': 1.0,
            'rhobar': 1.0,
            'cbar': 1.0,
            'sbar': 0.0,
            'betadd': beta,
            'betad': 0.0,
            'rhodold': 1.0,
            'tautildeold': 0.0,
            'thetatilde': 0.0,
            'zeta': 0.0,
            'taud': 0.0,
            'd': 0.0,
            'maxiter': maxiter,
        }

        final_vars = lax.while_loop(
            lambda vars: _iteration_condition(vars, maxiter),
            _iteration_body(A, damp, atol, btol, conlim, maxiter, show),
            init_vars
        )

        _print_final_info(final_vars, show)

        return (
            final_vars['x'],
            final_vars['istop'],
            final_vars['itn'],
            final_vars['normr'],
            final_vars['normar'],
            final_vars['normA'],
            final_vars['condA'],
            final_vars['normx']
        )

    return lax.cond(
        beta == 0,
        early_termination,
        continue_computation
    )