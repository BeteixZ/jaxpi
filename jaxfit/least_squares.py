"""Generic interface for least-squares minimization."""
from warnings import warn
import numpy as np
import time
from typing import Callable, Optional, Tuple, Union, Sequence, List, Any

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd
from jax.scipy.linalg import solve_triangular as jax_solve_triangular

from jaxfit.trf import TrustRegionReflective
from jaxfit.loss_functions import LossFunctionsJIT
from jaxfit.common_scipy import EPS, in_bounds, make_strictly_feasible



TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied."
}

def prepare_bounds(bounds, n) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare bounds for optimization.

    This function prepares the bounds for the optimization by ensuring that 
    they are both 1-D arrays of length `n`. If either bound is a scalar, it is 
    resized to an array of length `n`.

    Parameters
    ----------
    bounds : Tuple[np.ndarray, np.ndarray]
        The lower and upper bounds for the optimization.
    n : int
        The length of the bounds arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The prepared lower and upper bounds arrays.
    """
    lb, ub = [np.asarray(b, dtype=float) for b in bounds]
    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


def check_tolerance(ftol: float, 
                    xtol: float, 
                    gtol: float, 
                    method: str
                    ) -> Tuple[float, float, float]:
    """Check and prepare tolerance values for optimization.

    This function checks the tolerance values for the optimization and 
    prepares them for use. If any of the tolerances is `None`, it is set to 
    0. If any of the tolerances is lower than the machine epsilon, a warning 
    is issued and the tolerance is set to the machine epsilon. If all 
    tolerances are lower than the machine epsilon, a `ValueError` is raised.

    Parameters
    ----------
    ftol : float
        The tolerance for the optimization function value.
    xtol : float
        The tolerance for the optimization variable values.
    gtol : float
        The tolerance for the optimization gradient values.
    method : str
        The name of the optimization method.

    Returns
    -------
    Tuple[float, float, float]
        The prepared tolerance values.
    """
    def check(tol: float, name: str) -> float:
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn("Setting `{}` below the machine epsilon ({:.2e}) effectively "
                 "disables the corresponding termination condition."
                 .format(name, EPS))
        return tol

    ftol = check(ftol, "ftol")
    xtol = check(xtol, "xtol")
    gtol = check(gtol, "gtol")

    if ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError("At least one of the tolerances must be higher than "
                         "machine epsilon ({:.2e}).".format(EPS))

    return ftol, xtol, gtol


def check_x_scale(x_scale: Union[str, Sequence[float]],
                  x0: Sequence[float]
                  ) -> Union[str, Sequence[float]]:
    """Check and prepare the `x_scale` parameter for optimization.

    This function checks and prepares the `x_scale` parameter for the 
    optimization. `x_scale` can either be 'jac' or an array_like with positive
    numbers. If it's 'jac' the jacobian is used as the scaling. 

    Parameters
    ----------
    x_scale : Union[str, Sequence[float]]
        The scaling for the optimization variables.
    x0 : Sequence[float]
        The initial guess for the optimization variables.

    Returns
    -------
    Union[str, Sequence[float]]
        The prepared `x_scale` parameter.
    """

    if isinstance(x_scale, str) and x_scale == 'jac':
        return x_scale

    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with "
                         "positive numbers.")

    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale

"""Wraps the given function such that a masked jacfwd is performed on it
thereby giving the autodiff jacobian."""
class AutoDiffJacobian():
    """Wraps the residual fit function such that a masked jacfwd is performed 
    on it. thereby giving the autodiff Jacobian. This needs to be a class since 
    we need to maintain in memory three different versions of the Jacobian.
    """

    def create_ad_jacobian(self, 
                           func: Callable, 
                           num_args: int, 
                           masked: bool = True
                           ) -> Callable:
        """Creates a function that returns the autodiff jacobian of the 
        residual fit function. The Jacobian of the residual fit function is
        equivalent to the Jacobian of the fit function.

        Parameters
        ----------
        func : Callable
            The function to take the jacobian of.
        num_args : int
            The number of arguments the function takes.
        masked : bool, optional
            Whether to use a masked jacobian, by default True

        Returns
        -------
        Callable
            The function that returns the autodiff jacobian of the given
            function.
        """

        @jit
        def jac_func(args: List[float]) -> jnp.ndarray:
            """Returns the jacobian. Places all the residual fit function
            arguments into a single list for the wrapped residual fit function.
            Then calls the jacfwd function on the wrapped function with the
            the arglist of the arguments to differentiate with respect to which
            is only the arguments of the original fit function.
            """
            args = jnp.array(args)
            jac_fwd = jax.jacrev(func)(args)
            return jnp.atleast_2d(jac_fwd)

        return jac_func

    def create_ad_heassian(self, func: Callable, num_args: int, masked: bool = True) -> Callable:
        @jit
        def hessian_func(args: List[float]) -> jnp.ndarray:
            """Returns the hessian of the function. Places all the residual fit
            function arguments into a single list for the wrapped residual fit 
            function. Then calls the jacfwd function on the wrapped function with 
            the arglist of the arguments to differentiate with respect to which is 
            only the arguments of the original fit function.
            """
            args = jnp.array(args)
            hessian_fwd = jax.jacfwd(jax.jacfwd(func))(args)
            return hessian_fwd

        return hessian_func

class LeastSquares():
    
    def __init__(self):
        super().__init__() # not sure if this is needed
        self.trf = TrustRegionReflective()
        self.ls = LossFunctionsJIT()
        #initialize jacobian to None and f to a dummy function
        self.f = lambda x: None 
        self.jac = None

    
    def least_squares(self,
                      fun: Callable, 
                      x0: np.ndarray, 
                      jac: Optional[Callable] = None, 
                      bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
                      method: str = 'trf',
                      ftol: float = 1e-8,
                      xtol: float = 1e-8,
                      gtol: float = 1e-6,
                      x_scale: Union[str, np.ndarray, float] = 1.0,
                      loss: str = 'linear',
                      f_scale: float = 1.0,
                      diff_step=None, 
                      tr_solver=None, 
                      tr_options={},
                      jac_sparsity=None, 
                      max_nfev: Optional[float] = None, 
                      verbose: int = 0,
                      timeit: bool = False,
                      args=(), 
                      kwargs={}):
        

            
        if loss not in self.ls.IMPLEMENTED_LOSSES and not callable(loss):
            raise ValueError("`loss` must be one of {0} or a callable."
                             .format(self.ls.IMPLEMENTED_LOSSES.keys()))
        
        if method not in ['trf']:
            raise ValueError("`method` must be 'trf")
            
        if jac not in [None] and not callable(jac):
            raise ValueError("`jac` must be None or callable.")
    
        if verbose not in [0, 1, 2]:
            raise ValueError("`verbose` must be in [0, 1, 2].")
    
        if len(bounds) != 2:
            raise ValueError("`bounds` must contain 2 elements.")
    
        if max_nfev is not None and max_nfev <= 0:
            raise ValueError("`max_nfev` must be None or positive integer.")
    
        if np.iscomplexobj(x0):
            raise ValueError("`x0` must be real.")
    
        x0 = np.atleast_1d(x0).astype(float)
    
        if x0.ndim > 1:
            raise ValueError("`x0` must have at most 1 dimension.")
            
        self.n = len(x0)

    
        lb, ub = prepare_bounds(bounds, x0.shape[0])
    
        if lb.shape != x0.shape or ub.shape != x0.shape:
            raise ValueError("Inconsistent shapes between bounds and `x0`.")
    
        if np.any(lb >= ub):
            raise ValueError("Each lower bound must be strictly less than each upper bound.")
    
        if not in_bounds(x0, lb, ub):
            raise ValueError("`x0` is infeasible.")
            
    
        x_scale = check_x_scale(x_scale, x0)
        ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)
        x0 = make_strictly_feasible(x0, lb, ub)
        


        # this if/else is to maintain compatibility with the SciPy suite of tests
        # which assume the residual function contains the fit data which is not
        # the case for JAXFit due to how we've made the residual function
        # to be compatible with JAX JIT compilation
        def wrap_func(fargs):
            return jnp.atleast_1d(fun(fargs, **kwargs))

        def wrap_jac(fargs, xdata):
            return jnp.atleast_2d(jac(fargs, **kwargs))

        rfunc = wrap_func
        if jac is None:
            adj = AutoDiffJacobian()
            jac_func = adj.create_ad_jacobian(wrap_func, self.n, masked=False)
            # hessian_func = adj.create_ad_heassian(wrap_func, self.n, masked=False)
        else:
            jac_func = wrap_jac

        f0 = rfunc(x0)
        J0 = jac_func(x0)
        # H0 = hessian_func(x0)

        #print("Shape of residual:", f0.shape)
        #print("Shape of Jacobian:", J0.shape)
        
        # Calculate condition number of Jacobian
        # print("Condition number of Jacobian:", jnp.linalg.cond(J0))
        
        # export jacobian into a h5 file
        #with h5py.File('jacobian.h5', 'w') as f:
        #    f.create_dataset('jacobian', data=J0)


        if f0.ndim != 1:
            raise ValueError("`fun` must return at most 1-d array_like. f0.shape: {0}".format(f0.shape))

        if not np.all(np.isfinite(f0)):
            raise ValueError("Residuals are not finite in the initial point.")

        n = x0.size
        m = f0.size
        
        if J0 is not None:
            if J0.shape != (m, n):
                raise ValueError("The return value of `jac` has wrong shape: expected {0}, actual {1}.".format((m, n), J0.shape))

        
        loss_function = self.ls.get_loss_function(loss)

        if callable(loss):
            rho = loss_function(f0, f_scale, data_mask=None)
            if rho.shape != (3, m):
                raise ValueError("The return value of `loss` callable has wrong shape.")
            initial_cost_jnp = self.trf.calculate_cost(rho, None)
        elif loss_function is not None:
            initial_cost_jnp = loss_function(f0, f_scale, data_mask=None, cost_only=True)
        else:
            initial_cost_jnp = self.trf.default_loss_func(f0)
        initial_cost = np.array(initial_cost_jnp)
        
        result = self.trf.trf(rfunc, None, None, jac_func, None,
                              None, x0, f0, J0, lb, ub, ftol, xtol,
                     gtol, max_nfev, f_scale, x_scale, loss_function,
                     tr_options.copy(), verbose, timeit)
    
    
        result.message = TERMINATION_MESSAGES[result.status]
        result.success = result.status > 0
    
        if verbose >= 1:
            print(result.message)
            print("Function evaluations {0}, initial cost {1:.4e}, final cost {2:.4e}, first-order optimality {3:.2e}."
                  .format(result.nfev, initial_cost, result.cost,
                          result.optimality))
            
        return result
   
