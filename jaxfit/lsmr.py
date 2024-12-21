import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm

def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False, x0=None):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Matrix A in the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^H x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : array_like, shape (m,)
        Vector ``b`` in the linear system.
    damp : float
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::

         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2

        where damp is a scalar.  If damp is None or 0, the system
        is solved without regularization. Default is 0.
    atol, btol : float, optional
        Stopping tolerances. `lsmr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, `lsmr` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, `lsmr` terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        `lsmr` terminates if an estimate of ``cond(A)`` exceeds
        `conlim`.  For compatible systems ``Ax = b``, conlim could be
        as large as 1.0e+12 (say).  For least-squares problems,
        `conlim` should be less than 1.0e+8. If `conlim` is None, the
        default value is 1e+8.  Maximum precision can be obtained by
        setting ``atol = btol = conlim = 0``, but the number of
        iterations may then be excessive. Default is 1e8.
    maxiter : int, optional
        `lsmr` terminates if the number of iterations reaches
        `maxiter`.  The default is ``maxiter = min(m, n)``.  For
        ill-conditioned systems, a larger value of `maxiter` may be
        needed. Default is False.
    show : bool, optional
        Print iterations logs if ``show=True``. Default is False.
    x0 : array_like, shape (n,), optional
        Initial guess of ``x``, if None zeros are used. Default is None.

        .. versionadded:: 1.0.0

    Returns
    -------
    x : ndarray of float
        Least-square solution returned.
    istop : int
        istop gives the reason for stopping::

          istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a
                      solution.
                  = 1 means x is an approximate solution to A@x = B,
                      according to atol and btol.
                  = 2 means x approximately solves the least-squares problem
                      according to atol.
                  = 3 means COND(A) seems to be greater than CONLIM.
                  = 4 is the same as 1 with atol = btol = eps (machine
                      precision)
                  = 5 is the same as 2 with atol = eps.
                  = 6 is the same as 3 with CONLIM = 1/eps.
                  = 7 means ITN reached maxiter before the other stopping
                      conditions were satisfied.

    itn : int
        Number of iterations used.
    normr : float
        ``norm(b-Ax)``
    normar : float
        ``norm(A^H (b - Ax))``
    norma : float
        ``norm(A)``
    conda : float
        Condition number of A.
    normx : float
        ``norm(x)``

    Notes
    -----

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           :arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/

    `istop` indicates that the system is inconsistent and thus `x` is rather an
    approximate solution to the corresponding least-squares problem. `normr`
    contains the minimal distance that was found.
    """

    def _sym_ortho(a, b):
        """
        Stable implementation of Givens rotation using JAX.

        This function implements a numerically stable version of the Givens rotation. It is particularly
        useful in iterative methods for handling the orthogonal transformation needed in bidiagonalization
        processes, such as in LSMR algorithms.

        Parameters:
        - a (float): The coefficient of the rotation.
        - b (float): The coefficient of the rotation.

        Returns:
        - c (float): Cosine of the rotation angle.
        - s (float): Sine of the rotation angle.
        - r (float): Resultant length after rotation.

        Reference:
        S.-C. Choi, "Iterative Methods for Singular Linear Equations and Least-Squares Problems", Dissertation,
        http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
        """
        if b == 0:
            return jnp.sign(a), 0, jnp.abs(a)
        elif a == 0:
            return 0, jnp.sign(b), jnp.abs(b)
        elif jnp.abs(b) > jnp.abs(a):
            tau = a / b
            s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
            c = s * tau
            r = b / s
        else:
            tau = b / a
            c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
            s = c * tau
            r = a / c
        return c, s, r

    b = jnp.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    hdg1 = '   itn      x(1)       norm r    norm Ar'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20  # print frequency (for repeating the heading)
    pcount = 0  # print counter

    m, n = A.shape
    minDim = min(m, n)

    if maxiter is None:
        maxiter = minDim

    if x0 is None:
        dtype = jnp.result_type(A, b, float)
        x = jnp.zeros(n, dtype=dtype)
    else:
        dtype = jnp.result_type(A, b, x0, float)
        x = jnp.atleast_1d(x0.copy())

    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print(f'The matrix A has {m} rows and {n} columns')
        print('damp = %20.14e\n' % damp)
        print(f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n')
        print(f'btol = {btol:8.2e}             maxiter = {maxiter:8g}\n')


    normb = norm(b)
    beta = norm(b - A @ x) if x0 is not None else norm(b)
    u = b - A @ x if x0 is not None else b.copy()

    if beta > 0:
        u = (1 / beta) * u
        v = jnp.dot(A.T, u)
        alpha = norm(v)
    else:
        v = jnp.zeros(n, dtype)
        alpha = 0

    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for 1st iteration.
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = jnp.zeros(n, dtype)

    # Initialize variables for estimation of ||r||.
    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)
    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = jnp.sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    if normb == 0:
        x[()] = 0
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print(' ')
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {normr:10.3e} {normar:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(''.join([str1, str2, str3]))

    # Main iteration loop as specified
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the next beta, u, alpha, v
        u *= -alpha
        u += jax.jit(jnp.dot)(A, v)  # Adjusted to direct matrix-vector multiplication
        beta = norm(u)

        if beta > 0:
            u /= beta
            v *= -beta
            v += jax.jit(jnp.dot)(A.T, u)  # Adjusted for direct matrix-vector multiplication
            alpha = norm(v)
            if alpha > 0:
                v /= alpha

        # Construct rotation Qhat_{k,2k+1}
        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, h_hat, x
        hbar *= -thetabar * rho / (rhoold * rhobarold)
        hbar += h
        x += zeta / (rho * rhobar) * hbar
        h *= -thetanew / rho
        h += v

        # Estimate of ||r||
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        # betad   = betad_k here
        # rhodold = rhod_k  here

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = jnp.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        # Estimate ||A||
        normA2 = normA2 + beta * beta
        normA = jnp.sqrt(normA2)
        normA2 += alpha * alpha

        # Estimate cond(A)
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence
        normar = abs(zetabar)
        normx = norm(x)
        test1 = normr / normb
        test2 = normar / (normA * normr) if (normA * normr) != 0 else jnp.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # Output at specified intervals
        if show and ((n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or
                     (itn % 10 == 0) or (test3 <= 1.1 * ctol) or
                     (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or
                     (istop != 0)):
            print(
                f'{itn:6g} {x[0]:12.5e} {normr:10.3e} {normar:10.3e} {test1:8.1e} {test2:8.1e} {normA:8.1e} {condA:8.1e}')

        if istop > 0:
            break

    # Print the stopping condition.

    if show:
        print('\nLSMR finished')
        print(msg[istop])  # Ensure `msg` array is defined earlier with appropriate status messages
        print(f'istop ={istop:8g}    normr ={normr:8.1e}')
        print(f'    normA ={normA:8.1e}    normAr ={normar:8.1e}')
        print(f'itn   ={itn:8g}    condA ={condA:8.1e}')
        print(f'    normx ={normx:8.1e}')

        # Generate final output string for details
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {normr:10.3e} {normar:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        str4 = f' {normA:8.1e} {condA:8.1e}'
        print(str1, str2)
        print(str3, str4)

    return x, istop, itn, normr, normar, normA, condA, normx