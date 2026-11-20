"""
improved_root.py -- a reliability-checked wrapper around scipy.optimize.newton.

Motivation (SciPy issue #6379): ``newton`` stops when the *step size* is
small, so it can report ``converged=True`` at a point that is not a root.
For f(x) = x**4 - x**2 + 1 started at x0 = 0.001 the secant iteration stalls
near x = 0.001 where f(x) ~ 1, and SciPy still reports convergence.

``reliable_root`` runs SciPy's solver unchanged, then *verifies* the answer:

1. Residual test   |f(x)| <= ftol
2. Newton-step test |f(x) / f'(x)| <= xtol * max(1, |x|)
   (an estimate of the distance to the nearest simple root; f' is
   estimated by a central difference when no derivative is given).
   This catches "flat tail" false convergence where |f| is tiny but x is
   nowhere near a root, e.g. f(x) = x*exp(-x) at x = 40.

If verification fails, it falls back to Brent's method (``brentq``), which is
guaranteed to converge when given a bracket [a, b] with f(a)*f(b) < 0.
The bracket can be supplied by the user or searched for automatically.
If no bracket can be found the function says so -- it never invents a root.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, newton

# Status codes
RELIABLE = "reliable"            # SciPy's answer passed verification
RECOVERED = "recovered"          # SciPy's answer failed; Brent found a verified root
NO_ROOT_FOUND = "no_root_found"  # SciPy's answer failed and no bracket was found
UNRELIABLE = "unreliable"        # a bracket existed but the result still fails
                                 # verification (e.g. a pole / discontinuity)


class _CountingFunction:
    """Wraps f and counts how many times it is evaluated."""

    def __init__(self, f):
        self.f = f
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.f(x)


@dataclass
class RootReport:
    """Everything a user needs to decide whether to trust the answer."""

    root: float | None          # verified root, or None if none was found
    status: str                 # one of the status codes above
    reliable: bool              # True only if `root` passed verification
    method: str                 # method that produced the final answer
    residual: float             # |f(root)| (or of SciPy's candidate if no root)
    step_estimate: float        # |f/f'| at that point
    scipy_root: float           # what scipy.optimize.newton returned
    scipy_converged: bool       # what SciPy claimed
    scipy_residual: float       # |f(scipy_root)|
    false_convergence: bool     # SciPy said converged but verification failed
    bracket: tuple | None       # bracket used for the fallback, if any
    nfev: int                   # total function evaluations (all stages)
    scipy_nfev: int             # evaluations used by SciPy's newton alone
    message: str
    notes: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Status          : {self.status.upper()}",
            f"Root            : {self.root}",
            f"Method          : {self.method}",
            f"|f(root)|       : {self.residual:.3e}",
            f"|f/f'| estimate : {self.step_estimate:.3e}",
            f"SciPy returned  : {self.scipy_root!r} (converged={self.scipy_converged}, "
            f"|f|={self.scipy_residual:.3e})",
            f"Function evals  : {self.nfev} total ({self.scipy_nfev} by SciPy)",
            f"Message         : {self.message}",
        ]
        if self.false_convergence:
            lines.insert(1, "WARNING         : SciPy reported convergence, "
                            "but its answer is NOT a root.")
        return "\n".join(lines)


def _safe_eval(f, x):
    try:
        with np.errstate(all="ignore"):
            return float(f(x))
    except (OverflowError, ZeroDivisionError, ValueError):
        return math.nan


def _verify(f, fprime, x, ftol, xtol):
    """Return (ok, |f(x)|, |f/f'|) for a candidate root x."""
    if x is None or not np.isfinite(x):
        return False, math.inf, math.inf
    fx = _safe_eval(f, x)
    if not np.isfinite(fx):
        return False, math.inf, math.inf
    h = 1e-6 * max(1.0, abs(x))
    if fx == 0.0:
        # An exact zero is only trustworthy if f is not *also* zero next to x.
        # Otherwise f has underflowed (e.g. x*exp(-x) at x = 10000) and the
        # "root" is an artifact of floating point, not of the function.
        if _safe_eval(f, x - h) == 0.0 and _safe_eval(f, x + h) == 0.0:
            return False, 0.0, math.inf
        return True, 0.0, 0.0
    if fprime is not None:
        dfx = _safe_eval(fprime, x)
    else:
        dfx = (_safe_eval(f, x + h) - _safe_eval(f, x - h)) / (2 * h)
    step = abs(fx / dfx) if (np.isfinite(dfx) and dfx != 0.0) else math.inf
    ok = abs(fx) <= ftol and step <= xtol * max(1.0, abs(x))
    return ok, abs(fx), step


def find_bracket(f, x0, initial_step=None, growth=1.6, max_expansions=50):
    """Search outward from x0 for [a, b] with f(a) * f(b) < 0.

    Samples x0 +/- step * growth**k on both sides and returns the first pair of
    neighbouring samples whose signs differ, or None if none is found. This
    cannot find roots of even multiplicity (no sign change), and it can be
    fooled by poles (sign change without a root) -- the caller must verify.
    """
    step = initial_step if initial_step is not None else 1e-2 * max(1.0, abs(x0))
    f0 = _safe_eval(f, x0)
    if f0 == 0.0:
        return (x0, x0)
    left_x, left_f = x0, f0
    right_x, right_f = x0, f0
    for k in range(max_expansions):
        d = step * growth**k
        for side in (+1, -1):
            x_new = x0 + side * d
            f_new = _safe_eval(f, x_new)
            if not np.isfinite(f_new):
                continue
            prev_x, prev_f = (right_x, right_f) if side > 0 else (left_x, left_f)
            if np.isfinite(prev_f) and prev_f * f_new <= 0:
                return (min(prev_x, x_new), max(prev_x, x_new))
            if side > 0:
                right_x, right_f = x_new, f_new
            else:
                left_x, left_f = x_new, f_new
    return None


def reliable_root(f, x0, fprime=None, bracket=None, *, ftol=1e-6, xtol=1e-6,
                  auto_bracket=True, maxiter=50, **newton_kwargs) -> RootReport:
    """Find a root of f starting from x0, and say whether it can be trusted.

    Parameters
    ----------
    f : callable            scalar function f(x)
    x0 : float              initial guess passed to scipy.optimize.newton
    fprime : callable       optional derivative (Newton instead of secant)
    bracket : (a, b)        optional interval with f(a) * f(b) < 0 for the
                            Brent fallback
    ftol : float            residual tolerance |f(x)| <= ftol. 1e-6 is a
                            demonstration default, not a universal guarantee;
                            choose it relative to the scale of f.
    xtol : float            relative tolerance on the Newton-step error estimate
    auto_bracket : bool     search for a bracket if none is given
    maxiter : int           passed to newton
    """
    fc = _CountingFunction(f)

    # ---- Stage 1: SciPy's solver, exactly as a user would call it ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            x_sp, info = newton(fc, x0, fprime=fprime, maxiter=maxiter,
                                full_output=True, disp=False, **newton_kwargs)
            converged = bool(info.converged)
        except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
            x_sp, converged = math.nan, False
    x_sp = float(x_sp)
    scipy_nfev = fc.calls

    ok, res, step = _verify(fc, fprime, x_sp, ftol, xtol)
    scipy_res = res
    false_conv = converged and not ok
    common = dict(scipy_root=x_sp, scipy_converged=converged,
                  scipy_residual=scipy_res, false_convergence=false_conv,
                  scipy_nfev=scipy_nfev)

    if ok:
        return RootReport(root=x_sp, status=RELIABLE, reliable=True,
                          method="scipy.newton", residual=res, step_estimate=step,
                          bracket=None, nfev=fc.calls,
                          message="SciPy's result passed the residual and step checks.",
                          **common)

    notes = []
    if false_conv:
        notes.append("SciPy reported convergence but the result failed verification "
                     "(the step-size stopping criterion was misleading).")
    elif not converged:
        notes.append("SciPy did not converge.")

    # ---- Stage 2: find or validate a bracket ----
    if bracket is not None:
        a, b = map(float, bracket)
        fa, fb = _safe_eval(fc, a), _safe_eval(fc, b)
        if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0):
            notes.append(f"Supplied bracket [{a}, {b}] is invalid: f(a) and f(b) "
                         "have the same sign.")
            bracket = None
    if bracket is None and auto_bracket:
        bracket = find_bracket(fc, x0)
        if bracket is not None:
            notes.append(f"Automatic search found a sign change on {bracket}.")

    if bracket is None:
        return RootReport(root=None, status=NO_ROOT_FOUND, reliable=False,
                          method="none", residual=scipy_res, step_estimate=step,
                          bracket=None, nfev=fc.calls,
                          message="No verified root: SciPy's answer failed verification "
                                  "and no sign-change bracket was found. The function "
                                  "may have no real root near x0.",
                          notes=notes, **common)

    # ---- Stage 3: Brent's method fallback ----
    a, b = bracket
    if a == b:
        x_br = a
    else:
        x_br = brentq(fc, a, b, xtol=1e-14, rtol=4 * np.finfo(float).eps,
                      maxiter=200)
    ok, res, step = _verify(fc, fprime, x_br, ftol, xtol)
    if ok:
        return RootReport(root=x_br, status=RECOVERED, reliable=True,
                          method="scipy.brentq", residual=res, step_estimate=step,
                          bracket=(a, b), nfev=fc.calls,
                          message="Recovered a verified root with Brent's method.",
                          notes=notes, **common)
    fa, fb = abs(_safe_eval(fc, a)), abs(_safe_eval(fc, b))
    if res < min(fa, fb):
        why = ("|f| shrinks toward this point, so it is probably a root where f is "
               "very steep (ill-conditioned residual, e.g. cbrt(x)); loosen ftol "
               "if that is acceptable for your problem.")
    else:
        why = ("|f| grows toward this point, so the sign change is probably a pole "
               "or discontinuity, not a root.")
    return RootReport(root=None, status=UNRELIABLE, reliable=False,
                      method="scipy.brentq", residual=res, step_estimate=step,
                      bracket=(a, b), nfev=fc.calls,
                      message=f"Brent's method converged to x={x_br:.6g}, but "
                              f"|f| = {res:.2e} > ftol. {why}",
                      notes=notes, **common)


if __name__ == "__main__":
    def issue(x):
        return x**4 - x**2 + 1

    print("SciPy issue #6379: f(x) = x^4 - x^2 + 1, x0 = 0.001\n")
    print(reliable_root(issue, 0.001))

    print("\nRecovery example: f(x) = x^3 - 2x + 2, x0 = 0 (Newton cycles 0 -> 1 -> 0)\n")
    print(reliable_root(lambda x: x**3 - 2*x + 2, 0.0, fprime=lambda x: 3*x**2 - 2))
